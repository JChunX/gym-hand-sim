import os
import numpy as np

from gym import utils, error
from gym_hand_sim.envs import mpl_env
from gym.envs.robotics import rotations
from gym.envs.robotics.utils import robot_get_obs

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def quat_from_angle_and_axis(angle, axis):
    assert axis.shape == (3,)
    axis /= np.linalg.norm(axis)
    quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
    quat /= np.linalg.norm(quat)
    return quat

THUMB_GRASP_XML = os.path.join('MPL', 'MPL_Basic.xml')


class MPLThumbGraspEnv(mpl_env.MPLEnv):
    def __init__(
        self, model_path, target_body, target_position, target_rotation,
        target_position_range, reward_type, initial_qpos=None,
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.02, n_substeps=5, relative_control=False,
        ignore_z_target_rotation=False,
    ):
        """
        Initializes a new Hand manipulation environment.
        Args:
            model_path (string): path to the environments XML file
            target_body: the target body name in the XML file
            target_position (string): the type of target position:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the object
                - random: target position is fully randomized according to target_position_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y

            target_position_range (np.array of shape (3, 2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
            ignore_z_target_rotation (boolean): whether or not the Z axis of the target rotation is ignored
        """
        self.target_body = target_body
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range        
        self.reward_type = reward_type
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.ignore_z_target_rotation = ignore_z_target_rotation

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']
        initial_qpos = initial_qpos or {}

        mpl_env.MPLEnv.__init__(
            self, model_path=model_path, initial_qpos=initial_qpos, n_actions=4, 
            n_substeps=n_substeps, relative_control=relative_control)

    def _get_achieved_goal(self):
        # get goal object transform
        # perhaps not very helpful for grasping
        object_qpos = self.sim.data.get_joint_qpos(self.target_body)
        assert object_qpos.shape == (7,)
        return object_qpos

    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != 'ignore':
            delta_pos = goal_a[..., :3] - goal_b[..., :3]
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

            if self.ignore_z_target_rotation:
                # Special case: We want to ignore the Z component of the rotation.
                # This code here assumes Euler angles with xyz convention. We first transform
                # to euler, then set the Z component to be equal between the two, and finally
                # transform back into quaternions.
                euler_a = rotations.quat2euler(quat_a)
                euler_b = rotations.quat2euler(quat_b)
                euler_a[2] = euler_b[2]
                quat_a = rotations.euler2quat(euler_a)

            # Subtract quaternions and extract angle between them.
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    def compute_reward(self, achieved_goal, goal, info):
        # TODO: will penaltizing high forces improve performance?
        if self.reward_type == 'sparse':
            on_ground = self._is_on_ground().astype(np.float32)
            return (-1. * on_ground)
        else:
            raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        return achieved_pos

    def _is_on_ground(self):
        return (self.sim.data.get_joint_qpos(self.target_body)[2] - 0.0) <= 0.0401

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos(self.target_body).copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        initial_mocap_pos = self.sim.data.mocap_pos[0]
        initial_mocap_quat = self.sim.data.mocap_quat[0]

        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)
        assert initial_mocap_pos.shape == (3,)
        assert initial_mocap_quat.shape == (4,)
        initial_qpos = None

        # Randomization initial rotation for target object and mocap body
        if self.randomize_initial_rotation:
            if self.target_rotation == 'z':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)

                mocap_angle = self.np_random.uniform(-np.pi/20., np.pi/20.)
                mocap_axis = np.array([0., 1., 0.])
                mocap_offset_quat = quat_from_angle_and_axis(mocap_angle, mocap_axis)
                initial_mocap_quat = rotations.quat_mul(initial_mocap_quat, mocap_offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += self.np_random.normal(size=3, scale=0.005)

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos(self.target_body, initial_qpos)

        # set mocap body
        self.sim.data.mocap_pos[:] = np.array([0.03,-0.18,0.11])
        self.sim.data.mocap_quat[:] = initial_mocap_quat.copy()

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(self.n_actions))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return True

    def _sample_goal(self):
        # Select a goal for the object position
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_joint_qpos(self.target_body)[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            target_pos = self.sim.data.get_joint_qpos(self.target_body)[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            target_quat = self.sim.data.get_joint_qpos(self.target_body)[3:]
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def _render_callback(self):
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == 'ignore':
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos("target:joint", goal)
        self.sim.data.set_joint_qvel("target:joint", np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _set_action(self, action):
        '''Apply to thumb ABD, MCP, PIP, and DIP joints
        '''
        assert action.shape == (self.n_actions,)
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.
        actuation_range = actuation_range[3:7]
        actuation_center = actuation_center[3:7]

        self.sim.data.ctrl[3:7] = actuation_center + action * actuation_range
        self.sim.data.ctrl[3:7] = np.clip(self.sim.data.ctrl[3:7], ctrlrange[3:7, 0], ctrlrange[3:7, 1])

    def _get_obs(self):

        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        robot_qpos = robot_qpos[3:] # ignore wrist joints
        robot_qvel = robot_qvel[3:]
        object_qvel = self.sim.data.get_joint_qvel(self.target_body)
        object_transform = self._get_achieved_goal().ravel()  # this contains the object position + rotation
        mocap_pos = self.sim.data.mocap_pos.ravel()
        mocap_quat = self.sim.data.mocap_quat.ravel()


        observation = np.concatenate([robot_qpos, robot_qvel, mocap_pos, mocap_quat, object_transform, object_qvel])

        return {
            'observation': observation.copy(),
            'achieved_goal': object_transform.copy(),
            'desired_goal': self.goal.ravel().copy(),
        }

    def _step_callback(self):
        '''
        Emulates hand user behavior during grasp
        Initiates a grasp with digits
        Moves the hand's mocap body in some trajectory

        '''
        pass



class MPLThumbGraspBallEnv(MPLThumbGraspEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='ignore', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        MPLThumbGraspEnv.__init__(self, 
            model_path=THUMB_GRASP_XML, target_body='obj1:joint',
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.15, 0.2)]),
            reward_type=reward_type
            )