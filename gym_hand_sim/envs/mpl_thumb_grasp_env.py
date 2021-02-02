import os
import numpy as np
import random

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
        distance_threshold=0.004, n_substeps=20,
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
            distance_threshold (float, in meters): the threshold + ball radius determines wheter ball is off ground
            n_substeps (int): number of substeps the simulation runs on every call to step
            relative_control (boolean): whether or not the hand is actuated in absolute joint positions or relative to the current state
        """
        self.target_body = target_body
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range        
        self.reward_type = reward_type
        self.randomize_initial_rotation = randomize_initial_rotation
        self.randomize_initial_position = randomize_initial_position
        self.distance_threshold = distance_threshold
        self.init_object_pos = None
        self.off_ground_count = 0
        self.t = 0

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']
        initial_qpos = initial_qpos or {}

        mpl_env.MPLEnv.__init__(
            self, model_path=model_path, initial_qpos=initial_qpos, n_actions=4, 
            n_substeps=n_substeps)

    def _get_achieved_qpos(self):
        # get target object transform
        object_qpos = self.sim.data.get_joint_qpos(self.target_body)
        assert object_qpos.shape == (7,)
        return object_qpos

    def compute_reward(self, action):
        reward = 0.
        cost = -1. * self.control_cost(action)
        reward = reward + cost

        if self.reward_type == 'sparse':
            if self._is_on_ground():
                reward += 0
            if self.off_ground_count >= 2:
                reward += 5
            if self._is_done():
                reward += -100
            return (reward)
        else:
            raise NotImplementedError()

    def _is_done(self):
        return (np.linalg.norm(self.init_object_pos[:2] - self.sim.data.get_joint_qpos(self.target_body)[:2]) >= 0.1)

    def _is_on_ground(self):
        on_ground = (self.sim.data.get_joint_qpos(self.target_body)[2] - 0.0) <= (0.04 + self.distance_threshold)
        if not on_ground:
            self.off_ground_count += 1
        else:
            self.off_ground_count = 0
        return on_ground

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        self.off_ground_count = 0
        self.t = 0
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos(self.target_body).copy()
        initial_pos, initial_quat = np.zeros(3), initial_qpos[3:]
        initial_mocap_pos = self.sim.data.mocap_pos[0]
        initial_mocap_quat = np.array([0,0,0,1])

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

                mocap_angle = self.np_random.uniform(-np.pi/30., np.pi/30.)
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
                initial_pos += np.append(self.np_random.normal(size=2, scale=0.005), 0)

        self.init_object_pos = initial_pos
        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos(self.target_body, initial_qpos)

        # set mocap body
        self.sim.data.mocap_pos[:] = np.array([0.03,-0.17,0.100])
        self.sim.data.mocap_quat[:] = initial_mocap_quat.copy()

        # Run the simulation for a bunch of timesteps to let everything settle in.
        for _ in range(10):
            self._set_action(np.zeros(self.n_actions))
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        return True

    def _render_callback(self):
        self.sim.forward()

    def _set_action(self, action):
        '''Apply to thumb ABD, MCP, DIP, wrist PRO 
           thumb PIP follows MCP
        '''
        ctrl_idx = [2, 3, 4, 6]
        follow_idx = (2, 5) # actuator no. 5 follows action[2]
        # simulate user grasp on digits
        self.sim.data.ctrl[8:11] = min(0.2 + self.t/50., 0.6) + random.randrange(-1,1) * 0.01
        self.sim.data.ctrl[12] = min(.2 + self.t/50., 0.6) + random.randrange(-1,1) * 0.01
        self.sim.data.ctrl[7] = 0.2 + random.randrange(-1,1) * 0.01
        self.sim.data.ctrl[11] = 0.2 + random.randrange(-1,1) * 0.01

        assert action.shape == (self.n_actions,)
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

        #action = (action - 5) / 5.

        for j, idx in enumerate(ctrl_idx):
            self.sim.data.ctrl[idx] = actuation_center[idx] + action[j] * actuation_range[idx]
            self.sim.data.ctrl[idx] = np.clip(self.sim.data.ctrl[idx], ctrlrange[idx][0], ctrlrange[idx][1])

        self.sim.data.ctrl[follow_idx[1]] = actuation_center[follow_idx[1]] + action[follow_idx[0]] * actuation_range[follow_idx[1]] * 0.25

    def _get_obs(self):

        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        robot_qpos = np.delete(robot_qpos, [0, 1, 7, 14, 18]) # ignore fixed joints: Wrist UDEV+PRO, ABDs 
        robot_qvel = np.delete(robot_qvel, [1, 2, 7, 14, 18])
        object_qvel = self.sim.data.get_joint_qvel(self.target_body)
        object_transform = self._get_achieved_qpos().ravel()  # this contains the object position + rotation
        mocap_pos = self.sim.data.mocap_pos.ravel()
        mocap_quat = self.sim.data.mocap_quat.ravel()


        observation = np.concatenate([robot_qpos, robot_qvel, mocap_pos, mocap_quat, object_transform, object_qvel])

        return {
            'observation': observation.copy()
        }

    def _step_callback(self):

        self.t += 1
        return

    def step(self, action):
        action = np.clip(action, 0, 11)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = self._is_done()
        info = {
            'episode_done': self._is_done()
        }
        reward = self.compute_reward(action)

        return obs, reward, done, info



class MPLThumbGraspBallEnv(MPLThumbGraspEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='ignore', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        MPLThumbGraspEnv.__init__(self, 
            model_path=THUMB_GRASP_XML, target_body='obj1:joint',
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.02, 0.02), (0.15, 0.2)]),
            reward_type=reward_type
            )