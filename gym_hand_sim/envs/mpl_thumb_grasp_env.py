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
        self, model_path, n_targets, target_body, target_position, target_rotation,
        target_position_range, reward_type, initial_qpos=None,
        randomize_initial_position=True, randomize_initial_rotation=True,
        distance_threshold=0.01, n_substeps=10, control_mode=False
    ):
        """
        Initializes a new Hand manipulation environment.
        Args:
            model_path (string): path to the environments XML file
            n_targets: number of possible targets
            target_body: base name for all targets
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
            control_mode: type of control:
                - simulated: For training RL agent, no user input
                - mixed: For operating alongside agent
                - tracked: For full hand tracking
        """
        self.n_targets = n_targets
        self.target_body_template = target_body
        self.target_body = self.target_body_template.replace('_', str(1))
        self.cur_target = 1
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
        self.control_mode = control_mode

        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel']
        initial_qpos = initial_qpos or {}

        mpl_env.MPLEnv.__init__(
            self, model_path=model_path, initial_qpos=initial_qpos, n_actions=5, 
            n_substeps=n_substeps)

    def _get_achieved_qpos(self):
        # get target object transform
        object_qpos = self.sim.data.get_joint_qpos(self.target_body)
        assert object_qpos.shape == (7,)
        return object_qpos

    def compute_reward(self, action):
        reward = 0.
        #cost = -1. * self.control_cost(action)
        cost = -0.1 * np.linalg.norm(self.sim.data.get_joint_qvel(self.target_body))

        reward = reward + cost

        if self.reward_type == 'sparse':
            lifted, dropped = self._is_on_ground()
            if not lifted:
                reward += 0
            if self.off_ground_count >= 2:
                reward = +1
            if dropped:
                reward += -0
            if self._is_done():
                reward += -10.
            return reward
        else:
            raise NotImplementedError()

    def _is_done(self):
        qpos = self.sim.data.get_joint_qpos(self.target_body)
        if self.control_mode == 'simulated':
            if ((np.abs(qpos[0] - (self.sim.data.mocap_pos[0][0] - 0.00)) > 0.06) or 
                (np.abs(qpos[1] - (self.sim.data.mocap_pos[0][1] + 0.17)) > 0.07) or
                (np.abs(qpos[2] - self.sim.data.mocap_pos[0][2]) > 0.12)):
                '''if (np.abs(qpos[0] - (self.sim.data.mocap_pos[0][0] + 0.04)) > 0.06):
                    print("x fail")
                if (np.abs(qpos[1] - (self.sim.data.mocap_pos[0][1] + 0.17)) > 0.06):
                    print("y fail")
                if (np.abs(qpos[2] - self.sim.data.mocap_pos[0][2]) > 0.07):
                    print("z fail")'''
                return True
        else:
            return False

    def _is_on_ground(self):
        lifted = (self.sim.data.get_joint_qpos(self.target_body)[2] - 0.0) > (0.042 + self.distance_threshold)
        on_ground = (self.sim.data.get_joint_qpos(self.target_body)[2] - 0.0) < (0.0406)
        dropped = False

        if lifted:
            if (self.sim.data.sensordata[-19] + self.sim.data.sensordata[-18]) > 0.1:
                pass
            self.off_ground_count += 1
        elif (self.off_ground_count > 0) and on_ground:
            dropped = True
            self.off_ground_count = 0
        return lifted, dropped

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

    def _reset_sim(self):
        i = random.uniform(0, 1)
        if i > 0.30:
            if i > 0.65:
                self.cur_target = random.randrange(21, 30)
            else:
                self.cur_target = random.randrange(11, 20)
        else:
            self.cur_target = random.randrange(1, 10)

        self.target_body = self.target_body_template.replace('_', str(self.cur_target))
        self.off_ground_count = 0
        self.t = 0
        self.sim.set_state(self.initial_state)
        self.sim.forward()

        self.sim.data.ctrl[:] = 0.

        initial_qpos = self.sim.data.get_joint_qpos(self.target_body).copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
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
                angle = self.np_random.uniform(-np.pi/10, np.pi/10)
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
            elif self.target_rotation == 'fixed':
                pass
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

            mocap_angle_y = self.np_random.uniform(-np.pi/30., np.pi/30.)
            mocap_axis_y = np.array([0., 1., 0.])
            mocap_offset_quat_y = quat_from_angle_and_axis(mocap_angle_y, mocap_axis_y)
            initial_mocap_quat = rotations.quat_mul(initial_mocap_quat, mocap_offset_quat_y)

        # Randomize initial position.
        if self.randomize_initial_position:
            if self.target_position != 'fixed':
                initial_pos += np.array([np.random.uniform(self.target_position_range[0][0], self.target_position_range[0][1]), 
                                        np.random.uniform(self.target_position_range[1][0], self.target_position_range[1][1]), 
                                        np.random.uniform(self.target_position_range[2][0], self.target_position_range[2][1])])

        self.init_object_pos = initial_pos
        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos(self.target_body, initial_qpos)

        # set mocap body
        self.sim.data.mocap_pos[:] = self.init_object_pos + np.array([0.00,-0.17,0.04])
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
        '''Apply to thumb ABD, MCP, wrist PRO, ............TODO
           thumb PIP follows MCP
        '''


        ctrl_idx = [2, 3, 4, 12, 7]
        follow_idx = [(2, 5, 0.5), (4, 11, 1.)] # actuator no. 5 follows action[2] with multiplier 0.5.. etc
        assert action.shape == (self.n_actions,)


        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.
        actuation_range[2] = actuation_range[2] / 1.8
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.

        if self.control_mode == 'simulated':
            self.sim.data.ctrl[:] = 0.

            if self.t > 7:
                self.sim.data.mocap_pos[:] = (self.init_object_pos + 
                                    np.array([0.00, -0.17 - (self.t-5)/500, 0.04 + (self.t-5)/500]) 
                                    + self.np_random.normal(size=3, scale=0.001))

            # simulate user grasp on digits
            self.sim.data.ctrl[8:11] = min(0.2 + self.t/10., 0.85)

            if (self.t > 2):
                for j, idx in enumerate(ctrl_idx):
                    self.sim.data.ctrl[idx] = actuation_center[idx] + action[j] * actuation_range[idx]
                    self.sim.data.ctrl[idx] = np.clip(self.sim.data.ctrl[idx], ctrlrange[idx][0], ctrlrange[idx][1])

                for follow in follow_idx:
                    self.sim.data.ctrl[follow[1]] = actuation_center[follow[1]] + action[follow[0]] * actuation_range[follow[1]] * follow[2]

            self.sim.data.ctrl[:] += self.np_random.normal(size=self.sim.data.ctrl.size, scale=0.001)

        elif self.control_mode == 'mixed':

            #self.sim.data.ctrl[:3] = 0.
            #self.sim.data.ctrl[3] = 0.45
            #self.sim.data.ctrl[4] = 0.3
            #self.sim.data.ctrl[5] = 0.2
            self.sim.data.ctrl[6] = 0.2
            #self.sim.data.ctrl[11:] = 0.25

            if (np.max(self.sim.data.ctrl[8:10]) > 0.8):

                for j, idx in enumerate(ctrl_idx):
                    self.sim.data.ctrl[idx] = actuation_center[idx] + action[j] * actuation_range[idx]
                    self.sim.data.ctrl[idx] = np.clip(self.sim.data.ctrl[idx], ctrlrange[idx][0], ctrlrange[idx][1])

                for follow in follow_idx:
                    self.sim.data.ctrl[follow[1]] = actuation_center[follow[1]] + action[follow[0]] * actuation_range[follow[1]] * follow[2]

            self.sim.data.ctrl[2] = 0.

        elif self.control_mode == 'tracked':
            pass

    def _get_obs(self):
        palm = [self.sim.data.sensordata[-19], self.sim.data.sensordata[-18]]
        fingers = np.take(self.sim.data.sensordata, [-14, -10, -7, -4])
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        robot_qpos = np.delete(robot_qpos, [0,1]) # ignore fixed joints: Wrist UDEV+PRO
        robot_qvel = np.delete(robot_qvel, [0,1])
        object_qvel = self.sim.data.get_joint_qvel(self.target_body)
        object_pos = self._get_achieved_qpos().ravel()[:3]  # this contains the object position + rotation
        mocap_pos = self.sim.data.mocap_pos.ravel()
        delta = object_pos - mocap_pos

        observation = np.concatenate([palm, fingers, robot_qpos, robot_qvel, np.zeros(delta.size), np.zeros(object_qvel.size)])
        observation += self.np_random.normal(size=observation.size, scale=0.005)

        return {
            'observation': observation.copy()
        }

    def _step_callback(self):

        self.t += 1
        return

    def step(self, action):
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = self._is_done()
        info = {
            'episode_done': self._is_done()}
        reward = self.compute_reward(action)

        return obs, reward, done, info



class MPLThumbGraspTrainEnv(MPLThumbGraspEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='z', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        MPLThumbGraspEnv.__init__(self, 
            model_path=THUMB_GRASP_XML, n_targets=30, target_body='obj_:joint',
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.06, 0.06), (-0.03, 0.03), (0., 0.)]),
            reward_type=reward_type, 
            control_mode='simulated'
            )

class MPLThumbGraspOpEnv(MPLThumbGraspEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='z', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        MPLThumbGraspEnv.__init__(self, 
            model_path=THUMB_GRASP_XML, n_targets=30, target_body='obj_:joint',
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.02, 0.02), (0., 0.)]),
            reward_type=reward_type, 
            control_mode='mixed'
            )

class MPLThumbGraspTrackEnv(MPLThumbGraspEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='z', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        MPLThumbGraspEnv.__init__(self, 
            model_path=THUMB_GRASP_XML, n_targets=30, target_body='obj_:joint',
            target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.02, 0.02), (0., 0.)]),
            reward_type=reward_type, 
            control_mode='tracked'
            )