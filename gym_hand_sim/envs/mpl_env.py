import numpy as np
from gym_hand_sim.envs import robot_env


class MPLEnv(robot_env.RobotEnv):

    def __init__(
        self, model_path, initial_qpos,
        n_actions, n_substeps
    ):
        self.n_actions = n_actions
        self._ctrl_cost_weight =  1e-3

        super().__init__(
            model_path=model_path,
            initial_qpos=initial_qpos,
            n_actions=n_actions,
            n_substeps=n_substeps
        )

    # RobotEnv methods
    # ----------------------------

    def render(self, mode='human',width=500,height=500):
        return super().render(mode, width, height)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

