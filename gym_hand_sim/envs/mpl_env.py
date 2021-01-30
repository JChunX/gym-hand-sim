import numpy as np
from gym_hand_sim.envs import robot_env



class MPLEnv(robot_env.RobotEnv):

    def __init__(
        self, model_path, initial_qpos,
        n_actions, n_substeps, relative_control
    ):
        self.n_actions = n_actions

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
