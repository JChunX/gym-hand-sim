from gym.envs.registration import register



register(
    id='MplThumbGraspBall-v0',
    entry_point='gym_hand_sim.envs:MPLThumbGraspBallEnv',
    max_episode_steps=100,
)
