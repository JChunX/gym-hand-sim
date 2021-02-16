from gym.envs.registration import register


register(
    id='MplThumbGraspTrain-v0',
    entry_point='gym_hand_sim.envs:MPLThumbGraspTrainEnv',
    max_episode_steps=100,
)

register(
    id='MplThumbGraspOp-v0',
    entry_point='gym_hand_sim.envs:MPLThumbGraspOpEnv',
    max_episode_steps=100,
)
register(
    id='MplThumbGraspTrack-v0',
    entry_point='gym_hand_sim.envs:MPLThumbGraspTrackEnv',
    max_episode_steps=100,
)
