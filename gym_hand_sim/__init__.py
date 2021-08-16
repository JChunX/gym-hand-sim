from gym.envs.registration import register


register(
    id='MplGraspTrain-v0',
    entry_point='gym_hand_sim.envs:MPLGraspTrainEnv',
    max_episode_steps=100,
)

register(
    id='MplGraspOp-v0',
    entry_point='gym_hand_sim.envs:MPLGraspOpEnv',
    max_episode_steps=100,
)
register(
    id='MplGraspTrack-v0',
    entry_point='gym_hand_sim.envs:MPLGraspTrackEnv',
    max_episode_steps=100,
)
