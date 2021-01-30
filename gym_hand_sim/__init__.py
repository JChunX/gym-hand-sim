from gym.envs.registration import register

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='MplThumbGraspBall{}-v0'.format(suffix),
        entry_point='gym_hand_sim.envs:MPLThumbGraspBallEnv',
        kwargs=kwargs,
        max_episode_steps=100,
    )
