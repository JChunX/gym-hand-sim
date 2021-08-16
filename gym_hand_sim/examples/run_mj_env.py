from pathlib import Path

"""
Example script for running RL agent
"""

import gym
import os


home = str(Path.home())
model_dir = os.path.join(home,'repos/gym-hand-sim/gym_hand_sim/envs/assets/MPL')
checkpoint_dir = os.path.join(home, '')
model_xml = 'MPL_Boxes.xml'

def main():

    env = gym.make('gym_hand_sim:MplGraspTrain-v0').env

    while True:
        env.reset()
        returns = 0

        while True:
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            returns += reward

            if done:
                break


if __name__ == '__main__':
	main()