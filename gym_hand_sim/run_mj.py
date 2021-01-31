import numpy as np 
import time
import random
import mujoco_py
import gym
import os

from mjremote import mjremote
from mujoco_py import MjSim
from pathlib import Path

import tensorflow as tf 

from tf_agents.environments import suite_mujoco

home = str(Path.home())
model_dir = os.path.join(home,'.mujoco/mjhaptix150/model/MPL')
model_xml = 'MPL_Handle.xml'

def main():


    cap = False
    xml_path = os.path.join(model_dir, model_xml)
    model = mujoco_py.load_model_from_path(xml_path)
    env = gym.make('gym_hand_sim:MplThumbGraspBall-v0')

    MAX_EPISODE_STEPS = 50

    num_iterations = 20000 # @param {type:"integer"}
    initial_collect_steps = 100  # @param {type:"integer"} 
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}
    batch_size = 64  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}
    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}


    if cap:
        remote = mjremote()
        result = remote.connect()

    done = 0

    while True:
        env.reset()
        i = 0
        t0 = time.time()
        while True:
            if not cap:
                env.render()
                obs, reward, done, info = env.step(env.action_space.sample())

                if info['episode_done'] or i > MAX_EPISODE_STEPS:
                    break
            else:
                # mocap
                grip, pos, quat = remote.getOVRinput()
                env.sim.data.mocap_pos[:] = pos
                env.sim.data.mocap_quat[:] = quat
                remote.setmocap(pos, quat)

                # actuation
                env.sim.data.ctrl[5] = 0.45
                env.sim.data.ctrl[8:11] = grip
                env.sim.data.ctrl[12] = grip

                # render
                qpos = env.sim.data.qpos
                remote.setqpos(qpos)

            i += 1
            if i % 1000 == 0:
                t1 = time.time()
                print('FPS: ', 1000/(t1-t0))
                t0 = t1


if __name__ == '__main__':
	main()