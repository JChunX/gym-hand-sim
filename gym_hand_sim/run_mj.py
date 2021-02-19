import numpy as np 
import time
import random
import mujoco_py
import gym
import os

from mjremote import mjremote
from pathlib import Path

home = str(Path.home())
model_dir = os.path.join(home,'repos/gym-hand-sim/gym_hand_sim/envs/assets/MPL')
checkpoint_dir = os.path.join(home, '')
model_xml = 'MPL_Basic.xml'

def main():


    cap = False
    xml_path = os.path.join(model_dir, model_xml)
    model = mujoco_py.load_model_from_path(xml_path)
    env = gym.make('gym_hand_sim:MplThumbGraspOp-v0').env

    if cap:
        remote = mjremote()
        result = remote.connect()

    while True:
        env.reset()
        i = 0
        t0 = time.time()
        returns = 0
        print(env.sim.data.mocap_pos)
        if cap:
            remote.movecamera(env.init_object_pos)

        while True:
            if not cap:
                env.render()
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                print(obs)
                returns += reward

                if done:
                    break
            else:
                if i % 1 == 0:
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                # mocap
                grip, pos, quat = remote.getOVRControllerInput()
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
                if done:
                    break

            i += 1
            if i % 1000 == 0:
                t1 = time.time()
                print('FPS: ', 1000/(t1-t0))
                t0 = t1


if __name__ == '__main__':
	main()