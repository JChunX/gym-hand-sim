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


    cap = True
    xml_path = os.path.join(model_dir, model_xml)
    model = mujoco_py.load_model_from_path(xml_path)
    env = gym.make('gym_hand_sim:MplThumbGraspTrack-v0').env

    if cap:
        remote = mjremote()
        result = remote.connect()

    while True:
        env.reset()
        i = 0
        t0 = time.time()
        returns = 0
        if cap:
            remote.movecamera(env.init_object_pos)
        done= False

        while True:
            if not cap:
                env.render()
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                returns += reward

                if done:
                    break
            else:

                # mocap
                is_controller = remote.getOVRControlType()
                grip, pos, quat = remote.getOVRControllerInput()
                hand_pose = remote.getOVRHandInput()

                env.sim.data.mocap_pos[:] = pos
                env.sim.data.mocap_quat[:] = quat
                remote.setmocap(pos, quat)

                if is_controller == 0:
                    for j, pose in enumerate(hand_pose):
                        env.sim.data.ctrl[3 + j] = pose
                else:
                    env.sim.data.ctrl[8:11] = grip
                    env.sim.data.ctrl[12] = grip

                action = env.action_space.sample()
                obs, reward, done, info = env.step(np.zeros(action.size))
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