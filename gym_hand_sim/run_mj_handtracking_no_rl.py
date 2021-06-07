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
model_xml = 'MPL_Spherical.xml'

def main():


    cap = True
    xml_path = os.path.join(model_dir, model_xml)
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model, nsubsteps=4)

    if cap:
        remote = mjremote()
        result = remote.connect()

    while True:
        sim.reset()
        i = 0
        t0 = time.time()
        remote.movecamera(np.array([0,0,-0.05])) 
        while True:
            # mocap

            is_controller = remote.getOVRControlType()
            grip, pos, quat = remote.getOVRControllerInput()
            hand_pose = remote.getOVRHandInput()

            sim.data.mocap_pos[:] = pos
            sim.data.mocap_quat[:] = quat
            remote.setmocap(pos, quat)

            if is_controller == 0:
                for j, pose in enumerate(hand_pose):
                    sim.data.ctrl[3 + j] = pose
            else:
                sim.data.ctrl[3] = 1.4
                sim.data.ctrl[4] = 0.4 + grip / 2.5
                sim.data.ctrl[5] = 0.5 + grip / 3.0
                sim.data.ctrl[8:11] = grip #0.6 + grip
                sim.data.ctrl[12] = grip#0.6 + grip
            sim.step()
            # render
            qpos = sim.data.qpos

            remote.setqpos(qpos)


            i += 1
            if i % 100 == 0:
                t1 = time.time()
                print('FPS: ', 100/(t1-t0))
                t0 = t1


if __name__ == '__main__':
	main()
