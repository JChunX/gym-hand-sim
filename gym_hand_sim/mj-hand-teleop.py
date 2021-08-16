#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
mj-hand-teleop: Entrypoint for teleoperation of mujoco models in Unity

Usage:
    python mj-hand-teleop.py [OPTIONS]

    The xml file should match the model used in the Unity Scene

Example:
    python .\mj-hand-teleop.py --xml 'C:\Users\xieji\repos\gym-hand-sim\gym_hand_sim\envs\assets\MPL\MPL_Boxes.xml' --mode manual
"""


from mjremote import mjremote
from pathlib import Path

import argparse, textwrap
import sys
import numpy as np 
import time
import mujoco_py

__version__ = 'v1.0.0'

def run_mujoco(args):
    print(args)

    xml_path = args.xml
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model, nsubsteps=4)

    remote = mjremote()
    remote.connect()

    while True:
        sim.reset()
        i = 0
        t0 = time.time()
        remote.movecamera(np.array([0,0,-0.05])) 
        while True:
            # mocap
            try:
                is_controller = remote.getOVRControlType()
            except ConnectionResetError:
                print('Remote connection terminated')
                return
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
                sim.data.ctrl[8:11] = grip
                sim.data.ctrl[12] = grip
            sim.step()
            # render
            qpos = sim.data.qpos

            remote.setqpos(qpos)

            i += 1
            if i % 100 == 0:
                t1 = time.time()
                print('FPS: ', 100/(t1-t0))
                t0 = t1

def parsed_arguments():

    description = textwrap.dedent("""\
        usage: mj-hand-teleop [-h] [--version]
                              --xml
                              --mode
        Opens up TCP/IP socket to communicate between python & Unity
        Starts mujoco simulation based on mode and xml file
        
        required arguments:
            --xml          XML      Which xml to use 
            --mode         MODE     Control mode [manual | rl]
        """)

    epilog = textwrap.dedent("""\
        example:
        python mj-hand-teleop.py --xml '~/envs/assets/MPL/MPL_Boxes.xml' --mode manual
        version:
          {}
        """.format(__version__))
    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    # Adding Verison information
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
    parser.add_argument('--xml', 
                        type=str,
                        required=True,
                        help='Path to model xml file')
    parser.add_argument('--mode',
                        type=str,
                        required=False,
                        default='manual',
                        help='Mode of teleoperation [manual | rl]')

    # Sanity check for user command line arguments 
    print(sys.argv)
    if len(sys.argv) < 5:
        parser.error("""\n\t └── Fatal: Not enough arguments""")
    parser.set_defaults(func = run_mujoco)

    # Parse command-line args
    args = parser.parse_args()
    return args

def main():
    # Display version information
    if '--version' not in sys.argv:
        print('Entrypoint for teleoperation of mujoco models in Unity (mj-hand-teleop {})'.format(__version__))

    # Collect args for sub-command
    args = parsed_arguments()

    # Mediator method to call sub-command's set handler function
    args.func(args)


if __name__ == '__main__':
    main()