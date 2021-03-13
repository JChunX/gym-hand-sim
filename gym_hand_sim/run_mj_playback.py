import numpy as np 
import time
import mujoco_py
import os
import gym

from mjremote import mjremote
from pathlib import Path
from scipy import io

home = str(Path.home())
model_dir = os.path.join(home, 'repos/gym-hand-sim/gym_hand_sim/envs/assets/MPL')
data_dir = os.path.join(home, 'iCloudDrive/Documents/Data/gym-hand-sim')
checkpoint_dir = os.path.join(home, '')
model_xml = 'MPL_Basic.xml'
data_file = 'NERData.mat'
freq = 100

def main():

    xml_path = os.path.join(model_dir, model_xml)
    model = mujoco_py.load_model_from_path(xml_path)
    NERData = io.loadmat(os.path.join(data_dir, data_file))['resultsmat'][0]

    env = gym.make('gym_hand_sim:MplThumbGraspOp-v0').env
    remote = mjremote()
    result = remote.connect()


    env.reset()
    i = 0
    j = 0
    t0 = time.time()
    returns = 0
    action = env.action_space.sample()

    remote.movecamera(env.init_object_pos)

    for trial_num in range(NERData.shape[0]):
        trial = np.radians(NERData[trial_num][2])
        angles = []
        j = 0
        while True:
            # mocap
            grip, pos, quat = remote.getOVRControllerInput()
            env.sim.data.mocap_pos[:] = pos
            env.sim.data.mocap_quat[:] = quat
            remote.setmocap(pos, quat)

            # actuation
            env.sim.data.ctrl[5] = 0.45
            env.sim.data.ctrl[8:11] = -trial[j][0]
            env.sim.data.ctrl[12] = -trial[j][0]

            obs, reward, done, info = env.step(np.zeros(action.size))
            angles.append(obs['observation'][12])

            if (j == trial.shape[0]-1):
                with open(os.path.join(data_dir, 'simangles{}.npy'.format(trial_num)), 'wb') as f:
                    np.save(f, np.array(angles))
                print('next replay..')
                break
            else:
                j += 1

            obs, reward, done, info = env.step(np.zeros(action.size))

            # render
            qpos = env.sim.data.qpos
            remote.setqpos(qpos)
            if done:
                break

            i += 1
            if i % 100 == 0:
                t1 = time.time()
                print('FPS: ', 100/(t1-t0))
                t0 = t1


if __name__ == '__main__':
    main()