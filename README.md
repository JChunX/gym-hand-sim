# gym-hand-sim
OpenAI gym environment for the Mujoco MPL hand

![rlgrasp](https://user-images.githubusercontent.com/59701038/129583739-9f472458-c944-4e0a-9d14-2b387a1def5d.gif)

## Usage
```mj-hand-teleop.py``` is the main entrypoint for VR teleoperation

```train_hand.py``` contains scripts for training a precision grasp task using RL (PPO) 

## Dependencies:
[OpenAI gym](https://gym.openai.com/)

[hand-sim](https://github.com/JChunX/hand-sim)

## Installation:
```
$ pip install gym
$ cd gym-hand-sim
$ pip install -e .
```
