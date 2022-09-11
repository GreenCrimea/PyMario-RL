import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation
from stable_baseline3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt


#setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#run game and take random steps
'''
done = True
for step in range(100000):
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
env.close()
'''

#wrap in grayscale
env = GrayScaleObservation(env, keep_dim=True)



