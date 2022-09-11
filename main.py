import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt


#setup env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#wrap in grayscale
env = GrayScaleObservation(env, keep_dim=True)

#wrap in dummy env
env = DummyVecEnv([lambda: env])

#stack frames
env = VecFrameStack(env, 4, channels_order='last')


