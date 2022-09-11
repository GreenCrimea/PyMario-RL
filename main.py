import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

MODEL_NUM = 1

#PREPROCESS ENV
#setup env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

#wrap in grayscale
env = GrayScaleObservation(env, keep_dim=True)

#wrap in dummy env
env = DummyVecEnv([lambda: env])

#stack frames
env = VecFrameStack(env, 4, channels_order='last')


#SAVE MODEL
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(MODEL_NUM))
            self.model.save(model_path)

        return True

callback = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_DIR)


#IMPLEMENT RL MODEL

#new model
#model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

#load model
model = PPO.load('./train/best_model_1.zip')

#train model
#model.learn(total_timesteps=1000000, callback=callback)


#run model

state = env.reset()
while True:
    action, _state = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()


