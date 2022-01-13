import numpy as np
import pandas as pd
import gym
from enum import Enum
import numpy as np

import sys
sys.path.append('..')
from util_tensorboard import DummyLogger, TensorboardLoggerSimple

ACTIONS = [-1, 1]  # DOWN, UP
DIRECTION_TO_ACTION = {
    -1: 0,
    +1: 1
}


class NLPEnv(gym.Env):
    def __init__(self, data_path="../BloombergNRG.csv", window_size=5, last_index=10, env_type="train", logger=DummyLogger(log_dir=None)):
        self.observation_space = gym.spaces.Box(low=np.array([-1] * window_size), high=np.array([1] * window_size))
        self.action_space = gym.spaces.Discrete(2)
        # self.reward_range = reward_range
        self.data_path = data_path
        self.window_size = window_size
        self.env_type = env_type
        self.total_reward_key = f"total_reward_{self.env_type}"
        self.accuracy_key = f"accuracy_{self.env_type}"

        self.data = None
        self.data_len = None
        self.t = None
        self.t_last = last_index
        self.logger = logger
        self.run_idx = 0
        self.total_reward_for_episode = None

    def _get_state(self, t):
        return self.data.iloc[t-self.window_size:t]["roberta_large_score"]

    def reset(self):
        self.run_idx += 1
        self.total_reward_for_episode = 0

        self.data = pd.read_csv(self.data_path, sep=";")

        self.data_len = len(self.data)
        self.t = self.window_size

        return self._get_state(self.t)

    def step(self, action):
        label = DIRECTION_TO_ACTION[self.data.iloc[self.t - 1]["direction"]]

        # predicted_label = ACTIONS[action]
        predicted_label = action

        if label == predicted_label:
            reward = 1
        else:
            reward = 0

        self.total_reward_for_episode += reward
        self.t += 1

        if self.t < self.t_last + 1 + 1:
            state = self._get_state(self.t)
            done = False
        else:
            state = None
            done = True

            acc = self.total_reward_for_episode / (self.t - self.window_size)
            self.logger.write_metadata(epoch=self.run_idx, key=self.total_reward_key, value=self.total_reward_for_episode)
            self.logger.write_metadata(epoch=self.run_idx, key=self.accuracy_key, value=acc)

        info = {}

        return state, reward, done, info

    def render(self, mode='human'):
        pass


if __name__ == "__main__":
    import random

    env = NLPEnv(data_path="../../BloombergNRG.csv")

    state = env.reset()

    for i in range(1000):
        action = random.randint(0, 1)

        print("state", state)
        print("action", action)

        state, reward, done, info = env.step(action)

        print("reward", reward)

        if done:
            print("Done")
            
            state = env.reset()
            break
