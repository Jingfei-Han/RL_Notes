#encoding:utf-8
import numpy as np

class Bandit(object):
    def __init__(self, k=10, stationary=True):
        self.k = k
        self.stationary = stationary
        if self.stationary:
            self.values = list(np.random.randn(self.k))
        else:
            self.values = list(np.zeros(self.k)) #初始化为0

    def step(self, action):
        reward = np.random.randn() + self.values[action]

        if not self.stationary:
            self.values = list(np.array(self.values) + np.random.randn(self.k) * 0.01)
        return reward

    def get_optimal_action(self):
        return np.argmax(self.values)

