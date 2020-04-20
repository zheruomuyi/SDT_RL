import numpy as np

LAST_POINT_ADJUST_TIME = 30000


class Adjust_env(object):

    def __init__(self):
        self.comp_dev = 0
        self.comp_std = 0
        self.last_comp_std = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0
        self.s = np.array([0, 0, 0])

    def reset(self):
        self.comp_dev = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0
        self.comp_std = 0
        self.last_comp_std = 0
        self.s = np.hstack([0, 0, 0])
        return self.s

    def step(self, action):
        self.comp_dev = self.comp_dev + (0.125 * action[0] * self.comp_dev)
        reward = 0
        if self.last_comp_std < self.comp_std:
            reward = reward - 1
        else:
            reward = reward + 1
        if self.last_comp_proportion > self.comp_proportion:
            reward = reward - 1
        else:
            reward = reward + 1
        self.s = np.hstack([self.comp_dev, self.comp_proportion, self.comp_std])
        return self.s, reward

    def update(self, update):
        self.last_comp_proportion = self.comp_proportion
        self.last_comp_std = self.comp_std
        self.comp_proportion = update['comp_proportion']
        self.comp_std = update['comp_std']
        self.comp_dev = update['comp_dev']
        self.s = np.hstack([self.comp_dev, self.comp_proportion, self.comp_std])

    def getstate(self):
        return self.s
