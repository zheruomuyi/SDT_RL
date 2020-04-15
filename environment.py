import numpy as np

LAST_POINT_ADJUST_TIME = 30000


class Adjust_env(object):

    def __init__(self):
        self.comp_dev = 0
        self.comp_std = 0
        self.last_comp_std = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0

    def reset(self):
        self.comp_dev = 0
        self.comp_proportion = 0
        self.last_comp_proportion = 0
        self.comp_std = 0
        self.last_comp_std = 0
        s = np.hstack([self.comp_dev, [self.comp_std, self.last_comp_std]])
        return s

    def step(self, action):
        self.comp_dev = self.comp_dev + action[0]
        if self.last_comp_proportion > 0.0 and self.last_comp_std > 0.0:
            reward = self.last_comp_proportion / self.comp_proportion - self.last_comp_std / self.comp_std
        else:
            reward = 0
        s = np.hstack([self.comp_dev, [self.comp_std, self.last_comp_std]])
        return s, reward

    def update(self, update):
        self.last_comp_proportion = self.comp_proportion
        self.last_comp_std = self.comp_std
        self.comp_proportion = update['comp_proportion']
        self.comp_std = update['comp_std']
        self.comp_dev = update['comp_dev']

    def getstate(self):
        s = np.hstack([self.comp_dev, [self.comp_std, self.last_comp_std]])
        return s
