import numpy

LAST_POINT_ADJUST_TIME = 30000


class Adjust_env(object):

    def __init__(self, comp_dev, comp_std, comp_proportion):
        self.comp_dev = comp_dev
        self.comp_std = comp_std
        self.comp_proportion = comp_proportion

    def reset(self):
        self.comp_dev = 0
        self.comp_proportion = 0
        self.comp_std = 0
        return self

    def step(self, action):
        self.comp_dev = self.comp_dev + action["change"]
        reward = action['comp_proportion'] / self.comp_proportion - action['comp_std'] / self.comp_std
        done = 'false'
        return self.comp_dev, reward, false,

    def render(self):
        return self.comp_dev, self.comp_proportion, self.comp_std
