import numpy as np
import pandas as pd
import time

from dateutil.parser import parse
from datetime import datetime
import http.client


class Env(object):
    timestamp = 0

    def __init__(self):
        self.action_space = ['Enlarge', 'Unchanged', 'Reduce']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self)
