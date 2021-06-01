from os import error
import random
import numpy as np

from ENV import env                                 # raspi 4 environment
agent = env()                                       # object for class env


EPISODE = 10
STATE = np.array(agent.reset())

for i in range(EPISODE):
    ACTION = 0
    next_state, rew = env.step(ACTION)

                              

