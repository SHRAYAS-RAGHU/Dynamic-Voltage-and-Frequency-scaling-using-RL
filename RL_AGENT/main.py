from os import error
import random
import numpy as np

a = list(range(100000))
random.shuffle(a)

from state import state_values
AVG_CPU_UTILS = np.zeros((10,))
EPISODE = 10

for i in range(EPISODE):
    ind = i if i <10 else i % 10
    STATE = np.array(state_values())
    AVG_CPU_UTILS[ind] = STATE[2]
    STATE = np.append(STATE, sum(AVG_CPU_UTILS)/10)
    
    ACTION = 0
cpu.set_governors('ondemand')
                              

