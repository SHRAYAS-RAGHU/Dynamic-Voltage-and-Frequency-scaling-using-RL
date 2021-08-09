import numpy as np
from collections import defaultdict
import pandas as pd

from ENV import env                                                         # RASPI 4 CONTROL ENVIRONMENT
raspi_env = env() 
stat = defaultdict(list)

from AGENT import Agent
agent  = Agent(gamma=0.99, epsilon=1, lr = 0.00025, batch_size = 2)         # CREATING INSTANCE OF AGENT CLASS
EPISODE = 4
STATE = np.array(raspi_env.get_state())                                         # GRABBING THE INITIAL STATE

for i in range(EPISODE):
    ACTION = agent.choose_actions(STATE)
    NEXT_STATE, REW = raspi_env.step(ACTION)                                # TAKING THE ACTION
    agent.store_transitions(STATE, ACTION, REW, NEXT_STATE)           # STORING THE TRANSITIONS TO REPLAY MEMORY
    print(f'CURRENT_AVG_UTI : {STATE} \n AVG_UT : {NEXT_STATE} \n FREQ : {ACTION}  REW{REW}')
    agent.learn()

from cpufreq import cpuFreq
c = cpuFreq()

c.set_governors('ondemand')
print(c.get_governors())
"""
for i in range(100):
    a = raspi_env.get_state()                                   
    stat['FREQ'].append(raspi_env.cpu.get_frequencies()[0])
    stat['TEMP'].append(a[0])      
    stat['UTIL'].append(a[1])
    stat['AVG_UTIL'].append(a[2])
    stat['MEM_FREE'].append(a[3])
    stat['LOAD'].append(a[4])
D = pd.DataFrame(stat)
D.to_csv('new_state.csv')
"""
