from os import error
import numpy as np

from ENV import env                                                         # RASPI 4 CONTROL ENVIRONMENT
raspi_env = env()                                                           # INSTANCE OF CLASS ENVIRONMENT

from AGENT import Agent
agent  = Agent(gamma=0.99, epsilon=0.2, lr = 0.001, batch_size = 4)         # CREATING INSTANCE OF AGENT CLASS
EPISODE = 10
STATE = np.array(raspi_env.reset())                                         # GRABBING THE INITIAL STATE

for i in range(EPISODE):
    ACTION = agent.choose_actions(STATE)
    NEXT_STATE, REW, DONE = env.step(ACTION)                                      # TAKING THE ACTION
    agent.store_transitions(STATE, ACTION, REW, NEXT_STATE, DONE)                 # STORING THE TRANSITIONS TO REPLAY MEMORY
    agent.learn()


