import subprocess
from numpy.lib.function_base import average
from vcgencmd import Vcgencmd
from cpufreq import cpuFreq
import numpy as np
from EXEC_TIME import exec_time

class env():
    def __init__(self):
        self.cpu = cpuFreq()
        self.vcgm = Vcgencmd()
        self.old_action = 0
        self.speed = 0

        try:
            self.cpu.set_governors('userspace')
        except:
            print('UNABLE TO SET GOVERNOR')

        self.AVG_CPU_UTILS = np.zeros((10,))
        self.ind = 0
        self.done = False
    
    def temp_volt_utils(self):
        i = self.ind % 10
        cpu_utils = exec_time(2)
        self.AVG_CPU_UTILS[i] = cpu_utils  
        self.ind += 1
        return self.vcgm.measure_temp(), \
            self.vcgm.measure_volts('core'), \
                cpu_utils, sum(self.AVG_CPU_UTILS)/10

    def get_state(self):
        return np.array([*self.temp_volt_utils()], dtype=np.float32)

    def reward(self, state_info, f, speed):
        
        temp, util, avg_util = state_info[0], state_info[2], state_info[3]
        rew = 0
        self.done = False
        temp /= 20                    # TEMP (0 - 5)
        ratio = avg_util / (f+1)
        ratio_rew = -1.11 * ratio ** 2 + 22.22 * ratio - 101.1
        
        rew = ratio_rew - (0.5*np.exp(temp) + 0.5*speed)
        #print('rew', rew)
        if util < 30 and rew >= -10:
            self.done = True

        return rew
            
            
    def step(self, action):
        freq = self.cpu.available_frequencies     
        fre = freq[action]                                # action given to a function is in range(0,10). convert to corresponding frequency
        
        self.speed = abs(action - self.old_action)
        self.old_action = action
        
        try:
            self.cpu.set_frequencies(fre)
        except:
            print('UNABLE TO SET FREQ')
        
        next_ = self.get_state() 
        return np.reshape(next_, (1,4)), self.reward(next_, action, self.speed*5), self.done     


    
