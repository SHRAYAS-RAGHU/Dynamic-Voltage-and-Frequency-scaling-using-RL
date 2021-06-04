import subprocess
from vcgencmd import Vcgencmd
from cpufreq import cpuFreq
import numpy as np
from EXEC_TIME import exec_time

class env():
    def __init__(self):
        self.cpu = cpuFreq()
        self.vcgm = Vcgencmd()

        try:
            self.cpu.set_governors('userspace')
        except:
            print('UNABLE TO SET GOVERNOR')

        self.AVG_CPU_UTILS = np.zeros((10,))
        self.ind = 0
        self.done = False
    
    def temp_volt_utils(self):
        i = self.ind if self.ind < 10 else self.ind % 10
        cpu_utils = exec_time(3)
        self.AVG_CPU_UTILS[i] = cpu_utils  
        self.ind += 1
        return self.vcgm.measure_temp(), \
            self.vcgm.measure_volts('core'), \
                cpu_utils, sum(self.AVG_CPU_UTILS)/10

    def get_state(self):
        return np.array([*self.temp_volt_utils()], dtype=np.float32)

    def reward(self, state_info):
        pass
            
        return temp_rew + util_rew
            
            
    def step(self, action):
        freq = self.cpu.available_frequencies     
        action = freq[action]                                # action given to a function is in range(0,10). convert to corresponding frequency
        
        try:
            self.cpu.set_frequencies(action)
        except:
            print('UNABLE TO SET FREQ')
        
        next_ = self.get_state() 
        return np.reshape(next_, (1,4)), self.reward(next_), self.done         


    
