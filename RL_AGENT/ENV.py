from vcgencmd import Vcgencmd
from cpufreq import cpuFreq
import numpy as np
from EXEC_TIME import exec_time
from CPU_LOAD import measure_load
from MEMORY import mem_free

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
    
    def temp_utils(self):
        i = self.ind % 10
        cpu_utils = exec_time(2) / 100
        self.AVG_CPU_UTILS[i] = cpu_utils  
        self.ind += 1
        
        return self.vcgm.measure_temp() / 80, \
                cpu_utils, sum(self.AVG_CPU_UTILS) / 10

    def get_state(self):
        """
        RETURNS :       [TEMP, UTILS, AVG-UTILS, FREE_MEM, AVG-LOAD-1-MIN]
        NORMALISED:     [0-80, 0-100, 0-100, , 200MB to 1.83GB, no.of process on 4 cpus]
        """
        self.load = float(measure_load())/4   # 4 CPUS
        self.mem_free = mem_free()
        return np.array([*self.temp_utils(), self.mem_free, self.load], dtype=np.float32)

    def reward(self, state_info, f, speed):
        # f : 1 to 10 for 600 - 1500 MHz here
        temp, util, avg_util, free_mem, load = state_info
        rew = 0
        K_T, K_S, K_U = 2, 0.5, 1

        rew_F_U = -abs(np.ceil(avg_util*10) - f)
        rew_F_M = 1 if -(f-11)/np.ceil(free_mem*10) == 1 else - 1
        rew_L = -util * load

        rew = -K_T * np.exp(temp) + -K_S * np.exp(speed) +\
                K_U * (rew_F_U + rew_F_M + rew_L)

        #print(f'\nREW : TEMP:{-K_T * np.exp(temp)}  SPEED:{-K_S * np.exp(speed)}  F_U:{0.7*rew_F_U} F_M:{0.3*rew_F_M } L:{rew_L}')
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
        return np.reshape(next_, (1,5)), self.reward(next_, action+1, self.speed/10) # add self.done if needed


    
