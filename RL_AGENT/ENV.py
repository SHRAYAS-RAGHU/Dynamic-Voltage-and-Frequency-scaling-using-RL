import subprocess
from vcgencmd import Vcgencmd
from cpufreq import cpuFreq
import numpy as np

class env:
    def __init__(self):
        self.cpu = cpuFreq()
        self.vcgm = Vcgencmd()
        try:
            self.cpu.set_governors('userspace')
        except:
            print('UNABLE TO SET GOVERNOR')
        self.AVG_CPU_UTILS = np.zeros((10,))
        self.ind = 0

    @staticmethod
    def cpu_utilisation():
        io_c = subprocess.Popen(['iostat', '-c'], \
                                stdout=subprocess.PIPE, \
                                shell=False, stderr=subprocess.PIPE)
        s = io_c.communicate()[0]                                       # OUTPUT OF COMMAND STORED IN 'S' AS BYTES

        s = str(s)                                                      # CONVERTING BYTE DATA TO STRING FOR PROCESSING
        s = s.split('\\n')[3]                                           # OBTAINING CPU USAGE STATS ALONE AFTER SPLITTING SEPARATE LINES INTO LISTS
        s = s.strip(' ')                                                # STRIPPING INITIAL SPACES TO OBTAIN THE SET OF 5 NOS.
        s = s.split('   ')                                              # CONVERTING SET OF 5 VALUES INTO A LIST BBY SPLITTING
        cpu_usage = float(s[0]) + float(s[2])                           # CPU USAGE IS THE FIRST ELEMENT OF THE LIST 
        return cpu_usage
    
    def reset(self):
        temp = self.vcgm.measure_temp()
        volt = self.vcgm.measure_volts('core')
        cpu_utils = env.cpu_utilisation()
        return [temp, volt, cpu_utils, 0]
    
    def next_state(self):
        i = self.ind if self.ind < 10 else self.ind % 10
        temp = self.vcgm.measure_temp()
        volt = self.vcgm.measure_volts('core')
        cpu_utils = env.cpu_utilisation()
        self.AVG_CPU_UTILS[i] = cpu_utils  
        self.ind += 1
        return [temp, volt, cpu_utils, sum(self.AVG_CPU_UTILS)/10]

    def reward(self, state_info):
        if state_info[0] < 40:
            temp_rew = 1
        elif 40 < state_info[0] < 70:
            temp_rew = -state_info[0] // 20
        elif 70 < state_info[0] < 80:
            temp_rew = -20
        else:
            temp_rew = -100
            
        if state_info[2] < 30:
            util_rew = -10
        if 30 < state_info[2] < 60:
            util_rew = 1
        if 60 < state_info[2] < 80:
            util_rew = state_info[2] // 20
        if state_info[2] < 30:
            util_rew = -50
            
        return temp_rew + util_rew
            
            
    def step(self, action):
        freq = self.cpu.available_frequencies     
        action = freq.index(action)                                # action given to a function is in range(0,10). convert to corresponding frequency
        try:
            self.cpu.set_frequencies(action)
        except:
            print('UNABLE TO SET FREQ')
        done = False
        next_ = env.next_state() 
        return next_, env.reward(next_), done         


    
