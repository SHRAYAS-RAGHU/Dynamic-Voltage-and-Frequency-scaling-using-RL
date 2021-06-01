import subprocess
from vcgencmd import Vcgencmd
import time
vcgm = Vcgencmd()
from cpufreq import cpuFreq

class env:
    def __init__(self):
        self.cpu = cpuFreq()
        self.cpu.set_governors('userspace')

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
    
    def state_values(self):
        temp = vcgm.measure_temp()
        volt = vcgm.measure_volts('core')
        cpu_utils = env.cpu_utilisation()
    
    def step(self, action):
        try:
            self.cpu.set_frequencies(action)
        except:
            print('UNABLE TO SET USERSPACE GOVERNOR')

    def reward(self):
        pass

    @staticmethod
    def loop(x):
        start = time.time()
        b = sorted(x)
        for i in range(1000):
            for i in range(10000):
                pass
        end =  time.time()
        return f'{(end - start):.4f}'

    