from vcgencmd import Vcgencmd
import time
import random
import subprocess
from cpufreq import cpuFreq
cpu = cpuFreq()
import pandas as pd

vcgm = Vcgencmd()

a = list(range(100000))
random.shuffle(a)

def loop(x):
    start = time.time()
    b = sorted(x)
    for i in range(1000):
        for i in range(10000):
            pass
    end =  time.time()
    return f'{(end - start):.4f}'

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

data = {'GOVERNOR':['OD'], 'cpu_freq': [cpu.get_frequencies()[0]/1000], 'cpu_temp': [vcgm.measure_temp()], 'cpu_volt': [vcgm.measure_volts('core')], 'cpu_utils': [cpu_utilisation()], 'exec_time': [loop(a)]}

cpu.set_governors('userspace')
avail_freq = cpu.available_frequencies
for fre in avail_freq:
    try:
        cpu.set_frequencies(fre)
    except:
        print('INCORRECT FREQ')
    e_t = loop(a)
    data['GOVERNOR'].append('US')
    data['cpu_freq'].append(fre//1000)
    data['cpu_temp'].append(vcgm.measure_temp())
    data['cpu_volt'].append(vcgm.measure_volts('core'))
    data['cpu_utils'].append(cpu_utilisation())                              
    data['exec_time'].append(e_t)

data_F = pd.DataFrame(data)
data_F.to_csv('rew_stats.csv')
print(data_F)


                              

