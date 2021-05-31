import random
import time
from cpufreq import cpuFreq
cpu = cpuFreq()

def loop():
    start = time.time()
    a = list(range(100000))
    random.shuffle(a)
    a = sorted(a)
    for i in range(100000):
        pass
    end =  time.time()
    return f'{(end - start):.4f}'
print('Ondemand', loop())

cpu.set_governors('userspace')
avail_freq = cpu.available_frequencies
freq_vs_exe_table = {'FREQ': [], 'EXEC_TIME':[]}
for fre in avail_freq:
    try:
        cpu.set_frequencies(fre)
    except:
        print('INCORRECT FREQ')
    e_t = loop()
    freq_vs_exe_table['FREQ'].append(fre)
    freq_vs_exe_table['EXEC_TIME'].append(e_t)
for F,E in zip(freq_vs_exe_table['FREQ'], freq_vs_exe_table['EXEC_TIME']):
    print(f'Frequency : {F//1000} Mhz     Execution time : {E} s')
cpu.set_governors('ondemand')

