import random
import time
from cpufreq import cpuFreq
cpu = cpuFreq()

def loop():
    start = time.time()
    for i in range(1000):
        for i in range(10000):
            pass
    end =  time.time()
    return f'{(end - start):.4f}'

print('Ondemand', loop(), cpu.get_frequencies())

cpu.set_governors('userspace')
freq_vs_exe_table = {'FREQ': [], 'EXEC_TIME':[]}
for _ in range(10):
    try:
        cpu.set_frequencies(1500000)
    except:
        print('INCORRECT FREQ')
    e_t = loop()
    print(f'Frequency : {1500} Mhz    Execution time : {e_t} s')
cpu.set_governors('ondemand')

