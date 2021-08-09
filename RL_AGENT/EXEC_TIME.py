"""
run the command from terminal :sudo stress --cpu 4 --io 3 --vm 2 --vm-bytes 128M --timeout 10s
run the code from vS_terminal : python3 exec_time.py
"""

import subprocess

def exec_time(n):
    n_measure = n
    io_c = subprocess.Popen(['iostat', '-c', '1', str(n_measure)], \
                                stdout=subprocess.PIPE, \
                                shell=False, stderr=subprocess.PIPE, cwd='/home/pi/Desktop/PROJECT/RL_AGENT')
    s = io_c.communicate()[0]                                       # OUTPUT OF COMMAND STORED IN 'S' AS BYTES
    s = str(s).split('\\n\\n')[1:n_measure+1]
    x = []
    for i in s:
        util = 100 - float(i.split(' ')[-1].strip('\\n\''))
        x.append(util)
    
    return max(x)