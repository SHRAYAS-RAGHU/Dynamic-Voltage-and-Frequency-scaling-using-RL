import subprocess
import os
from sys import stdout
try:
    os.chdir('/home/pi/Desktop/PROJECT/RL_AGENT')
except OSError:
    print('enter a valid path')
    
a = subprocess.Popen(['time','-p','python3','WORK.py'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=False)
print(a.communicate())