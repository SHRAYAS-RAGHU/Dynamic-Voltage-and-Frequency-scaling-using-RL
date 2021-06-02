import subprocess
import re  

a = subprocess.Popen(['time','-p','python3','WORK.py'],
    cwd='/home/pi/Desktop/PROJECT/RL_AGENT',
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=False)

a = str(a.communicate()[1])
a = a.split('\\n')[:-1]
b = []
for i in a:
    x = re.search(r'[\d.]+', i)
    b.append(float(x.group()))
print(b)