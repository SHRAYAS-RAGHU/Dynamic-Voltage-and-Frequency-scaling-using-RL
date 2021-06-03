import subprocess
import re  

for i in range(1000):
    a = subprocess.Popen(['time','-f', '%P','python3','WORK.py'],
        cwd='/home/pi/Desktop/PROJECT/RL_AGENT',stdout=subprocess.PIPE
        ,stderr=subprocess.PIPE,shell=False)
    print(str(a.communicate()[1]).lstrip('b\'').rstrip('\\n\''))

"""
a = str(a.communicate()[1])
a = a.split('\\n')[:-1]
b = []
for i in a:
    x = re.search(r'[\d.]+', i)
    b.append(float(x.group()))
print(b)
"""