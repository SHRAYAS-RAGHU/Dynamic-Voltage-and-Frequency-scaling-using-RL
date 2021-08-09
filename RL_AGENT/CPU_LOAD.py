import subprocess as SP
import re 

def measure_load():
    io_c = SP.Popen(['uptime'], \
                                stdout=SP.PIPE, \
                                shell=False, stderr=SP.PIPE)
    s = str(io_c.communicate()[0]).strip('\\n\'').split(',')[-3:]  # TAKING THE LOAD AVG ALONE
    s = re.search(r'[\d]+.[\d]+', ''.join(s)) # FINDING ALL THE LOAD VALUES ALONE
    return s.group()
#print(measure_load())