import subprocess as SP
def measure_sched():
    a = SP.Popen(['cat', '/proc/schedstat'], \
                                    stdout=SP.PIPE, \
                                    shell=False, stderr=SP.PIPE)
    a = str(a.communicate()[0])
    a = a.strip('\'').split('\\n')
    a = [i for j,i in enumerate(a) if j>0 and j % 2 == 0][:4]
    print(*a, sep = '\n')
    return a[0]
measure_sched()