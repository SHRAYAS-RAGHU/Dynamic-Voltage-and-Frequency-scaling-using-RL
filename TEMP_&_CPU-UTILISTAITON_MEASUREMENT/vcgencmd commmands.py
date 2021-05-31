from vcgencmd import Vcgencmd # MODULE TO MEASURE CPU DATA
vcgm = Vcgencmd() # OBJECT OF VCGENCMD CLASS

# FOR MEASURING THE STATS THIS INSTANCE IS USED TO ACCESS THE MEMBERS OF THE CLASS. 

state = {'cpu_temp': [], 'cpu_volt': [], 'cpu_freq': [], 'arm_freq': []} # DICTIONARY TO STORE CPU STATS
for i in range(5):     
    cpu_temp = vcgm.measure_temp()                # MEASURING THE TEMPARATURE OF CORE
    cpu_volt = vcgm.measure_volts('core')         # MEASURING THE VOLTAGE OF CORE
    cpu_freq = vcgm.measure_clock('core')/1000000 # MEASURING THE FREQUENCY OF CORE
    arm_freq = vcgm.measure_clock('arm')/1000000  # MEASURING THE FREQUENCY OF ARM
    
    # VALUES MEASURED ARE STORED IN THE DICTIONARY
    
    state['cpu_temp'].append(cpu_temp)
    state['cpu_volt'].append(cpu_volt)
    state['cpu_freq'].append(cpu_freq)
    state['arm_freq'].append(arm_freq)
print(state)
