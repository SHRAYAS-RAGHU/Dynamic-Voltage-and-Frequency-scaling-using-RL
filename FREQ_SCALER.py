from cpufreq import cpuFreq

##### PRINTING CURRENT FREQUENCIES #############
cpu = cpuFreq()  
        
freqs = cpu.get_frequencies()  
print(freqs)

##### PRINTING CURRENT GOVERNORS

govns = cpu.get_governors()   
print(govns)

##### PRINTING AVAILABLE GOVERNORS ###########

avail_governors = cpu.available_governors
print(avail_governors)

######## CHANGING TO USERSPACE GOVERNOR ###########

cpu.set_governors('userspace')
govns = cpu.get_governors()   
print(govns)
####### CHANGING THE CPU FREQUENCIES ###########

avail_freq = cpu.available_frequencies
freq_table = []
for fre in avail_freq[1:]:
    try:
        cpu.set_frequencies(fre)
    except:
        print('INCORRECT FREQ')
    freq_table.append(cpu.get_frequencies())
print(freq_table)

######## RESETTING THE GOVERNOR TO ON-DEMAND
cpu.set_governors('ondemand')

