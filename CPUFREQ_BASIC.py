from cpufreq import cpuFreq

##### PRINTING CURRENT FREQUENCIES #############
cpu = cpuFreq()  
        
freqs = cpu.get_frequencies()  
print('\nCURRENT FREQ :',freqs)

##### PRINTING CURRENT GOVERNORS

govns = cpu.get_governors()   
print('\nCURRENT GOVN :',govns)

##### PRINTING AVAILABLE GOVERNORS ###########

avail_governors = cpu.available_governors
print('\nAVAILABLE GOVN :',avail_governors)

######## CHANGING TO USERSPACE GOVERNOR ###########

cpu.set_governors('userspace')
govns = cpu.get_governors()   
print('\nCURRENT GOVN :',govns)
####### CHANGING THE CPU FREQUENCIES ###########

avail_freq = cpu.available_frequencies
print('\nAVAILABLE FREQ :',avail_freq)

print('\nCHANGING FREQUENCIES....')
freq_table = []
for fre in avail_freq:
    try:
        cpu.set_frequencies(fre)
    except:
        print('INCORRECT FREQ')
    freq_table.append(cpu.get_frequencies()[0])
print(*freq_table, sep = '\n')

######## RESETTING THE GOVERNOR TO ON-DEMAND
cpu.set_governors('ondemand')

