from cpufreq import cpuFreq

cpu = cpuFreq()               ############ 
freqs = cpu.get_frequencies()
print(freqs)

govns = cpu.get_governors()
print(govns)