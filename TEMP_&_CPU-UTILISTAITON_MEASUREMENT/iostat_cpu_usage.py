import subprocess                                               # TO INVOKE TERMINAL FROM PYTHON
import csv                                                      # SAVING THE MEASURED PARAMS TO A CSV FILE
                                                                # FOR CPU UTILISATION $ iostat -c 
io_c = subprocess.Popen(['iostat', '-c'], \
                        stdout=subprocess.PIPE, \
                        shell=False, stderr=subprocess.PIPE)
s = io_c.communicate()[0]                                       # OUTPUT OF COMMAND STORED IN 'S' AS BYTES

s = str(s)                                                      # CONVERTING BYTE DATA TO STRING FOR PROCESSING
s = s.split('\\n')[3]                                           # OBTAINING CPU USAGE STATS ALONE AFTER SPLITTING SEPARATE LINES INTO LISTS
s = s.strip(' ')                                                # STRIPPING INITIAL SPACES TO OBTAIN THE SET OF 5 NOS.
s = s.split('   ')                                              # CONVERTING SET OF 5 VALUES INTO A LIST BBY SPLITTING

cpu_usage = float(s[0])                                         # CPU USAGE IS THE FIRST ELEMENT OF THE LIST
cpu_idle_time_percent = float(s[len(s) - 1])                    # IDLE TIME PERCENT IS THE LAST ELEMENT OF THE LIST
print(f"cpu_usage {cpu_usage}, cpu_idle_time_percent {cpu_idle_time_percent}")
