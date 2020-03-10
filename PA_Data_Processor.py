"""
This is a processor for the data recieved of a plasma actuator test
"""

import numpy as np
import csv
import os






path = 'C:\\Test, Analysis & Simulation DATA\\AE2223\\FFS_PA1\\t200\\IR\\R00\\Full'




if os.path.exists(path) :  
    os.chdir(path)
else:
    print("Working Directory error")
    
with open('run_2020-01-23_15-40-43.csv', newline='') as csvfile:
    data = csv.reader(csvfile)

aData = np.asarray(data)

print(data)