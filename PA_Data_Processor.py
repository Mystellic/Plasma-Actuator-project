"""
This is a processor for the data recieved of a plasma actuator test
"""

import numpy as np
import os


path = 'C:\\Test, Analysis & Simulation DATA\\AE2223\\FFS_PA1\\t200\\IR\\R00\\Full'

if os.path.exists(path) :  
    os.chdir(path)
else:
    print("Working Directory error")

my_data = np.genfromtxt('run_2020-01-23_15-40-43.csv', delimiter=';')

print(my_data[0])