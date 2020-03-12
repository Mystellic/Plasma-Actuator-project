"""
This is a processor for the data recieved of a plasma actuator test
"""

import numpy as np
import os
import pandas as pd
import glob
from timeit import default_timer as timer


#function to run through all folders in the data directory

def RunFile(fileName):
    file_list = np.array([])
    data = pd.read_csv(fileName,sep=";",header=None)
    data = np.nan_to_num(np.array(data))
    return data

def SplitFolder(folder):
    start = timer()
    files = np.array([])
    for name in glob.glob(folder + '\\*'):
        files = np.append(files,name)
    halved = int(len(files)/2)
    lowFiles = files[0:halved]
    highFiles = files[halved:]
    end = timer()
    return lowFiles, highFiles, end - start

def AverageThis(arrays):
    return np.mean(arrays)

def FileToArray(reynolds):
    newArray = []
    for element in reynolds:
        newPart = RunFile(element)
        newArray.append(newPart)
        useArray = np.array(newArray)
        return useArray
    
path = 'C:\\Test, Analysis & Simulation DATA\\AE2223\\FFS_PA1\\t200\\IR\\R00\\Full'
low,high,time = SplitFolder(path)

s=timer()

e=timer()

print(LowAverage,e-s)


