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

#this function splits the folders in half
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
    sm = np.sum(arrays,axis=0)
    average = sm/len(arrays)
    return average

def FileToArray(reynolds):
    newArray = []
    for element in reynolds:
        newPart = RunFile(element)
        newArray.append(newPart)
        useArray = np.array(newArray)
    return useArray

#find the difference between the high and low reynolds numbers files
def HighLowCorrection(high,low):
    return high - low

#begin of main program
path = 'C:\\Test, Analysis & Simulation DATA\\AE2223\\FFS_PA1\\t200\\IR\\R01\\Full'
low,high,time = SplitFolder(path)

notepad1_path = 'C:\\Test, Analysis & Simulation DATA\\AE2223\\FFS_PA1\\'
notepad1 = pd.read_csv(notepad1_path + 'Master_M3J_PA1.txt', header=None)
print(notepad1)

lowArray = AverageThis(FileToArray(low))
highArray = AverageThis(FileToArray(high))

s=timer()
#test = HighLowCorrection(highArray,lowArray)
e=timer()

#print(test,np.shape(test),e-s)