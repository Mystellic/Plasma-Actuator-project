# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:36:37 2020

@author: myste
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

def ReadInfo(filename):
    #imports notepad
    try:
        infographic = np.array(pd.read_csv(filename, sep = "\t"))
    except:
        print("File was not found")
    return infographic

def ReadData(folder):
    #This method retrieves the names of all files in a certain folder
    files = np.array([])
    for name in glob.glob(folder + '\\*'):
        files = np.append(files,name)
    return files

def FileToArray(fileThing):
    #This method takes in a file and creates an array out of it, the files are read by another function used here
    newArray = []
    for element in fileThing:
        newPart = RunFile(element)
        newArray.append(newPart)
        useArray = np.array(newArray)
    return useArray

def RunFile(fileName):
    #This function is used in the FileToArray function and reads a file and creates a simple array
    file_list = np.array([])
    data = pd.read_csv(fileName,header=None)
    data = np.nan_to_num(np.array(data))
    return data

    
def CreateImage(pixelFrame):
    #Levels are created by having a set amount of levels between the min and max of the pixel values
    levels = np.linspace(np.amin(pixelFrame),np.amax(pixelFrame),30) 
    plt.contourf(pixelFrame,levels,cmap='nipy_spectral')
    plt.show()

def CleanCorrection(pictureArray,cleanInfo,allInfo):
    for i in range(pictureArray):
        #check which clean case needs to be subtracted 'Unfinished'
    
    return CorrectedArray

def GenerateClean(info):
    #This function finds the clean cases and puts the number of the test and the day in a list
    clean = [0,0,0]
    for element in info:
        if element[2] == element[3] == 0:
            series_day = [int(element[0][1:]),element[6]]
            clean[element[6]-1] = series_day

def main():
    #generate the info lists from the .txt's
    info1 = ReadInfo('Info_PA1.txt')
    info2 = ReadInfo('Info_PA2.txt')
    info3 = ReadInfo('Info_PA3.txt')
    
    cleanCases = GenerateClean(info1)
    

main()
    
        

#PA1 = FileToArray(ReadData('PA1'))
#CreateImage(PA1[0])