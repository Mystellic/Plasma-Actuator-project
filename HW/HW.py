# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:47:39 2020

@author: Dick Daniel
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

path = "E:\\Downloads\\raw"
os.chdir( path )


def RunFile(number, Set):
    number =str(number)
    Set = str(Set)
    if len(number) < 2:
        number = '0' + number    
    if len(number) < 3:
        number = '0' + number
    
    fileName = Set + '\\data_'+number+'.txt'

    data = pd.read_csv(fileName, sep='\t', header=None )
    data = np.nan_to_num(np.array(data))[:,1]
    return data

def Average(List):
    average = sum(List)/len(List)
    return average

def Get_Data(PA_Number, Frequency):

    data = pd.read_csv('matrix.txt', sep='\t', header=None )
    data = np.nan_to_num(np.array(data))
    Data_Set = [0,0,0]
    for i in range(0,40):
        if data[i,1] == str(PA_Number) and data[i,3] == str(Frequency):
            if data[i,2] == '0.15':
               Data_Set[0] = data[i,0]
            elif data[i,2] == '0.175':
                Data_Set[1] = data[i,0]
            elif data[i,2] == '0.2':
                Data_Set[2] = data[i,0]
    return Data_Set

def Contour_Array(Data_Set):
  rows = []
  for k in Data_Set:
   row = np.empty((29,0))


   for j in range(0,35): 
      collumn = np.zeros(shape=(29,1))
      for i in range(1,30):
         n = i+(j)*30
         collumn[29-i,0] = Average(RunFile(n,k))
      row = np.hstack((collumn,row))
   
   row = np.flipud(row)
   row = np.fliplr(row)
   rows.append(row)
    
  return rows


def Plotting(Array,PA,Frequency):

    nz, ny = (35, 29)
    z = np.linspace(0, 16, nz)
    y = np.linspace(0, 2, ny)
    zv, yv = np.meshgrid(z, y)
    fig = plt.figure()
    #fig = plt.savefig('PA'+str(PA)+'_Freq'+str(Frequency)+ '.png', bbox_inches='tight')
    plt.suptitle('Velocity Contour PA='+ str(PA) +' ,Frequency=' + str(Frequency)+'Hz' )


    
    plt.subplot(311)
    plt.subplots_adjust(hspace = 0.9)
    cp = plt.contourf(zv,yv, Array[0])
    cb = plt.colorbar(cp)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    plt.show()
    plt.plot()
    plt.xlabel('z [mm]')
    plt.ylabel('y [mm]')
    plt.title('Chord =0.15 x/c')
    
    plt.subplot(312)
    plt.subplots_adjust(hspace = 0.9)
    cp = plt.contourf(zv,yv, Array[1])
    cb = plt.colorbar(cp)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    plt.plot()
    plt.xlabel('z [mm]')
    plt.ylabel('y [mm]')
    plt.title('Chord =0.175 x/c')
    
    plt.subplot(313)
    plt.subplots_adjust(hspace = 0.9)
    cp = plt.contourf(zv,yv, Array[2])
    cb = plt.colorbar(cp)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    plt.plot()
    plt.xlabel('z [mm]')
    plt.ylabel('y [mm]')
    plt.title('Chord =0.2 x/c')
    
    plt.savefig('PA'+str(PA)+'_Freq'+str(Frequency)+ '.png', bbox_inches='tight')
    
Plotting(Contour_Array(Get_Data(2,1000)),2,1000)

