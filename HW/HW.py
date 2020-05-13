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
from scipy.signal import butter, lfilter, sosfilt, sosfreqz
import time

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
    
#Plotting(Contour_Array(Get_Data(2,1000)),2,1000)


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

def Standard_Deviation(List):
    x = 0
    for i in List:
        x = x + (i-Average(List))**2
    s = np.sqrt(x/(len(List)-1))
    return s

def Filter(data,Frequency):
    fs = 51200
    lowcut = Frequency-3
    highcut = Frequency+3
    order = 3
    Extension = 40000
    front = data[0:Extension]
    back = data[102400-Extension:102400]
    data = np.hstack((front,data,back)) 
    filter = butter_bandpass_filter(data,lowcut,highcut,fs,order)
    return filter[Extension:102399+Extension]



def Filtered_Contour_Array(Data_Set):
  rows = []
  for k in Data_Set:
   row = np.empty((29,0))


   for j in range(0,35): 
      collumn = np.zeros(shape=(29,1))
      for i in range(1,30):
         n = i+(j)*30
         collumn[29-i,0] = Standard_Deviation(Filter(RunFile(n,k),400))
      row = np.hstack((collumn,row))
   
   row = np.flipud(row)
   row = np.fliplr(row)
   rows.append(row)
    
  return rows

def Filtered_Plot(Set,Frequency):
    
  
   row = np.empty((29,0))


   for j in range(0,35):
      collumn = np.zeros(shape=(29,1))
      for i in range(1,30):
         n = i+(j)*30
         collumn[29-i,0] = np.std(Filter((RunFile(n,Set)),Frequency))
      row = np.hstack((collumn,row))
   
   row = np.flipud(row)
   row = np.fliplr(row)
   
    
   return row


#Plotting(Filtered_Contour_Array(Get_Data(2,1000)),2,1000)

def Run():
    data = pd.read_csv('R00\data_1010.txt', sep='\t', header=None )
    data = np.nan_to_num(np.array(data))[1:,1]

    
    Extension = 40000
    front = data[0:Extension]
    back = data[102400-Extension:102400]
    data = np.hstack((front,data,back))  
    print(len(data))
    filter = Filter(data, 600)
    time = np.linspace(0,2,51200*2-1)
    w, h = sosfreqz(butter_bandpass(597,603,51200,6))
                    
    plt.figure()
    plt.subplot(311)
    plt.subplots_adjust(hspace = 0.5)
    plt.plot(time,data[Extension:102399+Extension])
    plt.xlabel('Measured HW Signal')
    plt.plot()
    
    plt.subplot(312)
    plt.subplots_adjust(hspace = 0.5)
    plt.plot(time, filter[Extension:102399+Extension])
    plt.xlabel('Filtered HW Signal (600 Hz)')
    plt.plot()
    
    plt.subplot(313)
    
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.xlabel('Frequency [rad/sample]')
    plt.ylabel('Amplitude [dB]')
    plt.plot()
    
start = time.time()    
plt.figure()
nz, ny = (35, 29)
z = np.linspace(0, 16, nz)
y = np.linspace(0, 2, ny)
zv, yv = np.meshgrid(z, y)
cp = plt.contourf(zv,yv,Filtered_Plot('R01',600))
cb = plt.colorbar(cp)
end = time.time()
print(end - start)
