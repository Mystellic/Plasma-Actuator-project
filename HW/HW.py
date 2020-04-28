# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:47:39 2020

@author: Dick Daniel
"""

import numpy as np
import os
import pandas as pd
import glob
from timeit import default_timer as timer
import matplotlib.pyplot as plt



path = "E:\\Downloads\\raw"
os.chdir( path )


def RunFile(number):
    number =str(number)

    if len(number) < 2:
        number = '0' + number    
    if len(number) < 3:
        number = '0' + number
    
    fileName = 'R09\\data_'+number+'.txt'

    data = pd.read_csv(fileName, sep='\t', header=None )
    data = np.nan_to_num(np.array(data))[:,1]
    return data

def Average(List):
    average = sum(List)/len(List)
    return average



def Contour_Array():
   row = np.empty((29,0))


   for j in range(0,35): 
      collumn = np.zeros(shape=(29,1))
      for i in range(1,30):
         n = i+(j)*30
         collumn[29-i,0] = Average(RunFile(n))
      row = np.hstack((collumn,row))
   return row


#print (Contour_Array())
def Plotting(Array):

    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 
    ax.set_title('Contour Plot')
    ax.set_xlabel('z')
    ax.set_ylabel('y')
    cp = plt.contourf(Array)
    plt.colorbar(cp)
    plt.plot

Plotting(Contour_Array())
