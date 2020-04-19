import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import glob
import os
from scipy import ndimage
from sklearn.linear_model import LinearRegression
import cv2 as cv

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

    
def CreateImage(pixelFrame,i):#pixelFrame2,Array,i):
    #Levels are created by having a set amount of levels between the min and max of the pixel values
    levels = np.linspace(np.amin(pixelFrame),np.amax(pixelFrame),30) 
    plt.contourf(pixelFrame,levels,cmap= 'gist_gray')#'RdGy')#'nipy_spectral')
    #plt.plot(Array[i],np.arange(0,428),'ro')
    
    #xx = np.arange(0,428)
    #plt.plot(f(xx,i,gradient(pixelFrame2)), xx)
    

    ax=plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis      
    ax.yaxis.tick_left() 
    plt.show()
    
    

def f(x,i,pixel_grad):
    fx = linearregression(pixel_grad)[1,i] + linearregression(pixel_grad)[2,i]*x
    return fx

def CleanCorrection(pictureArray,cleanInfo,dirtyInfo):
    correctedArray = []
    for dirty in dirtyInfo:
        for clean in cleanInfo:
            if dirty[1] == clean[1]:
                newArray = pictureArray[int(dirty[0])] - pictureArray[int(clean[0])]
                correctedArray.append(newArray)
                
        #check which clean case needs to be subtracted 'Unfinished'
    correctedArray = np.array(correctedArray)
    return correctedArray

def GenerateClean(info):
    #This function finds the clean cases and puts the number of the test and the day in a list
    #Same is done for "dirty" cases so it is known what to subtract from
    clean = [0,0,0]
    dirty = []
    for element in info:
        if element[2] == element[3] == 0:
            series_day = [int(element[0][1:]),element[6]]
            clean[element[6]-1] = series_day
        else:
            series_day = [int(element[0][1:]),element[6]]
            dirty.append(series_day)
    return clean, dirty

#Written by Sofia----------------------------------------------------


def slicing(pictureArray):
    #This function zooms in on the picture
    zoom_picture = pictureArray[:,25:453,21:619]
    #Now we have an array
    return zoom_picture                          # of shape 428*598


#def threshold(Array):


def gradient(pictureArray):
    
    gradient_PA = np.gradient(pictureArray, axis = 1)
    
    return np.array(gradient_PA)



def maxelement(Array):
    transition_per_picture =[]
    
    for i in range(len(Array)):
        max_value_per_row = np.amin(Array[i],axis=1)  #Selected maximum value per row
        transition_per_picture.append(max_value_per_row)
        
    return np.array(transition_per_picture)

def indexofmax(Array):
    index_of_max = []

    for i in range(len(Array)):
        index = np.argmin(Array[i],axis = 1) #Index of max value
        index_of_max.append(index)
    return np.array(index_of_max)

def rotating_pic(Array):

    rot = ndimage.rotate(Array,-45, axes =(1,2), mode = 'constant')
    return rot


def linearregression(Array):
    
    x = np.arange(0,428).reshape((-1,1))
    a_0list = []
    a_1list = []                                            # Function for least
    r_sqlist = []                                           # squares linear regression
   
    for k in range(len(Array)):
        y = indexofmax(gradient(Array))[k].reshape((-1,1))
        model = LinearRegression().fit(x,y)
        r_sq = model.score(x,y)
        a_0 = model.intercept_   # y = a_0 +a_1*x
        a_1 = model.coef_
        
        r_sqlist.append(r_sq)
        a_0list.append(float(a_0))
        a_1list.append(float(a_1))

    output = [np.array(r_sqlist),np.array(a_0list),np.array(a_1list)]
        
    return np.array(output)

def thresholdadaptive(Array):
    #src = cv.imread(Array)
    #src = np.full((38, 428, 598), 12, np.uint8)
    #src = mpimg.imread()
    
    grayArray = cv.cvtColor(Array, cv.COLOR_BGR2GRAY)
    
    blockSize = 11
    subtractConstant = 4

    #fill in the Array here
    img = cv.adaptiveThreshold(grayArray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,blockSize,subtractConstant)
    
    return img
    
#------------------------------------------------#------------------------------------------------------------

def main():

    #generate the info lists from the .txt's
    
    #info1 = ReadInfo('Info_PA1.txt')
    info2 = ReadInfo('Info_PA2.txt')
    #info3 = ReadInfo('Info_PA3.txt')
    
    #Change PA1 as a string to PA2 or PA3 to look at those instead
    
    pictureArray = FileToArray(ReadData('PA2'))
    cleanCases, dirtyCases = GenerateClean(info2)
    pictures = slicing(pictureArray)
    
    index_of_trans = indexofmax(gradient(pictures))
    index_trans = indexofmax(pictures)
           
    #pictures = CleanCorrection(pictureArray,cleanCases,dirtyCases)
  
    #print(linearregression(gradient(pictures)))
   
    #CreateImage(pictures[0])
    #print(CleanCorrection(pictureArray,cleanCases,dirtyCases))
    
    
    for i in range(len(pictureArray)):
        print(slicing(pictureArray).shape)
        #CreateImage(pictures[i],gradient(pictures),index_of_trans,i)  #Added by me
        CreateImage(pictures[i],i)
    
    
main()
#PA1 = FileToArray(ReadData('PA1'))
#CreateImage(PA1[0])
