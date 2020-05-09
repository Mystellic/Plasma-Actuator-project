import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *
import numpy as np
import pandas as pd
import glob
import os
from scipy import ndimage
from sklearn.linear_model import LinearRegression
import cv2 as cv
from PIL import Image

def ReadInfo(filename):
    #imports notepad
    try:
        infographic = np.array(pd.read_csv(filename, sep = "\t"))
        return infographic
    except:
        print("File was not found")
    

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

    
def CreateImage(pixelFrame,pixelFrame2,Array,i):
    #Levels are created by having a set amount of levels between the min and max of the pixel values
    levels = np.linspace(np.amin(pixelFrame),np.amax(pixelFrame),30) 
    plt.contourf(pixelFrame,levels,cmap= 'RdGy')#'nipy_spectral')#'gray'
    #plt.plot(Array,np.arange(0,428),'ro')
    plt.plot(Array[i],np.arange(0,428),'ro')     #725 #For all pictures
    
    xx = np.arange(0,428)
    plt.plot(f(xx,i,pixelFrame2), xx)
    
    #name = str(i)
    
    ax=plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
    ax.xaxis.tick_top()                     # and move the X-Axis      
    ax.yaxis.tick_left()
    plt.axis('off')
    plt.title(i)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    #plt.imsave(name,pixelFrame, cmap = 'gray')
    plt.show()

def f(x,i,pixel_grad):
    fx = linearregression(pixel_grad)[1] + linearregression(pixel_grad)[2]*x
    #For 38 pictures
    #fx = linearregression(pixel_grad)[1,i] + linearregression(pixel_grad)[2,i]*x
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

def PrePlotTables(info,deltas):
    useful = []
    for element in info:
        if element[2] != 0:
            useful.append(element[2])
    frequency4,delta4,i = [],[],0
    #voltage 4
    for element in info:
        if element[3] == 4:
            frequency4.append(useful[i])
            delta4.append(deltas[i])
        if element[2] != 0:
            i+=1
    
    #voltage6 
    frequency6,delta6,i = [],[],0
    for element in info:
        if element[3] == 6:
            frequency6.append(useful[i])
            delta6.append(deltas[i])
        if element[2] != 0:
            i+=1
    
    #voltage8
    frequency8,delta8,i = [],[],0

    for element in info:
        if element[3] == 8:
            frequency8.append(useful[i])
            delta8.append(deltas[i])
        if element[2] != 0:
            i+=1
    
    #voltage10
    frequency10,delta10,i = [],[],0
    for element in info:
        if element[3] == 10:
            frequency10.append(useful[i])
            delta10.append(deltas[i])
        if element[2] != 0:
            i+=1
    
    #voltage12
    frequency12,delta12,i = [],[],0
    for element in info:
        if element[3] == 12:
            frequency12.append(useful[i])
            delta12.append(deltas[i])
        if element[2] != 0:
            i+=1    
    
    return frequency4,frequency6,frequency8,frequency10,frequency12,delta4,delta6,delta8,delta10,delta12
            

#Written by Sofia, Mustafa, Keiya and Serena----------------------------------------------------


def slicing(pictureArray):
    #This function zooms in on the picture
    zoom_picture = pictureArray[:,25:453,21:619]
    #Now we have an array
    return zoom_picture                          # of shape 428*598


def gradient(pictureArray):
    
    gradient_PA = np.gradient(pictureArray, axis = 1)#axis = 0)
    gaussian_filter = ndimage.gaussian_filter(gradient_PA, sigma = 0)
    return np.array(gaussian_filter)



def maxelement(Array):
    #This function finds the maximum gradient
    #value per row of the picture array
    transition_per_picture =[]
    
    for i in range(len(Array)):
        max_value_per_row = np.amax(Array[i],axis = 1)#axis=0)  #Selected maximum value per row
        transition_per_picture.append(max_value_per_row)
        
    return np.array(transition_per_picture)

def indexofmax(Array):
    #This function finds the index of the
    #max gradient point per row of the picture array
    index_of_max = []

    for i in range(len(Array)):
        index = np.argmax(Array[i],axis = 1)#axis = 0) #Index of max value
        index_of_max.append(index)
    index_of_max = np.array(index_of_max)

    return index_of_max

    
    
def standarddevi(Array):
    #Function aims to eliminate the outliers of
    #the max gradient location and being able to plot it
    ArrayCorrected = []
    Finallist = []
    for m in range(len(Array)):
        mean = np.mean(Array[m],axis=0)#axis = 1
        sd = np.std(Array[m], axis=0) #axis = 1

        x = Array[m] > (mean-sd)   # First condition
        y = Array[m] < (mean+sd)   #Second condition
        Array1 = np.where(x,Array[m],np.nan)
        Array2 = np.where(y,Array1, np.nan)

        final_list = [x for x in Array[m] if (x > mean - sd)]
        final_list = [x for x in final_list if (x < mean + sd)]
        Finallist.append(np.array(final_list))
        ArrayCorrected.append(Array2)
        
    ArrayCorrected = np.array(ArrayCorrected)
    Finallist = np.array(Finallist)
   
    return ArrayCorrected,Finallist


def rotating_pic(Array):
    #This functions rotates the picture 45 deg
    rot = ndimage.rotate(Array,-45, axes =(1,2), mode = 'constant')
    return rot


def linearregression(Array):
    
    x = np.arange(0,Array.shape[0]).reshape((-1,1))
    a_0list = []
    a_1list = []                                            # Function for least
    r_sqlist = []                                           # squares linear regression
   
    #For one picture only, therefore a 2D array
    
    y = Array.reshape((-1,1))     #indexofmax(gradient(Array)).reshape((-1,1))
    model = LinearRegression().fit(x,y)
    r_sq = model.score(x,y)
    a_0 = model.intercept_
    a_1 = model.coef_
    
    r_sqlist.append(r_sq)
    a_0list.append(float(a_0))
    a_1list.append(float(a_1))
    output = [np.array(r_sqlist),np.array(a_0list),np.array(a_1list)]
    
    return np.array(output)
    
def arrayconvert(Array):
    #This function converts a picture into an array
    templist = []
    for i in range(len(Array)):
        
        temp= np.asarray(Image.open(Array[i]))
        temp1 = np.mean(temp,axis = 2)
        templist.append(temp1)
    templist = np.array(templist)
    return templist
    

def FinalDeltaTransition(pictureArray,info):
    

    #Gives delta transition.
    #Inputs: 3D Array of transition location points( # of pics, one row, 428 columns)
    #Output: 3D Array with delta transition (one per row) scaled to a physical distance
    #       on the wing and considering sweep as well
    
    cleanCases, dirtyCases = GenerateClean(info)

    DeltaTransition = CleanCorrection(pictureArray,cleanCases,dirtyCases)
    DeltaTransition = 0.2123745819*DeltaTransition          #Gives delta transition in centimeters
                                                            #One matrix unit is 0.2123745 cm
    DeltaTransition = DeltaTransition/cos((45*np.pi)/180)  #Considering sweep
    
    return DeltaTransition

def AveragingTransition(Array,i):
    #This function averages the transition points, giving one
    #transition location(This is still a position on the array
    #not a physical distance to the leading edge)

    xx = np.arange(0,428)
    fx = f(xx,i,Array)
    
    FinalDeltaTransition = np.mean(fx, axis = 0)#axis = 1 Check after
    return FinalDeltaTransition



#------------------------------------------------#------------------------------------------------------------

def main():

    #generate the info lists from the .txt's
    

    info1 = ReadInfo('Info_PA1.txt')
    info2 = ReadInfo('Info_PA2.txt')
    info3 = ReadInfo('Info_PA3.txt')

    
    
    #Change PA1 as a string to PA2 or PA3 to look at those instead
    ThisPA = 'PA1' 
    
    pictureArray = FileToArray(ReadData(ThisPA))

    

    '''
    Taking all of the threshholded pictures of PA1, PA2,PA3 and put them into a
    3D array through the use of the functions FileToArray(ReadData('Filename'))

    After this is just running it through the functions already written
    to get the transition points and linear regression.

    Take the mean of the transition points to
    get an averaged transition location

    Subtract the clean cases from the PA cases

    Scale up with the wing dimensions and take into account the 45 degree sweep

    '''
    Threshold_pics = 'Images_after_Re_subtraction\\ThresholdPictures\\'
    pictures2 = slicing(pictureArray)
    pictures = ReadData(Threshold_pics + 'ThresholdPA1')  #Change to PA1, PA2, PA3
    pictures1 = arrayconvert(pictures)

    ArrayCorrected, Final = standarddevi(indexofmax(gradient(pictures1)))

    
    
    
    #CreateImage(pictures2[53],gradient(pictures1),ArrayCorrected,53)            #To Produce only one picture
    #CreateImage(gradient(pictures1)[53],gradient(pictures1),ArrayCorrected,53)  #To produce the picture of the gradient of
                                                                                #the threshold pics
    
    Averagedtransition = []
    
    for i in range(len(pictures1)):
        
        #CreateImage(pictures2[i],Final[i],ArrayCorrected,i) #To plot image with transition points on
        
        Averaged_transition  = AveragingTransition(Final[i],i)     # To average the transition location
        Averagedtransition.append(np.array(Averaged_transition))

    
    Averagedtransition = np.array(Averagedtransition).reshape((len(pictures1),1))
  
    
    DeltaTransition = FinalDeltaTransition(Averagedtransition,info1)  #Change to info1 or info2 or info3 
    #print('The Delta Transition are:',DeltaTransition,'cm')
    #print()
    #print(DeltaTransition.shape)
   # print()
    
    
    frequency4,frequency6,frequency8,frequency10,frequency12,delta4,delta6,delta8,delta10,delta12 = PrePlotTables(info1,DeltaTransition)


    plt.plot(frequency4,delta4)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    #plt.title('Voltage = 4kV')
    plt.savefig(ThisPA + '_4')
    plt.clf()
    
    plt.plot(frequency6,delta6)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    #plt.title('Voltage = 6kV')
    plt.savefig(ThisPA + '_6')
    plt.clf()
    
    plt.plot(frequency8,delta8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    #plt.title('Voltage = 8kV')
    plt.savefig(ThisPA + '_8')
    plt.clf()


    plt.plot(frequency10,delta10)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    #plt.title('Voltage = 10kV')
    plt.savefig(ThisPA + '_10')
    plt.clf()
    
    
    plt.plot(frequency12,delta12)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    #plt.title('Voltage = 12V')
    plt.savefig(ThisPA + '_12')
    plt.close()
    
'''
    plt.figure
    plt.subplot(231)
    plt.plot(frequency4,delta4)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    plt.title('Volatage = 4V')
    plt.subplot(232)
    plt.plot(frequency6,delta6)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    plt.title('Volatage = 6V')
    plt.subplot(233)
    plt.plot(frequency8,delta8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    plt.title('Volatage = 8V')
    plt.subplot(234)
    plt.plot(frequency10,delta10)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    plt.title('Volatage = 10V')
    plt.subplot(235)
    plt.plot(frequency12,delta12)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Delta Transition (cm)')
    plt.title('Volatage = 12V')
    plt.show()
'''  
    
main()

