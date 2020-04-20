# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:01:28 2020

@author: Robin
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob 


def ThresholdOnPicture(filename,i):
    img = cv2.imread(filename,0)
    #img = cv2.medianBlur(img,5)
    
    ret,th1 = cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
    #th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                #cv2.THRESH_BINARY,11,4)
    
    #This is the line of code you'll be using
    #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                #cv2.THRESH_BINARY,11,1)
    
    
    
    '''
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    '''
    
    
    plt.imshow(th1,'gray')
    plt.xticks([]),plt.yticks([])
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    
    img_file = 'Images_after_Re_subtraction\\Threshold' + PA + '\\ThresholdCorrection' + str(i)
    plt.imsave(img_file + '.png',th1, cmap = 'gray')
    
    #print(img.shape)
    #print(type(images[3]))



#change PA1 to PA2 or PA3 for their respective results
PA = 'PA3'
Images = 'HighminusLow' + PA

for file in glob.glob('Images_after_Re_subtraction\\' + Images + '\\*'):
    i = file[-6:-4]
    ThresholdOnPicture(file,i)



