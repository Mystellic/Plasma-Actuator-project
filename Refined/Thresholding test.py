# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:01:28 2020

@author: Robin
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('picture[0]3.png',0)
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
plt.imsave('threshold1',th1, cmap = 'gray')
plt.show()

print(img.shape)
#print(type(images[3]))
