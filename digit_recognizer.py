#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:38:53 2017

@author: vicenterotman
"""
from matplotlib import pyplot as plt
import numpy
from PIL import Image

def displayDigit(image):
    npimage = numpy.array(image)

    #plt.imshow(npimage, cmap='gray', interpolation='nearest');
    #pixels = npimage.image.reshape((28,28))
    #plt.imshow(pixels, cmap='gray')
    return

def convertToImage(array):
    image = []
    current=[]
    count = 0
    for i in array:
        if(count < 28):
            current.append(i)
        if(count == 27):
            image.append(current)
            current = []
            count = -1
        count+=1
        
    return image

imageNumber = 0
for line in open("data/train.csv"):    
    if imageNumber == 0:
        imageNumber=+1
        continue
    
    array = line.strip('\n\r').split(',')
    array = [ int(x) for x in array ]
    array.pop(0)
    image = convertToImage(array)
    displayDigit(array)
    
    break