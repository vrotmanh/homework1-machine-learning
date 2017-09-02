#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:38:53 2017

@author: vicenterotman
"""
from matplotlib import pyplot as plt
import numpy as np
import csv
import re

def displayDigit(image):
    print_image = np.reshape(image,(28,28))
    plt.imshow(print_image,cmap = 'gray_r')
    return

def main():
    with open("data/train.csv") as file_object:
        lines = file_object.readlines()
        print(len(lines))
    
    total_images_count = len(lines)-1
    test_data = np.zeros([total_images_count,784])
    labels_test_data = []
    i = 0
    
    for line in lines:
        sample = line.split(',')
        if i > 0 & i < total_images_count:
            labels_test_data.append(sample[0])
            j = 0
            for j in range(783):
                test_data[i-1,j] =  int(sample [j+1])
        i = i+1
    
    displayDigit(test_data[1,:])
    
main()