#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:38:53 2017

@author: vicenterotman
"""
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

def displayDigit(image):
    print_image = np.reshape(image,(28,28))
    plt.figure()
    plt.imshow(print_image,cmap = 'gray_r')
    return

def displayHistogram(labels):
    plt.hist(labels)
    
def getOneSampleOfEachDigit(labels, data):
    array = []
    i = 0
    items = [0,1,2,3,4,5,6,7,8,9]
    while (len(items) > 0):
        if(labels[i] in items):
            array.append(data[i])
            items.remove(labels[i])
        i+=1
    return array

def getNearestNeighbour(digit, data):
    d = float("inf")
    result = data[0]
    for item in data:
        if np.array_equal(item, digit):
            continue
        dist = abs(getDistance(item, digit))
        if dist < d:
            d = dist
            result = item
    return result

def getDistance(first_digit, second_digit):
    return np.linalg.norm(first_digit-second_digit)

def printNearestNeighbourOfEachDigit(labels, data):
    array = getOneSampleOfEachDigit(labels, data)
    for digit in array:  
        displayDigit(digit)
        displayDigit(getNearestNeighbour(digit, data))
        
def getZerosAndOnesIndexes(labels):
    array = []
    i = 0
    for item in labels:
        if item == 0 or item == 1:
            array.append(i)
        i+=1
    return array

def printImpostorsAndGenuineMatchesForZerosAndOnes(labels, data):
    array = getZerosAndOnesIndexes(labels)
    genuine_matches = 0
    impostor_matches = 0
    print("total: " + str(len(array)))
    for digit_index in array:
        print("it:" + str(digit_index))
        d = float("inf")
        result = 0
        for index in array:
            if np.array_equal(data[index], data[digit_index]):
                continue
            dist = abs(getDistance(data[index], data[digit_index]))
            if dist < d:
                d = dist
                result = index
        #Genuine match
        if labels[result] == labels[digit_index]:
            genuine_matches+=1
        else:
            impostor_matches+=1
    print(str(genuine_matches))
    print(str(impostor_matches))

def main():
    with open("data/train.csv") as file_object:
        lines = file_object.readlines()
    
    total_images_count = len(lines)-1
    test_data = np.zeros([total_images_count,784])
    labels_test_data = []
    i = 0
    
    for line in lines:
        sample = line.split(',')
        if i > 0 & i < total_images_count:
            labels_test_data.append(int(sample[0]))
            j = 0
            for j in range(783):
                test_data[i-1,j] =  int(sample [j+1])
        i = i+1
    

    
    #printNearestNeighbourOfEachDigit(labels_test_data, test_data)
    
    #Print all the digits, one of each label
    #for item in getOneSampleOfEachDigit(labels_test_data, test_data):
    #    displayDigit(item)
    
    #distances, indices  = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(test_data).kneighbors(test_data)
    #print(index)
    
    #displayDigit(test_data[1,:])
    
main()