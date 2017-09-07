#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:38:53 2017

@author: vicenterotman
"""
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

#def getZerosAndOnesIndexes(labels):
#    array = []
#    i = 0
#    for item in labels:
#        if item == 0 or item == 1:
#            array.append(i)
#        i+=1
#    return array

#def printImpostorsAndGenuineMatchesForZerosAndOnes(labels, data):
#    array = getZerosAndOnesIndexes(labels)
#    genuine_matches = 0
#    impostor_matches = 0
#    print("total: " + str(len(array)))
#    for digit_index in array:
#        print("it:" + str(digit_index))
#        d = float("inf")
#        result = 0
#        for index in array:
#            if np.array_equal(data[index], data[digit_index]):
#                continue
#            dist = abs(getDistance(data[index], data[digit_index]))
#            if dist < d:
#                d = dist
#                result = index
#        #Genuine match
#        if labels[result] == labels[digit_index]:
#            genuine_matches+=1
#        else:
#            impostor_matches+=1
#    print(str(genuine_matches))
#    print(str(impostor_matches))

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
        
def getArrayOf(value, labels, data):
    array = []
    i = 0
    for item in labels:
        if item == value:
            array.append(data[i])
        i+=1
    return array

def getAllDistances(data):
    print("Get all distances")
    array = []
    i = 0
    print("Total of: " + str(len(data)))
    while i < len(data):
        print("it: " + str(i))
        j = i+1
        while j < len(data):
            array.append(getDistance(data[i], data[j]))
            j+=1
        i+=1
    return array

def getImpostorDistances(data1, data2):
    print("Get impostor distances")
    array = []
    i = 0
    print("Total of: " + str(len(data1)))
    while i < len(data1):
        print("it: " + str(i))
        item = data1[i]
        j = 0
        while j < len(data2):
            array.append(getDistance(item, data2[j]))
            j+=1
        i+=1
    return array

def getImportMatches(digit1, digit2, labels, data):
    return getImpostorDistances(getArrayOf(digit1, labels, data), getArrayOf(digit2, labels, data))

def getGenuineMatches(digit1, digit2, labels, data):
    return getAllDistances(getArrayOf(digit1, labels, data)) + getAllDistances(getArrayOf(digit2, labels, data))
                
def displayDoubleHistogram(array1, array2, label1, label2):    
    plt.hist(array1, label=str(label1), alpha=0.5)
    plt.hist(array2, label=str(label2), alpha=0.5)
    plt.legend(loc='upper right')
    plt.show()
    
def getAndDisplayROCCurve(genuine, impostors):
    truePositives = [0,0,0,0,0,0,0,0]
    falsePositives = [0,0,0,0,0,0,0,0]
    print("getAndDisplayROCCurve")
    for item in genuine:
        i = int(item/500.0)
        truePositives[i-1] +=1
    print("Now impostors")
    for item in impostors:
        i = int(item/500.0)
        falsePositives[i-1] +=1
        
    print("true positive")
    print(truePositives)
    
    print("false positive")
    print(falsePositives)
    
    truePositivesProb = [0,0,0,0,0,0,0,0]
    falsePositivesProb = [0,0,0,0,0,0,0,0]
    
    i = 0
    while i < 8:
        total = truePositives[i] + falsePositives[i]
        truePositivesProb[i] = truePositives[i]/float(total)
        falsePositivesProb[i] = falsePositives[i]/float(total)
        i+=1
        
    print("true positive prob")
    print(truePositivesProb)
    
    print("false positive prob")
    print(falsePositivesProb)
    
    plt.plot(falsePositivesProb, truePositivesProb)
    plt.show()
    
def displayROC(array1, array2):
    plt.plot(array1, array2)
    plt.show()
    
def getAUC(array1, array2):
    return np.trapz(array1 , array2)

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
    
    for digit in getOneSampleOfEachDigit(labels_test_data, test_data):
        displayDigit(digit)
    
    #a1 = getImportMatches(0,1,labels_test_data,test_data)
    #a2 = getGenuineMatches(0,1,labels_test_data,test_data)
    #getAndDisplayROCCurve(a2, a1)
     
    
    
    #displayDoubleHistogram(getGenuineMatches(0,1,labels_test_data,test_data),
    #                       getImportMatches(0,1,labels_test_data,test_data), "Genuine", "Impostor")
    
    
    #printNearestNeighbourOfEachDigit(labels_test_data, test_data)
    
    #Print all the digits, one of each label
    #for item in getOneSampleOfEachDigit(labels_test_data, test_data):
    #    displayDigit(item)
    
    #distances, indices  = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(test_data).kneighbors(test_data)
    #print(index)
    
    #displayDigit(test_data[1,:])
    
main()