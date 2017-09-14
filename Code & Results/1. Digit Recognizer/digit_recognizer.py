#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:38:53 2017

@author: vicenterotman
"""
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import operator
import csv

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
        
def getKNearestNeighbours(k, digit, train_data):
    d = float("inf")
    i = 0
    index = -1
    results = [0]*k
    for item in train_data:
        index+=1
        if np.array_equal(item, digit):
            continue
        dist = abs(getDistance(item, digit))
        if dist < d:
            d = dist
            results[i%k] = index
            i+=1
    return results
        
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

def getImpostorMatches(digit1, digit2, labels, data):
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
    total = str(len(genuine))
    j = 0
    for item in genuine:
        print("Now genuine, total: " + total)
        print("It: " + str(j))
        j+=1
        i = int(item/500.0)
        truePositives[i-1] +=1
    total = str(len(impostors))
    j = 0
    for item in impostors:
        print("Now impostors, total: " + total)
        print("It: " + str(j))
        j+=1
        i = int(item/500.0)
        falsePositives[i-1] +=1
        
    i = 1
    while i < 8:
        truePositives[i] += truePositives[i-1]
        falsePositives[i] += falsePositives[i-1]
        i+=1
        
        
    total_tp = truePositives[7]
    total_fp = falsePositives[7]
    
    pr_true_positive = [0,0,0,0,0,0,0,0]
    pr_false_positive = [0,0,0,0,0,0,0,0]
    
    i = 1
    while i < 8:
        pr_true_positive[i] = truePositives[i]/float(total_tp)
        pr_false_positive[i] = falsePositives[i]/float(total_fp)
        i+=1
    
    
    plt.plot(pr_false_positive, pr_true_positive)
    plt.plot([0.0,1.0],[1.0,0.0])
    plt.show()
    
def displayROC(array1, array2):
    plt.plot(array1, array2)
    plt.show()
    
def kNearestNeighboursClassifier(k, train_data, train_labels, test_data):
    print("KNN Classifier")
    test_labels = []
    i = 0
    for digit in test_data:
        print("Data size: " + str(len(test_data)))
        print("it: " + str(i))
        i+=1
        votes = [0,0,0,0,0,0,0,0,0,0]
        neighbours = getKNearestNeighbours(k, digit, train_data)
        for neighbour in neighbours:
            votes[train_labels[neighbour]]+=1
        index, value = max(enumerate(votes), key=operator.itemgetter(1))
        test_labels.append(index)
    return test_labels
    
def accuracyPercentage(testLabels, realLabels):
    print("Calculating Accuracy percentage")
    i = 0
    gL = 0
    while i < len(realLabels):
        if realLabels[i] == testLabels[i]:
            gL+=1
        i+=1

    return (gL*100)/float(len(realLabels))
    
def threeFoldCrossValidationKNN(k, data, labels):
    size = int(len(data)/3)
    a1= []
    l1= []
    a2= []
    l2= []
    a3= []
    l3= [] 
    i = 0
    print("Dividing datasets")
    while i < len(data):
        if i < size:
            print("1) it: " + str(i) + ", total: " + str(size))
            a1.append(data[i])
            l1.append(labels[i])
        elif i < 2*size:
            print("2) it: " + str(i) + ", total: " + str(size*2))
            a2.append(data[i])
            l2.append(labels[i])
        else:
            print("3) it: " + str(i) + ", total: " + str(len(data)))
            a3.append(data[i])
            l3.append(labels[i])
        i+=1
    print("Accuracy 1 set")
    accuracy = accuracyPercentage(kNearestNeighboursClassifier(k, a2+a3, l2+l3, a1), l1)
    print("Accuracy 2 set")
    accuracy+= accuracyPercentage(kNearestNeighboursClassifier(k, a1+a3, l1+l3, a2), l2)
    print("Accuracy 3 set")
    accuracy+= accuracyPercentage(kNearestNeighboursClassifier(k, a1+a2, l1+l2, a3), l3)
    print("Accuracy Done")
    return accuracy/3

def generateConfusionMatrix(real_labels, predicted_labels):
    w, h = 10, 10;
    matrix = [[0 for x in range(w)] for y in range(h)]
    printMatrix(matrix)
    i=0
    while i < len(real_labels):
        print(str(predicted_labels[i]))
        print(str(real_labels[i]))
        matrix[predicted_labels[i]][real_labels[i]]+=1
        i+=1
    return matrix

def printMatrix(matrix):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in matrix]))

def main():    
    with open("train.csv") as file_object:
        lines = file_object.readlines()
    
    total_images_count = len(lines)-1
    data = np.zeros([total_images_count,784])
    labels_data = []
    i = 0
    
    for line in lines:
        sample = line.split(',')
        if i > 0 & i < total_images_count:
            labels_data.append(int(sample[0]))
            j = 0
            for j in range(783):
                data[i-1,j] =  int(sample [j+1])
        i = i+1
        
    with open("test.csv") as file_object:
        lines = file_object.readlines()
    
    total_images_count = len(lines)-1
    test_data = np.zeros([total_images_count,784])
    i = 0
    
    for line in lines:
        sample = line.split(',')
        if i > 0 & i < total_images_count:
            j = 0
            for j in range(783):
                test_data[i-1,j] =  int(sample [j+1])
        i = i+1
    print("Data getted")
    
    results = kNearestNeighboursClassifier(3, data, labels_data, test_data)
    with open("results.csv", 'wb') as myfile:
		writer = csv.writer(myfile, delimiter=',')
		writer.writerow(('ImageId' , 'Label'))
		i = 1
		for item in results:
			writer.writerow((i,item))
			i+=1

#    knn_labels = kNearestNeighboursClassifier(3, data, labels_data, data)
#    print("KNN done")
#    matrix = generateConfusionMatrix(labels_data, knn_labels)
#    printMatrix(matrix)

    #print("Accuracy 3 fold cross validation k=3: " + str(threeFoldCrossValidationKNN(3, data, labels_data)))
    
#    test = [data[40], data[41], data[440], data[4670], data[2], data[87], data[658]]
#    l = [labels_data[40], labels_data[41], labels_data[440], labels_data[4670], labels_data[2], labels_data[87], labels_data[658]]
#    accuracyPercentage(kNearestNeighboursClassifier(7, data, labels_data, test), l)
    
    #displayDigit(data[40])
    #print(kNearestNeighboursClassifier(7, test_data, labels_data, [data[40]]))
    
    #for digit in getOneSampleOfEachDigit(labels_data, data):
    #    displayDigit(digit)
    
    #a1 = getImpostorMatches(0,1,labels_data,data)
    #a2 = getGenuineMatches(0,1,labels_data,data)
    #getAndDisplayROCCurve(a2, a1)

#true positive    
#[927268, 3366353, 4945431, 5396683, 3992071, 851487, 10399, 12540]
#[927268, 4293621, 9239052, 14635735, 18627806, 19479293, 19489692, 19502232]
#pr
#[0, 0.22016049239902385, 0.47374331307308826, 0.7504646134862922, 0.9551627731636051, 0.9988237756580888, 0.9993569966760728, 1.0]

#false positive
#[0, 23, 24147, 1725358, 10692027, 6576643, 334661, 1429]
#[0, 23, 24170, 1749528, 12441555, 19018198, 19352859, 19354288]
#pr
#[0, 1.1883671463398705e-06, 0.0012488188663928117, 0.09039485203485656, 0.6428319657121977, 0.9826348559037666, 0.9999261662325165, 1.0]  

#true positive prob
#[1.0, 0.9999931677269562, 0.9951410361201696, 0.7577438826875611, 0.27186354926261047, 0.11463006167097237, 0.030136787805019417, 0.8977020545493593]
#false positive prob
#[0.0, 6.8322730437717e-06, 0.00485896387983044, 0.24225611731243896, 0.7281364507373895, 0.8853699383290277, 0.9698632121949806, 0.1022979454506407]
   
    
    
    #displayDoubleHistogram(getGenuineMatches(0,1,labels_data,data),
    #                       getImpostorMatches(0,1,labels_data,data), "Genuine", "Impostor")
    
    
    #printNearestNeighbourOfEachDigit(labels_data, data)
    
    #Print all the digits, one of each label
    #for item in getOneSampleOfEachDigit(labels_data, data):
    #    displayDigit(item)
    
    #distances, indices  = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data).kneighbors(data)
    #print(index)
    
    #displayDigit(data[1,:])
    
main()