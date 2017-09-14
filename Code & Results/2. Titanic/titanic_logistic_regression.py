
# coding: utf-8

# In[50]:

import csv 
get_ipython().magic('matplotlib inline')
def transformData(trainingFile, features):
    # This function will read the data in the training file and transform it 
    # into a list of lists. 
    # Let's initialize a variable to hold this list of lists
    transformData=[]
    # The function will also return a list with the labels (Survived (0/1)) for
    # each of the passengers
    labels = []
    # Now we'll set up a couple of maps. These will be used to convert the 
    # categorical variables like gender to numeric variables 
    
    
    genderMap = {"male":1,"female":2,"":1}
    # We include a key in the map for missing values
     
    embarkMap = {"C":1,"Q":2,"S":3,"":3}
    # For blank values, we are just going to choose that value that has the maximum frequency
    blank=""
    # Now we are finally ready to read the csv file
    with open(trainingFile,'r') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        lineNum=1
        # lineNum will help us keep track of which row we are in 
        for row in lineReader:
            if lineNum==1:
                # if it's the first row, just store the names in the header in a list. 
                header = row
                #print(header)
            else: 
                allFeatures=list(map(lambda x:genderMap[x] if row.index(x)==4
                               else embarkMap[x] if row.index(x)==11 else x, row))  
                if allFeatures[1]=="":
                    allFeatures[1]=2
                # Let the default age be 30
                if allFeatures[4]=="":
                    allFeatures[4]=30
                # Let the default number of companions be 0 (assume if we have no info, the passenger
                # was travelling alone)
                if allFeatures[5]=="":
                    allFeatures[5]=0
                # By eyeballing the data , the average fare seems to be around 30
                if allFeatures[8]=="":
                    allFeatures[8]=32
    
    
    # allFeatures is a list where we have converted the categorical variables to 
                # numerical variables
                featureVector = [float(allFeatures[header.index(feature)]) for feature in features]
                # featureVector is a subset of allFeatures, it contains only those features
                # that are specified by us in the function argument
                
                transformData.append(featureVector)
                labels.append(int(row[1]))
                    # if the featureVector contains missing values, skip it, else add the featureVector
                    # to our transformedData and the corresponding label to the list of labels
            lineNum=lineNum+1
        return transformData,labels
    # return both our list of feature vectors and the list of labels 
                
            
    


# In[38]:

# Let's take this for a spin now
trainingFile="/Users/navdeep/Desktop/AML/titanic_train.csv"
features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
#features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
trainingData=transformData(trainingFile,features)


# In[52]:

# We are now ready to train our Decision Tree classifier

import numpy as np
import re

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import datasets, linear_model, cross_validation
from sklearn.cross_validation import cross_val_score

X=np.array(trainingData[0])
y=np.array(trainingData[1])


scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
k = scores.mean()
print(k)


def transformtestData(trainingFile, features):
    # This function will read the data in the training file and transform it 
    # into a list of lists. 
    # Let's initialize a variable to hold this list of lists
    transformData=[]
    # The function will also return a list with the labels (Survived (0/1)) for
    # each of the passengers
    # Now we'll set up a couple of maps. These will be used to convert the 
    # categorical variables like gender to numeric variables 
    
    ids = []
    genderMap = {"male":1,"female":2,"":1}
    # We include a key in the map for missing values
     
    embarkMap = {"C":1,"Q":2,"S":3,"":3}
    # For blank values, we are just going to choose that value that has the maximum frequency
    # Now we are finally ready to read the csv file
    with open(trainingFile,'r') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        lineNum=1
        # lineNum will help us keep track of which row we are in 
        for row in lineReader:
            if lineNum==1:
                # if it's the first row, just store the names in the header in a list. 
                header = row
                #print(header)
            else: 
                allFeatures=list(map(lambda x:genderMap[x] if row.index(x)==3
                               else embarkMap[x] if row.index(x)==10 else x, row))  
                if allFeatures[1]=="":
                    allFeatures[1]=2
                # Let the default age be 30
                if allFeatures[3]=="":
                    print(allFeatures[3])
                    allFeatures[3]=30
                # Let the default number of companions be 0 (assume if we have no info, the passenger
                # was travelling alone)
                if allFeatures[4]=="":
                    #print(allFeatures[4])
                    allFeatures[4]=0
                # By eyeballing the data , the average fare seems to be around 30
                if allFeatures[8]==""or  re.search('[a-zA-Z]',allFeatures[8])== True :
                    allFeatures[8]=32
                
    
    
    # allFeatures is a list where we have converted the categorical variables to 
                # numerical variables
                #print(allFeatures[4])
                ids.append(allFeatures[0])
                featureVector = [float(allFeatures[header.index(feature)]) for feature in features]
                # featureVector is a subset of allFeatures, it contains only those features
                # that are specified by us in the function argument
                #print(featureVector)
                transformData.append(featureVector)
                
                    # if the featureVector contains missing values, skip it, else add the featureVector
                    # to our transformedData and the corresponding label to the list of labels
            lineNum=lineNum+1
        return transformData,ids
    # return both our list of feature vectors and the list of labels 
                
            
testFile="/Users/navdeep/Desktop/AML/titanic_test.csv"
features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
#features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
testData=transformtestData(testFile,features)


# In[52]:

ids = testData[1]


model = LogisticRegression()
model = model.fit(X,y)
predicted_data = model.predict(testData[0])



resultFile="/Users/navdeep/Desktop/AML/result_test.csv"
    
with open(resultFile,"w") as f:
    ids=testData[1]
    lineWriter=csv.writer(f,delimiter=',',quotechar="\"")
    lineWriter.writerow(["PassengerId","Survived"])#The submission file needs to have a header
    for rowNum in range(len(predicted_data)):
        try:
            lineWriter.writerow([ids[rowNum],predicted_data[rowNum]])
        except Exception as e:
            print (e)


# In[1]:

for train,test in cross_validation.kFold(len(Y), n= fo):
    model = LogisticRegression()
    model = model.fit(train, test)
    score = model.score(X, y)
    print(score)


# In[ ]:



