#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

#Print header
print("Grigor Sargsyan")
print()

#open csv files
input_train = "iris-training-data.csv"
input_test  = "iris-testing-data.csv"

# Parse data from training and testing data
training_attributes = np.loadtxt(input_train, delimiter=',', usecols=(0,1,2,3))
training_labels = np.loadtxt(input_train, dtype='<U15', delimiter=',', usecols=(4))

testing_attributes = np.loadtxt(input_test, delimiter=',', usecols=(0,1,2,3))
testing_labels = np.loadtxt(input_test, dtype='<U15', delimiter=',', usecols=(4))

# Compute predicted labels and accuracy
k = len(training_labels)
min_distance_indx = 0

#function that calculates the distance between two rows
def getDistance(trainingRow, testingRow):
    dist = 0.0;
    for x in range(4):
        dist = dist + np.power(trainingRow[x] - testingRow[x], 2)
    return float(np.sqrt(dist))

#function finds the closest neighbor from training table for given row in testing
def getNearestNeighbor(trainingTable, testingRow):
    #we start off with training table and testing row
    minDist = 100.0
    result = 0.0
    closestNeighbor = 0
    #go through all training rows and compare to testing row
    for x in range(k):
        result = getDistance(trainingTable[x], testingRow)
        if result < minDist: 
            minDist = result
            #this records the row that should be testing's closest neighbor
            closestNeighbor = x
    return int(closestNeighbor)

predicted_labels = np.array(['' for i in range(k)], dtype='<U15')

#populate predicted labels
for i in range(k):
    min_distance_indx = getNearestNeighbor(training_attributes, testing_attributes[i])
    predicted_labels[i] = training_labels[min_distance_indx]

#print testing and training labels and count the percentage
percentage_counter = k
print("#, True, Predicted")
for i in range(k):
    print("%d,%s,%s" % (i+1,testing_labels[i],predicted_labels[i]))
    #count number of mistakes
    if testing_labels[i] != predicted_labels[i]:
        percentage_counter -= 1

print("Accuracy: %.2f%%" % ((percentage_counter/k)*100))

