#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#MIDTERM

#Question #4
'''Task: separate into colors'''
#K-means clustering for 4 colors

import numpy as np
import cv2
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math


######---------Functions-------########
def euclidDistance(arr1, arr2):
    #Arr1 is center points/centroids
    #Arr2 is values from image
    
    distance=np.sqrt(((arr2-arr1[:,np.newaxis])**2).sum(axis=2))
    
    return distance

'''Update Centroids of Clusters'''
def updateCentroids(old, idx, points):
    K, D = old.shape  #K-clusters by columns
    
    # print("index is ", idx) #4x1 array
    # print("K is ", K)   #verify is 4
    new = np.empty(old.shape)
    
    for i in range(K): #iterate through all clusters
        new[i] = np.mean(points[idx==i], axis = 0)
        # new[i] = np.array(points[idx==i].mean(axis = 0))
    return new

'''Solve for Loss/error between points and centroids to determine to stop iterating
K-means algorithm'''
def solveLoss(centers, idx, pts):
    # dists = euclidDistance(pts, centers)
    # print("dists is ", dists)
    
    dists=np.sqrt(((pts-centers[:,np.newaxis])**2).sum(axis=2))
    loss = 0.0
    N, D = pts.shape
    
    # print("N is ", N)
    # print("D is ", D)
    for i in range(D):
        loss = loss + np.square(dists[i][idx[i]])
    
    return loss

def solveKMeans(values, k, iter):
    print("Commencing K means algorithm....")
    print("Number of clusters (K) is:  ", k)
    '''Initialize random center'''
    r, c = values.shape
    
    centroids=np.empty([k,c]) #K x Columns
    index=np.empty([r]) #rows x 1 
    
    '''Initialize random centers'''
    for num in range(k):
            randIndex = np.random.randint(r)
            centroids[num] = values[randIndex]
            
    #error tolerance to complete K-means less than iteration number
    error= 1
    
    count=1#Set iteration count to 1
    for j in range(iter):
        dist = euclidDistance(centroids, values)
        # index=np.argmin(dist, axis=1)
        # dist = euclidDistance(centroids, values)
        index=np.argmin(dist, axis=0)

        '''Update Centroids of Clusters'''
        centroids = updateCentroids(centroids, index, values)

        loss = solveLoss(centroids, index, values)
        K = centroids.shape[0]
        if j:
            diff = np.abs(prev_loss - loss)
            if diff < error:
                break
        prev_loss = loss
        
        count+=1
    
    print("That took ",count, " iterations!!")
    
    return index, centroids

#################################################

#import image and convert to data
image=cv2.imread("Q4image.png")
image_vals=plt.imread("Q4image.png")

rows=image_vals.shape[0]
columns=image_vals.shape[1]
channels=image_vals.shape[2]

image_vals=image_vals.reshape(rows*columns, channels)

'''Select K value'''
K=4 #4 colors
iterations=50 #Max iterations, so doesn't go on forever

cluster_index, centers=solveKMeans(image_vals, K, iterations)
curr_image_vals=np.copy(image_vals)

'''Put each pixel to a Cluster'''
for i in range(0,K):
    curr_cluster_index = np.where(cluster_index==i)[0]
    curr_image_vals[curr_cluster_index] = centers[i]

curr_image_vals = curr_image_vals.reshape(rows, columns, channels)

# cv2.imwrite('midterm_q4_clusterd.jpg', curr_image_vals)

plt.imshow(curr_image_vals)
plt.savefig('midterm_q4_clusterd.png')

cv2.destroyAllWindows()
plt.close('all')