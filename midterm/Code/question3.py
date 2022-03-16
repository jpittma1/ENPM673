#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#MIDTERM

#Question #3 Part 4

import numpy as np
from numpy import linalg as LA

def solveHomography(image, world):
    #Define the eight points to compute the homography matrix
    x=[]
    y=[]
    xp=[]
    yp=[]
    # zp=[]
    
    #convert corners into x and y coordinates
    for point in image:
        x.append(point[0])
        y.append(point[1])
        
    for point in world:
        xp.append(point[0])
        yp.append(point[1])
        # zp.append(point[2])

    #make A an 8x9 matrix
    n = 9 #9 columns
    m = 8 #8 rows
    A = np.empty([m, n])

    val = 0
    for row in range(0,m):
        if (row%2) == 0: #Even rows
            A[row,0] = x[val]
            A[row,1] = y[val]
            A[row,2] = 1
            A[row,3] = 0
            A[row,4] = 0
            A[row,5] = 0
            A[row,6] = -x[val]*xp[val]
            A[row,7] = -y[val]*xp[val]
            A[row,8] = -xp[val]

        else: #odd rows
            A[row,0] = 0
            A[row,1] = 0
            A[row,2] = 0
            A[row,3] = x[val]
            A[row,4] = y[val]
            A[row,5] = 1
            A[row,6] = -x[val]*yp[val]
            A[row,7] = -y[val]*yp[val]
            A[row,8] = -yp[val]
            val += 1

    #Conduct SVD to get V
    U,S,V = np.linalg.svd(A)
    
    #Find the eigenvector column of V that corresponds to smallest value (last column)
    x=V[-1]

    # reshape x into 3x3 matrix to have H
    H = np.reshape(x,[3,3])

    return H

def solveForIntrinsicMatrix(M):
    Q, R = np.linalg.qr(M)
    return Q


'''#4: Calibrate camera using numpy only'''
#Provided values

image_pts= np.array([[757,213],[758,415],[758,686],[759,966],[1190,172],[329,1041],[1204,850],[340,159]])
# image_pts= np.array(((757,213),(758,415),(758,686),(759,966),(1190,172),(329,1041),(1204,850),(340,159)))

# print("image pts: ", image_pts)


world_pts=np.array([[0,0,0],[0,3,0],[0,7,0],[0,11,0],[7,1,0],[0,11,7],[7,9,0],[0,1,7]])
# world_pts=np.array(((0,0,0),(0,3,0),(0,7,0),(0,11,0),(7,1,0),(0,11,7),(7,9,0),(0,1,7)))

# print("world_pts: ", world_pts)

'''K=[f, 0, 0,
      0, f, 0,
      c_x, c_y, 1]
      f=focal length (pixels)
      s=skew = 0
      c=optical center (pixels)'''
# K=solveForProjectionMatrix(image_pts, world_pts)
# print("K matrix is: ", K)

M=solveHomography(image_pts, world_pts)
print("M matrix is: ", M)

#-----Solve for K----#
K=solveForIntrinsicMatrix(M)

print("K matrix is: ", K)



