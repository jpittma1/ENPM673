#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Homework #1; Problem # 4

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from sympy import N
from numpy import linalg as LA
import csv

def computeSVD(A):
    #STEP 1: solve for U matrix
    #A.AT
    AT= A.transpose()
    AAT = A.dot(AT)
    
    #get eigenvalues and eigenvectors
    eig_val_U,eig_vect_U=LA.eig(AAT)
    
    #sort values and vectors
    sort_U=eig_val_U.argsort()[::-1]
    sorted_val_U=eig_val_U[sort_U]
    # print("sorted val U shape", sorted_val_U.shape) #8
    U=eig_vect_U[:,sort_U]
        
    #STEP 2: Solve for V transpose matrix (VT)
    #AT.A
    ATA = AT.dot(A)
    eig_val_V,eig_vect_V=LA.eig(ATA)
    sort_V=eig_val_V.argsort()[::-1]
    sorted_val_U=eig_val_V[sort_V]
    V=eig_vect_V[:,sort_V]
    VT=V.transpose()
    
    
    # sorted_val_U=eigenvalue[sort]
    
    #remove negative eigenvalues
    for i in range(len(sorted_val_U)):
        if sorted_val_U[i]<=0:
            sorted_val_U[i]*=-1
            
    #STEP 3: Solve for Sigma Matrix (E)
    #square-rooted diagonal matrix of U eigenvalues squared
    diag=np.diag((np.sqrt(sorted_val_U)))
    # print("diag shape", diag.shape) #9x9

    sigma=np.zeros_like(A).astype(np.float64) #8x9
   
    for i in range(len(sigma)):
        sigma[i][i]=diag[i][i]
        
    # print("sigma shape", sigma.shape) #8x9
    
    #STEP 4: solve for Homography Matrix (H)
    #U.E.VT
    H=V[:,8]  #H is last column of V
    H=np.reshape(H,(3,3)) #assigment wants H in a 3x3 matrix format
    
    return VT, U, sigma, H


x1,x2,x3,x4,y1,y2,y3,y4= 5,150,150,5,5,5,150,150
xp1,xp2,xp3,xp4,yp1,yp2,yp3,yp4 = 100,200,220,100,100,80,80,200

A = np.array([[-x1,-y1,-1,0,0,0,x1*xp1,y1*xp1,xp1],
              [0,0,0,-x1,-y1,-1,x1*yp1,y1*yp1,yp1],
              [-x2,-y2,-1,0,0,0,x2*xp2,y2*xp2,xp2],
              [0,0,0,-x2,-y2,-1,x2*yp2,y2*yp2,yp2],
              [-x3,-y3,-1,0,0,0,x3*xp3,y3*xp3,xp3],
              [0,0,0,-x3,-y3,-1,x3*yp3,y3*yp3,yp3],
              [-x4,-y4,-1,0,0,0,x4*xp4,y4*xp4,xp4],
              [0,0,0,-x4,-y4,-1,x4*yp4,y4*yp4,yp4]])

VT, U, sigma, H = computeSVD(A)


print("U matrix is: ", U)
print("V transpose (VT) matrix is: ", VT)
print("Sigma (E) matrix is: ", sigma)
print("Homography (SVD) matrix is: ", H)
