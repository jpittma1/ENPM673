#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Homework #1; Problem 2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from sympy import N

#---PROBLEM 2----
#Use LS to fit curves of given videos
import cv2

#split red from frame image, find center of ball
def findCoordinatesCenter (image):
    a=np.empty((0,2),int)
    b=np.empty((0,2),int)
    count=0
    count1=0
    
    #converting into HSV workspace
    hsv1 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Obtaining the ranges of red values
    # Range for lower red
    lower_red = np.array([0,120,70])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv1, lower_red, upper_red)
    
    # Range for upper red
    lower_red = np.array([170,120,70])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv1,lower_red,upper_red)
    
    # Generating the final mask to detect red color
    mask = mask1+mask2

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-red regions
    result1 = cv2.bitwise_and(image, image, mask = mask)

    dx,dy,chn =result1.shape

    result1 = cv2.resize(result1,(int(dx/5),int(dy/5)))

    dx,dy,chn =result1.shape
    
    #Obtaining the top and bottom pixels of the red values
    for i in range(dx):
        for j in range(dy):
            if result1[i,j,0]!=0:
                count+=1
                if count == 1:
                    i=dx-i
                    a=np.append(a,np.array([[i,j]]),axis=0)

    for l in reversed(range(dx)):
        for k in reversed(range(dy)):
            if result1[l,k,0]!=0:
                count1+=1
                if count1 == 1:
                    l=dx-l
                    b=np.append(b,np.array([[l,k]]),axis=0)

    # c=np.append(a,b,axis=0)
    i+=1
    cv2.waitKey(1)

    return a,b

def calcStandardLeastSquares(stack):
    #x=(A.T*A)inv*A.T*b
    #B=(X.T*X)inv*(X.T*Y)
    x_axis=stack[:,0]
    y_axis=stack[:,1]
    
    x_squared=np.power(x_axis, 2)
    
    #Parabolic quadratic equation (y=ax^2 + bx +c)
    A = np.stack((x_squared, x_axis, np.ones((len(x_axis)), dtype = int)), axis = 1) 
    
    A_transpose = A.transpose()
    ATA = A_transpose.dot(A)
    ATY = A_transpose.dot(y_axis)
    ls_estimate = (np.linalg.inv(ATA)).dot(ATY)
    ls_value= A.dot(ls_estimate)
    
    return ls_value

#----Video 1----
vid1=cv2.VideoCapture('ball_video1.mp4')
success,image1 = vid1.read()
count2 = 0
x_1=[]
y_1=[]

while success:
    
    a,b = findCoordinatesCenter(image1)
    for i in range(len(a)):
        x_1=np.append(x_1,((a[i][1]+b[i][1])/2))

    for i in range(len(a)):
	    y_1=np.append(y_1,((a[i][0]+b[i][0])/2))
    
    #To save individual frames as images
    # cv2.imwrite("Vid1_frame%d.jpg" % count2, image1)     # save frame as JPEG file           
    
    success,image1 = vid1.read()
    count2 += 1
    
stack1 = np.vstack((x_1, y_1)).T
    
#---Repeat for Video 2--
vid2=cv2.VideoCapture('ball_video2.mp4')
success2,image2 = vid2.read()
count3 = 0
x_2=[]
y_2=[]

while success2:
    a,b = findCoordinatesCenter(image2)
    for i in range(len(a)):
    	x_2=np.append(x_2,((a[i][1]+b[i][1])/2))

    for i in range(len(a)):
	    y_2=np.append(y_2,((a[i][0]+b[i][0])/2))
  
    #To save individual frames as images
    # cv2.imwrite("Vid2_frame%d.jpg" % count3, image2)
    
    success2,image2 = vid2.read()
    count3 += 1
    
stack2 = np.vstack((x_2, y_2)).T

#--Calculate Standard Least Squares
y1_ls=calcStandardLeastSquares(stack1)
y2_ls=calcStandardLeastSquares(stack2)

#plotting graph 1
fig = plt.figure()
plt.title('Jerry Pittman Homework #1')
plt.subplot(121)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.scatter(x_1,y_1,c='red', label='data points')
plt.plot(x_1,y1_ls, c='blue', label='Least Squares')
plt.legend()
plt.title('Video 1')

#plotting graph 2
plt.subplot(122)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Video 2')
plt.scatter(x_2,y_2,c='pink',label='data points')
plt.plot(x_2,y2_ls, c='blue', label='Least Squares')
plt.legend()

plt.savefig('jpittma1_homework1_p2.png')
plt.show()

vid1.release()
vid2.release()
cv2.destroyAllWindows()
