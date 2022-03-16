#!/usr/bin/env python3

# ENPM673 Spring 2022
# Section 0101
# Jerry Pittman, Jr. UID: 117707120
# jpittma1@umd.edu
# MIDTERM

# Question #1

from turtle import circle
import numpy as np
import cv2
import scipy
from scipy import fft, ifft
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math


image = cv2.imread('Q1image.png')
'''Image is binary, already black and white'''


'''STEP 1: Separate coins'''

# Edge Detection
# canny = cv2.Canny(image, 0,250)
canny = cv2.Canny(image, 20, 100)

plt.imshow(canny, cmap='gray')
plt.savefig('midterm_q1_canny.png')

# Determine Kernel Size
kernel=np.ones((5,5), np.uint8)
# kernel=np.ones((3,3), np.uint8)
# kernel=(1,1)
# kernel=np.array([[0,0,0],
#                   [0,1,0],
#                   [0,0,0]])

# Try Dilation
dilate = cv2.dilate(canny, kernel, iterations=4)

plt.imshow(dilate, cmap='gray')
plt.savefig('midterm_q1_dilated.png')


gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)

plt.imshow(gradient, cmap='gray')
plt.savefig('midterm_q1_gradient.png')


# gradient2 = cv2.morphologyEx(dilate, cv2.MORPH_GRADIENT, kernel)

# plt.imshow(gradient2, cmap='gray')
# plt.savefig('midterm_q1_gradient2.png')

# Opening (erosion followed by dilation)
open = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)

plt.imshow(open, cmap='gray')
plt.savefig('midterm_q1_opening.png')

# # Closing (dilation followed by erosion)

close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

plt.imshow(close, cmap='gray')
plt.savefig('midterm_q1_closing.png')

# # Erosion
# erode = cv2.erode(canny, kernel, iterations=1)

# plt.imshow(erode, cmap='gray')
# plt.savefig('midterm_q1_erode.png')

#########-----2D Convolution------####
#try with identity kernel
# kernel2=np.array([[0,-1,0],
#                   [-1,4,-1],
#                   [0,-1,0]])
# kernel2=np.array([[0,0,0],
#                   [0,1,0],
#                   [0,0,0]])
# kernel2=np.ones((3,3), np.uint8)
kernel2=kernel
# # kernel2=(1,1)

# kernel2=kernel/(np.sum(kernel2) if np.sum(kernel2)!=0 else 1)

convo_img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
plt.imshow(convo_img, cmap='gray')
plt.savefig('midterm_q1_2dconvolution.png')

####--------Hough Circle--------##########
circles= cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, minDist=15,
                          param1=250, param2=20, minRadius=20, maxRadius=60)

# circles= cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, minDist=20,
#                           param1=250, param2=15, minRadius=25, maxRadius=60)

# print("cirles ", circles)

#Draw circles on image
circles=np.uint16(np.around(circles))
for i in circles[0,:]:
    cv2.circle(image, (i[0], i[1]), i[2], (0,255,0),2)
    
    cv2.circle(image, (i[0], i[1]), 2, (0,0,255),3)

cv2.imwrite('image_with_circles.jpg', image)
####---END Hough Circles Pipeline----#####

'''STEP 2: Count the coins'''
print("there are ",circles.shape[1], "circles")

# tmp = image.copy()

(count, hierarchy) = cv2.findContours(
    canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

canny_draw=cv2.drawContours(canny, count, -1, (0,255,0),3) #Green contour lines
cv2.imwrite('midterm_q1_canny_drawn.jpg', canny_draw)

print("Number of Coins in image (canny): ", len(count))

(count_dilate, hierarchy) = cv2.findContours(
    dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

dilate_draw=cv2.drawContours(dilate, count_dilate, -1, (0,255,0),3) #Green contour lines
cv2.imwrite('midterm_q1_dilate_drawn.jpg', dilate_draw)

print("Number of Coins in image (dilate): ", len(count_dilate))

# (count2, hierarchy) = cv2.findContours(
#     gradient, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print("Number of Coins in image (gradient): ", len(count2))

# (count3, hierarchy) = cv2.findContours(
#     close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print("Number of Coins in image (closing): ", len(count3))

# (count4, hierarchy) = cv2.findContours(
#     open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# print("Number of Coins in image (opening): ", len(count4))


# (count_convol, hierarchy) = cv2.findContours(
#     convo_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# print("Number of Coins in image (2d convolution): ", len(count_convol))

cv2.destroyAllWindows()
plt.close('all')