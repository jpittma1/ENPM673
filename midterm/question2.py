#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#MIDTERM

#Question #2

import numpy as np
import cv2
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math

'''Stich together two images using homography'''

imageA=cv2.imread('Q2imageA.png')
imageB=cv2.imread('Q2imageB.png')

#convert to grayscale
imageA_gray=cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY)
imageB_gray=cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY)

#detect and compute Scale Invariant Feature Transform (SIFT) keypoints
# and descriptors
sift=cv2.xfeatures2d.SIFT_create()
kpA, desA = sift.detectAndCompute(imageA_gray, None)
kpB, desB = sift.detectAndCompute(imageB_gray, None)

#Match images using Brute-Force Matcher
bf=cv2.BFMatcher()

matches=bf.knnMatch(desA, desB, k=2)
# print("matches are: ", matches)

#Apply ratio test or else get errors
good=[]
for m in matches:
    if m[0].distance <0.5*m[1].distance:
        good.append(m)

#replace matches array with good array
matches=np.asarray(good)

#Find keypoints from matches for homography
src_pts=np.float32([ kpA[m.queryIdx].pt for m in matches[:,0]]).reshape(-1,1,2)
dest_pts=np.float32([ kpB[m.trainIdx].pt for m in matches[:,0]]).reshape(-1,1,2)

# print("src_pts are: ", src_pts)
# print("dest_pts are: ", dest_pts)

# H, mask = cv2.findHomography(src_pts, dest_pts)
H, mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC)


height=imageB.shape[0]
width_new=imageB.shape[1]+imageA.shape[1]
# width=imageA.shape[1]

# print("stitched image desired height and width: ", height, ", ", width_new)
#300x800

image_stitched=cv2.warpPerspective(imageA, H, (width_new,height))
# print("stitched image shape: ", image_stitched.shape[0], ", ", image_stitched.shape[1])

#Fix warping
image_stitched[0:imageB.shape[0], 0:imageB.shape[1]] =imageB

cv2.imwrite('midterm_q2_stitch.jpg', image_stitched)
compare=np.hstack((imageA, imageB, image_stitched))
cv2.imwrite('midterm_q2_compare.jpg', compare)

cv2.destroyAllWindows()