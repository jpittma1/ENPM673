#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #1 Functions

import numpy as np
import cv2
import scipy
from scipy import fft, ifft
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
import math
import time

def findContours(frame,threshold):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgray= cv2.medianBlur(imgray,5)
    ret, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print("(frame) contours found are ", contours)
    
    # remove any contours that do not have a parent or child
    #removes background and paper contours so can focus on AR tag
    wrong_contours = []
    for i,h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:
            wrong_contours.append(i)
    contours_clean = [c for i, c in enumerate(contours) if i not in wrong_contours]

    # sort the contours to include only the three largest
    contours_clean = sorted(contours_clean, key = cv2.contourArea, reverse = True)[:3]

    # print("(frame) sorted contours are ", contours_clean)
    
    return [contours,contours_clean]

def approxInnerGrid(contours):
    AR_tag_contours = []
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri*.015, True)
        # if the countour can be approximated by a polygon with four sides include it
        if len(approx) == 4:
            AR_tag_contours.append(approx)

    # print("AR tag contours are: ", AR_tag_contours)
    
    corners = []
    for shape in AR_tag_contours:
        coords = []
        for p in shape:
            coords.append([p[0][0],p[0][1]])
        corners.append(coords)
    # print("Corners of AR tag 2x2 grid are: ", corners)
    
    return AR_tag_contours,corners

def removeBackground(img):
    original = img.copy()

    #-------Remove the background--------
    l = 85 #255/3
    u = 255

    ed = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(ed, (21, 51), 3)
    # edges = cv2.cvtColor(orginal, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(edges, l, u)

    _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)

    data = mask.tolist()
    sys.setrecursionlimit(10**8)
    for i in  range(len(data)):
        for j in  range(len(data[i])):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
        for j in  range(len(data[i])-1, -1, -1):
            if data[i][j] !=  255:
                data[i][j] =  -1
            else:
                break
    image = np.array(data)
    image[image !=  -1] =  255
    image[image ==  -1] =  0

    mask = np.array(image, np.uint8)

    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask ==  0] =  255
    cv2.imwrite('img_no_background.jpg', result)
    # cv2.imshow('img_no_background.jpg', result)
    
    img_rgb=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    #---Cropping the background and white paper from image--------
    # print("size in rows", img_rgb.shape[0])
    # print("size in columns", img_rgb.shape[1])
    cropped_image = img_rgb[520:680, 1020:1180]

    # Display cropped image
    # cv2.imshow("cropped", cropped_image)

    # Save the cropped image
    cv2.imwrite("Cropped_Image.jpg", cropped_image)
    
    return cropped_image, result

def conductFFTonImage(image):
    #STEP1: convert to grayscale-------
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image_gray',image_gray)
    
    #STEP 2: blur image using FFT------------
    image_grey = image_gray.copy()
    
    fft_blur = scipy.fft.fft2(image_grey, axes = (0,1))
    fft_shift_blur = scipy.fft.fftshift(fft_blur)
    mag_fft_shift_blur = 20*np.log(np.abs(fft_shift_blur))

    #---Create Mask using Gaussian---
    image_size=image_grey.shape
    sigma_x=50          #arbitrary values of sigma
    sigma_y=50
    cols, rows = image_size
    x_center=rows / 2
    y_center = cols / 2
    x = np.linspace(0, rows, rows)
    y = np.linspace(0, cols, cols)
    X, Y = np.meshgrid(x,y)
    mask = np.exp(-(np.square((X - x_center)/sigma_x) + np.square((Y - y_center)/sigma_y)))
    
    #apply mask to fft_blurred image
    fft_masked_blur = fft_shift_blur * mask
    mag_masked_blur = 20*np.log(np.abs(fft_masked_blur))

    #inverse FFT for background
    img_back_blur = scipy.fft.ifftshift(fft_masked_blur)
    img_back_blur = scipy.fft.ifft2(img_back_blur)
    img_back_blur = np.abs(img_back_blur)

    #Plot images of gray, FFT, Mask, and blurred Images
    fig1, plts1 = plt.subplots(2,2,figsize = (15, 10))
    plts1[0][0].imshow(image_gray, cmap = 'gray')
    plts1[0][0].set_title('Gray Image')
    plts1[0][1].imshow(mag_fft_shift_blur, cmap = 'gray')
    plts1[0][1].set_title('FFT of Gray Image')
    plts1[1][0].imshow(mag_masked_blur, cmap = 'gray')
    plts1[1][0].set_title('Mask + FFT of Gray Image')
    plts1[1][1].imshow(img_back_blur, cmap = 'gray')
    plts1[1][1].set_title('Blurred Image')
    plt.savefig("blurred_image_compare.jpg")

# find number of points in the polygon
def findPolyPoints(frame,contour):
    height = frame.shape[0]
    length = frame.shape[1]
    matrix =np.zeros((height,length),dtype=np.int32)
    cv2.drawContours(matrix,[contour],-1,(1),thickness=-1)
    indexes=np.nonzero(matrix)
    poly_points = len(indexes[0])
    return poly_points

#solve for the homography matrix
def solveHomography(corners, dimension):
    #Define the eight points to compute the homography matrix
    x = []
    y = []
    
    #convert corners into x and y coordinates
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    
    xp=[0,dimension,dimension,0]
    yp=[0,0,dimension,dimension]

    #make A an 8x9 matrix
    n = 9 #9 columns
    m = 8 #8 rows
    A = np.empty([m, n])
    
    #A matrix is:
    # Even rows (0,2,4,6): [[-x, -y, -1,0,0,0, x*x', y*x', x'],
    # Odd rows (1,3,5,7): [0,0,0, -x, -y, -1, x*y', y*y', y']]

    val = 0
    for row in range(0,m):
        if (row%2) == 0: #Even rows
            A[row,0] = -x[val]
            A[row,1] = -y[val]
            A[row,2] = -1
            A[row,3] = 0
            A[row,4] = 0
            A[row,5] = 0
            A[row,6] = x[val]*xp[val]
            A[row,7] = y[val]*xp[val]
            A[row,8] = xp[val]

        else: #odd rows
            A[row,0] = 0
            A[row,1] = 0
            A[row,2] = 0
            A[row,3] = -x[val]
            A[row,4] = -y[val]
            A[row,5] = -1
            A[row,6] = x[val]*yp[val]
            A[row,7] = y[val]*yp[val]
            A[row,8] = yp[val]
            val += 1

    #Conduct SVD to get V
    U,S,V = np.linalg.svd(A)
    
    #Find the eigenvector column of V that corresponds to smallest value (last column)
    x = V[-1] #9 values

    # reshape x into 3x3 matrix to have H
    H = np.reshape(x,[3,3])
    return H

#Warps image using Homography matrix
def warp(H,image,h,w):
    # create indexes of the destination image and linearize
    ind_y, ind_x = np.indices((h, w), dtype=np.float32)
    index_linearized = np.array([ind_x.ravel(), ind_y.ravel(), np.ones_like(ind_x).ravel()])

    # warp the coordinates of src to those of true_dst
    map_ind = H.dot(index_linearized)
    map_x, map_y = map_ind[:-1]/map_ind[-1] 
    map_x = map_x.reshape(h,w).astype(np.float32)
    map_y = map_y.reshape(h,w).astype(np.float32)

    # generate new image
    warped_img = np.zeros((h,w,3),dtype="uint8")    

    map_x[map_x>=image.shape[1]] = -1
    map_x[map_x<0] = -1
    map_y[map_y>=image.shape[0]] = -1
    map_x[map_y<0] = -1

    for new_x in range(w):
        for new_y in range(h):
            x = int(map_x[new_y,new_x])
            y = int(map_y[new_y,new_x])

            if x == -1 or y == -1:
                pass
            else:
                warped_img[new_y,new_x] = image[y,x]

    
    
    return warped_img

def decodeARtag(frame):
    dim = frame.shape[0]
    april_img = np.zeros((dim,dim,3), np.uint8)
    grid_size = 8
    k = dim//grid_size
    sx = 0
    sy = 0
    font = cv2.FONT_HERSHEY_TRIPLEX
    decode = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            roi = frame[sy:sy+k, sx:sx+k]
            if roi.mean() > 255//2:  #if white, code as a "1"
                decode[i][j] = 1
                cv2.rectangle(april_img,(sx,sy),(sx+k,sy+k),(255,255,255),-1)
            cv2.rectangle(april_img,(sx,sy),(sx+k,sy+k),(127,127,127),1)
            sx += k
        sx = 0
        sy += k
    # Id of April Tag is contained in the inner four elements of the 8x8 tag
    # a  b
    # d  c
    a = str(int(decode[3][3]))
    b = str(int(decode[3][4]))
    c = str(int(decode[4][4]))
    d = str(int(decode[4][3]))

    #Add the binary value as text to the appropriate cell
    cv2.putText(april_img,a,(3*k+int(k*.3),3*k+int(k*.7)),font,.6,(227,144,27),2)
    cv2.putText(april_img,b,(4*k+int(k*.3),3*k+int(k*.7)),font,.6,(227,144,27),2)
    cv2.putText(april_img,d,(3*k+int(k*.3),4*k+int(k*.7)),font,.6,(227,144,27),2)
    cv2.putText(april_img,c,(4*k+int(k*.3),4*k+int(k*.7)),font,.6,(227,144,27),2)

    #Determine orientation of image to properly add up binary values
    #Correct orientation has bottom right corner of AR tag as white
    ##ordered clockwise direction starting from top left square 
    #   (least significant bit) to bottom left as most significant bit
    if decode[5,5] == 1: #Bottom-right (BR) corner is at TR
        orientation = 3
        id_binary = a+b+c+d
        center = (5*k+(k//2),5*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    elif decode[2,5] == 1: #Bottom-right (BR) corner is at TL
        orientation = 2
        id_binary = d+a+b+c
        center = (5*k+(k//2),2*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    elif decode[2,2] == 1: #Bottom-right (BR) corner is at BL
        orientation = 1
        id_binary = c+d+a+b
        center = (2*k+(k//2),2*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    elif decode[5,2] == 1: #Bottom-right (BR) corner is at BR
        orientation = 0
        id_binary = b+c+d+a
        center = (2*k+(k//2),5*k+(k//2))
        cv2.circle(april_img,center,k//4,(0,0,255),-1)
    else:  #Just in case used on different image
        orientation = 0
        id_binary = '0000'
    
    return april_img,id_binary,orientation

def markUpImageCorners(image):
    #----STEP 1: Get corners----------
    marked_corners=image.copy()
    
    #Get tag corners using Shi-Tomasi
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img, 50, 0.1, 10)
    corners = np.int0(corners)
    # print("Corners of AR tag are: ", corners)
    
    #Draw circles on corners
    for i in corners:
        x, y = i.ravel()
        cv2.circle(marked_corners, (x, y), 3, (0, 0, 255), -1)
        #red circles, 1 pixel, filled in
    
    # cv2.imwrite('corners_marked_up.jpg', marked_corners)
    return marked_corners

#-------Problem 2 only functions----------
#To orientate Testudo image to match the orientation of the AR tag thoughout the movie
#   specifically up, down, left, and right
def rotateTestudo(image, orientation):
    if orientation == 1:
        new_img = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 2:
        new_img = cv2.rotate(image,cv2.ROTATE_180)
    elif orientation == 3:
        new_img = cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        new_img = image

    return new_img

def solveProjectionMatrix(K , H):
    h_1=H[:,0]
    h_2=H[:,1]

    K=np.transpose(K)

    K_inv=np.linalg.inv(K)
    a=np.dot(K_inv,h_1)
    c=np.dot(K_inv,h_2)

    #Eq. 16
    #lambda is average length of the first two columns of B
    lamda=1/((np.linalg.norm(a)+np.linalg.norm(c))/2)

    Bhat=np.dot(K_inv,H)

    #Eq.15
    #Ensure positive Bhat
    if np.linalg.det(Bhat)>0:
        B=1*Bhat
    else:
        B=-1*Bhat

    #Solve for rotation matrix and translation vector
    b_1=B[:,0]
    b_2=B[:,1]
    b_3=B[:,2]
    r1=lamda*b_1
    r2=lamda*b_2
    r3=np.cross(r1,r2)
    t=lamda*b_3

    #Eq. 11: P = K * [R | t]
    projectionMatrix=np.dot(K,(np.stack((r1,r2,r3,t), axis=1)))

    return projectionMatrix

#solves for points directly above tag to build a cube from
def projectionPoints(AR_corners, P, H, size):

    projected_corners=[]
    #Separate corners of AR tag into x and y coordinates
    x = []
    y = []
    for point in AR_corners:
        x.append(point[0])
        y.append(point[1])

    # Camera homogenous coordinates
    # X_c1=[x1, x2,x3, x4],
    #      [y1, y2,y3, y4],
    #      [1,  1,  1,  1]
    
    #AR tag coordinates in image plane
    X_c1 = np.stack((np.array(x),np.array(y),np.ones(len(x))))
    # print("skewed camera bottom corner points ",X_c)

    #scaled homogenous coordinates in world frame
    sX_s=np.dot(H,X_c1)

    #normalize so last row is "1"
    X_s=sX_s/sX_s[2]

    #points shifted in world frame
    # x(w)= [0, 0, −1,1]T
    #- z direction
    X_w=np.stack((X_s[0],X_s[1],np.full(4,-size),np.ones(4)),axis=0)
    print("skewed camera top corner points ",X_w)

    #use Projection Matrix to shift back to camera frame
    sX_c2=np.dot(P,X_w)

    #camera frame homography
    X_c2=sX_c2/sX_c2[2]

    for i in range(4):
        projected_corners.append([int(X_c2[0][i]),int(X_c2[1][i])])
        
    return projected_corners

def connectCubeCornerstoTag(AR_corners,cube_corners):
    lines = []
    #point 1 (i=0): (0,0), (1,0), (0,1), (1,1)
    
    for i in range(len(AR_corners)):
        if i==3:
            p1 = AR_corners[i]
            p2 = AR_corners[0]
            p3 = cube_corners[0]
            p4 = cube_corners[i]
        else:
            p1 = AR_corners[i]
            p2 = AR_corners[i+1]
            p3 = cube_corners[i+1]
            p4 = cube_corners[i]
        # print("Cube to AR points ", [p1,p2,p3,p4])
         #build array of connecting lines   
        lines.append(np.array([p1,p2,p3,p4], dtype=np.int32))
        # print("Current contours ", contours[i])
        #append tag corners and top square corners
    lines.append(np.array([AR_corners[0],AR_corners[1],AR_corners[2],AR_corners[3]], dtype=np.int32))
    lines.append(np.array([cube_corners[0],cube_corners[1],cube_corners[2],cube_corners[3]], dtype=np.int32))

    return lines

#draw cube based on scaled coordinates of cube points
def drawCube(bottom, top,frame,face_color,edge_color):
    thickness=3
    #-1 for fill; 0 for transparent
    # cv2.drawContours(frame,[bottom],0,face_color,thickness)
    # cv2.drawContours(frame,[top],0,face_color,thickness)
    
    #Lines connecting top and bottom of cube
    sides= connectCubeCornerstoTag(bottom, top)
    for s in sides: #red faces of cube
        cv2.drawContours(frame,[s],0,face_color,thickness)
        # cv2.drawContours(frame,[contour],-1,face_color,thickness=-1)

    for i in range(4): #black lines
        cv2.line(frame, (bottom[i,0],bottom[i,1]),(top[i,0],top[i,1]),edge_color,thickness)
    #     cv2.line(frame,tuple(tag_corners[i]),tuple(tag_corners[0]),edge_color,thickness)
    #     cv2.line(frame,tuple(cube_corners[i]),tuple(cube_corners[0]),edge_color,thickness)
    
    for i, point in enumerate(bottom):
        cv2.line(frame, tuple(point), tuple(top[i]), edge_color, thickness) 

    #draw square at top of cube and around AR tag (bottom of cube)
    for i in range (4):
        if i==3: #connect last corner to first corner
            cv2.line(frame,tuple(tag_corners[i]),tuple(tag_corners[0]),edge_color,thickness)
            cv2.line(frame,tuple(cube_corners[i]),tuple(cube_corners[0]),edge_color,thickness)
        else:
            cv2.line(frame,tuple(tag_corners[i]),tuple(tag_corners[i+1]),edge_color,thickness)
            cv2.line(frame,tuple(cube_corners[i]),tuple(cube_corners[i+1]),edge_color,thickness)

    return frame