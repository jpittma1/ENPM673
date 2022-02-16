#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #1

import numpy as np
import cv2
import scipy
from scipy import fft, ifft
from numpy import linalg as LA
import matplotlib.pyplot as plt
import sys
from  PIL  import Image

#Problem#1- AR Detection and Decode

#1 a: AR Code Detection
# def findHomography():

def removeBackground(img):
    original = img.copy()

    l = 85 #255/3
    u = 255

    ed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.GaussianBlur(img, (21, 51), 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
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

    # Cropping an image
    # print("size in rows", img_rgb.shape[0])
    # print("size in columns", img_rgb.shape[1])
    cropped_image = img_rgb[520:680, 1020:1180]

    # # Display cropped image
    # cv2.imshow("cropped", cropped_image)

    # # Save the cropped image
    cv2.imwrite("Cropped_Image.jpg", cropped_image)
    
    return cropped_image

    
#Converts to grayscale, conducts FFT, finds edges
def conductFFTonImage(image):
    #STEP1: convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image_gray',image_gray)
    
    #STEP 2: blur image using FFT
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
    
    #apply mask to fft_blur
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
    plt.savefig("blurred_image.jpg")
    
    

    


#Read the video, save a frame
vid1=cv2.VideoCapture('1tagvideo.mp4')
success,image1 = vid1.read()
count = 0

# print("success?", success)

if (vid1.isOpened() == False):
    print('Please check the file name again and file location!')

while success:
    
    #To save frame 133 (frame I found has AR in correct orientation)
    if count==133:
        cv2.imwrite("1tagvideo_frame%d.jpg" % count, image1)     # save frame as JPEG file           

    success,image1 = vid1.read()
    count += 1

vid1.release()

frame = cv2.imread("1tagvideo_frame133.jpg")
# cv2.imshow('image',frame)

img_AR_only=removeBackground(frame)
print ("Image with no background saved as 'img_no_background.jpg'")
print ("Image cropped saved as 'Cropped_Image.jpg'")

conductFFTonImage(img_AR_only)
print("The image of the AR tag using FFT is saved as 'blurred_image.jpg'")

#----1b: AR Code Decode-------
def readARtagFromImage(image):
    #STEP 1: Get corners
    marked_corners=image.copy()
    
    #Get tag corners using Shi-Tomasi
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_img, 50, 0.1, 5)
    corners = np.int0(corners)
    # print("Corners of AR tag are: ", corners)
    
    #Draw circles on corners
    for i in corners:
        x, y = i.ravel()
        cv2.circle(marked_corners, (x, y), 3, (0, 0, 255), -1)
        #red circles, 1 pixel, filled in
    
    cv2.imwrite('corners_marked_up.jpg', marked_corners)
    # plt.imshow(marked_corners)
    
    #STEP 2: Get Tag from Image
    img2=gray_img.copy()
    img2 = cv2.GaussianBlur(img2, (21, 21), 0)
    
    #threshold image using THRESH_BINARY
    # ret,img_thresh = cv2.threshold(np.uint8(img2), 200 ,255,cv2.THRESH_BINARY)
    # print("Thresholds are ", img_thresh)
    
    #STEP 3: Get Information from Tag

    return corners
    # return corners, AR_info


tag_corners=readARtagFromImage(img_AR_only)
# tag_corners, tag_info=readARtagFromImage(frame)

print("The image of the AR tag corners marked up is saved as 'marked_up_image.jpg'")
# print("The April tag corners are ", tag_corners)

#STEP 4: Translate Tag information to Binary

# def decodeARcode(AR_info):
#     while not AR_info[3,3]:
#         AR_info = np.rot90(AR_info, 1)

#     # print(AR_info)
#     id_info = AR_info[1:3, 1:3]
#     id_info_flat = np.array([id_info[0,0], id_info[0,1], id_info[1,1], id_info[1,0]])
#     id = 0
#     id_bin = []
#     for i in range(4):
#         if(id_info_flat[i]):
#             id = id + 2**(i)
#             id_bin.append(1)
#         else:
#             id_bin.append(0)

#     id_bin.reverse()

#     return id, id_bin

# tag, tag_in_binary= decodeARcode(tag_info)


# print("The April tag in binary is ", tag_in_binary)
# print("The April tag translates to ", tag)

# print("The image of the AR tag thresholded and edges using FFT are saved as fft_edges_image.jpg")

if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows()


