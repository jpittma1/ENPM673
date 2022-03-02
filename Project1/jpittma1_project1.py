#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #1

#********************************************
#Requires the following in same folder to run:
# 1) "functions.py"
# 2) "testudo.png"
# 3) "1tagvideo.mp4"
#********************************************
from functions import * 

#Provided K matrix for Cube embedding
K=np.array([[1346.10059534175,0,932.163397529403],
   [0,1355.93313621175,654.898679624155],
   [0,0,1]])

#----to toggle making Videos----
#Recommend only 1 at a time
show_contours = False
show_Testudo = False
show_cube = False

#---Read the video, save a frame
thresHold=180
start=1 #start video on frame 1
vid=cv2.VideoCapture('1tagvideo.mp4')

#--Read and save Testudo image
testudo=cv2.imread('testudo.png')
# cv2.imshow('image', testudo)

#---Values for making videos
if show_contours == True or show_Testudo == True or show_cube == True:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # today = time.strftime("%m-%d__%H.%M.%S")
    fps_out = 29
    print("Making a video...this will take some time...")

if show_contours == True:
    videoname=('jpittma1_proj1_problem1_contours')
    output = cv2.VideoWriter(str(videoname)+".avi", fourcc, fps_out, (1920, 1080))

if show_Testudo == True:
    videoname1=('jpittma1_proj1_problem2_testudo')
    out1 = cv2.VideoWriter(str(videoname1)+".avi", fourcc, fps_out, (1920, 1080))
                
if show_cube == True:
    videoname2=('jpittma1_proj1_problem2_cube')
    out2 = cv2.VideoWriter(str(videoname2)+".avi", fourcc, fps_out, (1920, 1080))


# move the video to the start frame and adjust the counter
vid.set(1,start)
count = start

# print("success?", success)

if (vid.isOpened() == False):
    print('Please check the file name again and file location!')

while(vid.isOpened()):
    count+=1
    success,image1 = vid.read()
    
    if success:
        #----STEP 1: Get corners/contours----------
        [all_contours, sorted_contours] = findContours(image1,thresHold)
        # print ("Found Contours of AR tag")
        
        #---STEP 2: Find corners of tag
        #approximate quadralateral to each contour and extract corners
        [tag_contours,corners] = approxInnerGrid(sorted_contours)
        # print ("Corners AR tag 2x2 grid are: ", corners) 
    
        #To save frame 133 for FFT pictures
        if count==133:
            cv2.imwrite("1tagvideo_frame%d.jpg" % count, image1)     # save frame as JPEG file           

        #Draw contours/corners on frames of movie
        if show_contours == True:
            img_plus_contours=image1.copy()
            cv2.drawContours(img_plus_contours,all_contours,-1,(0,255,0), 4) #Green
            cv2.drawContours(img_plus_contours,tag_contours,-1,(255,0,0), 4) #Blue
            
            #--Draw circles on corners---
            grey = cv2.cvtColor(img_plus_contours, cv2.COLOR_BGR2GRAY)
            c = cv2.goodFeaturesToTrack(grey, 50, 0.1, 10)
            c = np.int0(c)
    
            #Draw circles on corners
            for i in c:
                x, y = i.ravel()
                cv2.circle(img_plus_contours, (x, y), 3, (0, 0, 255), -2)
             
        #------------1b: AR Code Decode------------------       
        #---STEP 3: Get Information from Tag (make into 4 cells)-------
        #getAR tag id in binary and orientation
        for i,tag in enumerate(corners):
            # find number of points in the polygon
            num_points = findPolyPoints(image1,tag_contours[i])
            
            # set the dimension for new warped image
            dim = int(math.sqrt(num_points))
            # print("Dimension for warping are ", dim)

            #create homography matrix for warping
            H = solveHomography(tag,dim)
            H_inv = np.linalg.inv(H) #for forward warp
            # print("Homography matrix is ", H)
            
            #warp the image of AR tag into desired dimensions
            square_img = warp(H_inv,image1,dim,dim)
            
            # threshold the squared tag image
            gray_image = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
            ret, img_warp = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY)
            
            #decode squared tile
            tag_img,tag_id_binary,orientation = decodeARtag(square_img)
            
            #----PROBLEM 2a---TESTUDO-------
            if show_Testudo == True:
                img_match_orientation=rotateTestudo(testudo, orientation)
                
                #Dimensions of frame and Testudo
                h=image1.shape[0]
                w=image1.shape[1]
                testudo_dim=img_match_orientation.shape[0]
                
                #Homography of Testudo
                H_testudo=solveHomography(tag,testudo_dim)

                #---Warp Testudo onto frame
                warped_frame=warp(H_testudo,img_match_orientation,h,w)
                
                #make the AR tag black so "blank"
                img_plus_testudo=image1.copy()
                blank_frame=cv2.drawContours(img_plus_testudo,[tag_contours[i]],-1,(0),thickness=-1)
                
                #put warped Testudo onto blank space
                img_plus_testudo=cv2.bitwise_or(warped_frame,blank_frame)
                
            #----PROBLEM 2b--CUBE------------------------
            if show_cube == True:   
                # edge_color = (0, 0, 0) #black
                edge_color = (0,0, 255) #red
                face_color = (0,0, 255) #red
                # cube_dim = 160 #200
                cube_dim=dim
                # print("dim is ", dim) #~~227-280
                
                # print("AR tag corners are: ", tag)
                
                #solve for points of cube directly above tag
                cube_height=np.array([-(cube_dim-1),-(cube_dim-1),-(cube_dim-1),-(cube_dim-1)]).reshape(-1,1)
                cube_corners = np.concatenate((tag, cube_height), axis = 1)
                # print("Cube corners are (concate): ", cube_corners)
                
                H_cube=solveHomographyCube(tag,cube_corners)
                # print("Cube Homography Matrix of Cube is: ", H_cube)
                
                P=solveProjectionMatrix(K , H_cube)
                # print("Projection Matrix of Cube is: ", P)
                
                #Project cube points above AR tag
                cube_corners_P=projectionPoints(cube_corners, P)
                # print("Projected Cube corners are: ", cube_corners_P)
                
                #draw contours, lines, and cube on top of AR tag
                img_plus_cube=drawCube(tag, cube_corners_P,image1,face_color,edge_color)
                
        #To save frame 133's as example pictures
        if count==133:
            cv2.imwrite('warped_image_frame133.jpg', img_warp)
            cv2.imwrite('AR_tag_labeled_image_frame133.jpg', tag_img)

            if show_contours==True:
                cv2.imwrite('contours_embeddeed_image_frame133.jpg', img_plus_contours)
            if show_Testudo==True:
                cv2.imwrite('testudo_embeddeed_image_frame133.jpg', img_plus_testudo)
            
            if show_cube==True:
                cv2.imwrite('cube_embeddeed_image_frame133.jpg', img_plus_cube)            
            
        tag_id_int=int(tag_id_binary, 2)
        # print("Tag ID converted to an integer is ", tag_id_int)
        
        if show_contours == True:
            output.write(img_plus_contours)
        
        if show_Testudo == True:
            out1.write(img_plus_testudo)    
                
        if show_cube == True:
            out2.write(img_plus_cube)
    
    else: #read video is not success; exit loop
        vid.release()
    
    
    # if the user presses 'q' release the video which will exit the loop
    if cv2.waitKey(1) == ord('q'):
        vid.release()
        
        if show_contours == True:
            output.release()
        if show_Testudo == True:
            out1.release()
        if show_cube == True:
            out2.release()

# vid.release()

#-----Image Creation----
frame133 = cv2.imread("1tagvideo_frame133.jpg")
img_AR_only, img_AR=removeBackground(frame133)
print ("Image with no background saved as 'img_no_background.jpg'")
print ("Image cropped saved as 'Cropped_Image.jpg'")

img_marked_up=markUpImageCorners(img_AR_only)
cv2.imwrite('corners_marked_up.jpg', img_marked_up)

conductFFTonImage(img_AR_only)
print("The image of the AR tag using FFT is saved as 'blurred_image.jpg'")

print ("Corners AR tag 2x2 grid are (final frame): ", corners) 
print("Tag ID in binary is ", tag_id_binary)
print("Tag ID converted to an integer is ", tag_id_int)

print("The image of the AR tag warped is saved as 'warped_image_frame133.jpg'")
print("The image of the AR tag numbered is saved as 'AR_tag_labeled_image_frame133.jpg'")

if show_contours == True:
    print("The image with contours is saved as 'contours_embedded_image_frame133.jpg'")
    print("Video made named 'jpittma1_proj1_problem1_contours.avi'")
        
if show_Testudo == True:
    print("The image with testudo is saved as 'testudp_embedded_image_frame133.jpg'")
    print("Video made named 'jpittma1_proj1_problem2_testudo.avi'")
                
if show_cube == True:
    print("The image with cube is saved as 'cube_embedded_image_frame133.jpg'")
    print("Video made named 'jpittma1_proj1_problem2_cube.avi'")

vid.release()
if show_contours == True:
    output.release()
if show_Testudo == True:
    out1.release()
if show_cube == True:
    out2.release()
cv2.destroyAllWindows()