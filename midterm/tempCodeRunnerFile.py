circles= cv2.HoughCircles((255*convo_img).astype(np.uint8), cv2.HOUGH_GRADIENT, 1, minDist=5,
#                           param1=250, param2=5, minRadius=5, maxRadius=90)