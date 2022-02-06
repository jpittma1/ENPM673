#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Homework #1; Problem # 1

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from sympy import N

#----PROBLEM 1----
#--Part 1: Compute Field of View
camera_width=0.014 #14mm square shaped
camera_height=0.014
focal_length=0.025 #25mm

#angle/2=arctan(retina/2focalLength)
width_halfAngle=math.atan2(camera_width,(2*focal_length))
height_halfAngle=math.atan2(camera_height,(2*focal_length))
VFOV=2*width_halfAngle
HFOV=2*height_halfAngle

print("Horizontal Field of View is: ", N(HFOV,4))
print("Vertical Field of View is: ",N(VFOV,4))

#--Part 2: Compite minimum number of pixels object occupy in image
object_width=0.50 #5cm
object_height=object_width
z=20 #20 meters distance
resolution=5000000 #5MPix
pixel_width=math.sqrt(resolution)
pixel_height=pixel_width
#print(N(pixel_width,7))

# x=(z*pixel_width)/focal_length
# print(x)
# y=(z*pixel_height)/focal_length
# print(y)
# x_prime=(pixel_width*focal_length)/z
# # print(x_prime)
# y_prime=(pixel_height*focal_length)/z

# projected_pixels=x_prime*y_prime
# print(projected_pixels)

# print("\nBased on a 5 MPix camera with object 20 meters away, ")
# print("The object is projected at ", N(x_prime,5),  "horizontal pixels and ")
# print(N(y_prime,5), "pixels vertically, equalling ", N(projected_pixels,5), "pixels in total")

object_pixel_height=(focal_length*object_height*pixel_height)/(z*camera_height)
min_pixels_object=object_pixel_height*object_pixel_height

print("The minimum number of pixels occupied by the object is ", min_pixels_object, " or ", round(min_pixels_object), "rounded to whole pixel")