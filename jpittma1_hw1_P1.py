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
width=0.014 #14mm square shaped
height=0.014
focal_length=0.025 #25mm

#angle/2=arctan(retina/2focalLength)
width_halfAngle=math.atan2(width,(2*focal_length))
height_halfAngle=math.atan2(height,(2*focal_length))
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

x=(z*pixel_width)/focal_length
y=(z*pixel_width)/focal_length

projected_pixels=x*y

print("\nBased on a 5 MPix camera, ")
print("The object is projected at ", N(x,5),  "horizontal pixels and ")
print(N(y,5), "pixels vertically, equalling ", N(projected_pixels/1000000,5), "pixels in total")
