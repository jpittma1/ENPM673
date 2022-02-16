#!/usr/bin/env python3

#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Project #1

import numpy as np
import cv2
import scipy
from numpy import linalg as LA

#Problem#2- Tracking
K=np.array([[1346.10059534175,0,932.163397529403],
   [0,1355.93313621175,654.898679624155],
   [0,0,1]])


#2a: Superimpose Image onto tag

#2b: Placing virtual cube onto tag