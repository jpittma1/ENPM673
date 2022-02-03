#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Homework #1; Problem # 3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from sympy import N
from numpy import linalg as LA
import csv

def createCovarianceMatrix(x_values,y_values):
    #S=1/n*sum[(x-mean)*(x-mean).T]
    n=len(ages)
    print(n)
    
    mean_x=np.mean(x_values)
    print("mean x is ",mean_x)
    mean_y=np.mean(y_values)
    sum_x=0
    sum_y=0
    i=0
    
    for i in len(ages):
        diff_x=x_values[i]-mean_x
        diff_y=y_values[i]-mean_y
        
        diff_xT=diff_x.transpose()
        diff_yT=diff_y.transpose()
        
        sum_x+=diff_x*diff_xT
        sum_y+=diff_y*diff_yT
        
    covarianceMatrix_x=(1/n)*sum_x
    covarianceMatrix_y=(1/n)*sum_y
    
    return covarianceMatrix_x, covarianceMatrix_y

#---Problem 3
#Task Fit line for age and insurance cost

#--PART 1: Solve Covariance and plot eigenvectors
#Get data from CSV
file=open('ENPM673_hw1_linear_regression_dataset.csv')
#headers are age, sex, bmi, children, smoker, region, charges
csvreader=csv.reader(file)
header=next(csvreader)
# print(header)
rows=[]
ages=[]
charges=[]
for row in csvreader:
    rows.append(row)
    ages.append(row[0])
    charges.append(row[6])

# print(rows)
# print(ages)
# print(charges)  
file.close()

#Create Covariance Matrix
S_x, S_y=createCovarianceMatrix(ages,charges)
print("covariance of x is ",S_x)
print("covariance of y is ",S_y)

