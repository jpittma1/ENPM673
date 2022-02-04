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

def solveCovariance(A):
    #S=1/(n-1)*sum[(x-mean)*(x-mean).T]
    # print("mean is matrix is ", A.mean(axis=0))
    A-=A.mean(axis=0)
    
    n=len(A)
    norm=n-1
    
    covarianceMatrix=np.dot(A.T, A.conj())/norm
    print("covarianceMatrix is ",covarianceMatrix)
    
    eigenvalues,eigenvectors=LA.eig(covarianceMatrix)
    # print("eigenvalues are: ", eigenvalues)
    # print("eigenvectors are: ", eigenvectors)
    
    return covarianceMatrix, eigenvectors, eigenvalues

def calcStandardLeastSquares(stack):
    #x=(A.T*A)inv*A.T*b
    #B=(X.T*X)inv*(X.T*Y)
    x_axis=stack[:,0]
    y_axis=stack[:,1]
    
    x_squared=np.power(x_axis, 2)
    
    #Parabolic quadratic equation (y=ax^2 + bx +c)
    A = np.stack((x_squared, x_axis, np.ones((len(x_axis)), dtype = int)), axis = 1) 
    
    A_transpose = A.transpose()
    ATA = A_transpose.dot(A)
    ATY = A_transpose.dot(y_axis)
    ls_estimate = (np.linalg.inv(ATA)).dot(ATY)
    ls_value= A.dot(ls_estimate)
    
    return ls_value

def calcLeastSquares(A,y):
    #x=(A.T*A)inv*A.T*b
    #B=(X.T*X)inv*(X.T*Y)
    
    A_transpose = A.transpose()
    ATA = A_transpose.dot(A)
    ATY = A_transpose.dot(y)
    ls_estimate = (np.linalg.inv(ATA)).dot(ATY)

    # print("shape of RANSAC ls_estimate is ", ls_estimate.shape)
  
    return ls_estimate

def computeSVD(A):
    AT=A.T
    AAT=A.dot(AT)
    
    #solve for eigenvector and eigenvalues
    eigenvalue,U_temp=LA.eig(AAT)
    
    sort=eigenvalue.argsort()[::-1]
    eigval_sorted=eigenvalue[sort]
    
    #remove negative eigenvalues
    for i in range(len(eigval_sorted)):
        if eigval_sorted[i]<=0:
            eigval_sorted[i]*=-1
    
    U=U_temp[:,sort]
    # print("U is ", U)
    
    #solve for diagonal sigma (eigenvalues) matrix
    diag=np.diag((np.sqrt(eigval_sorted)))
    
    #inverse sigma
    diag=LA.inv(diag)
    sigma_inv=np.zeros_like(A).astype(np.float64)
    sigma_inv[:diag.shape[0],:diag.shape[1]]=diag
    # print("Sigma inverse is ", sigma_inv)
    
    #solve for (VT) transposed orthogonal eigenvectors of ATA
    #U-transpose dot sigma_inv dot A
    VT=sigma_inv.dot(U.T)
    VT=VT.dot(A)
    # print("VT is ", VT)
    
    return U, sigma_inv, VT

def calcTotalLeastSquares (x_values,y_values):
    #need to solve for x^2, x^3, x^4, xy,x^2y, xy^2
    x=x_values
    y=y_values
    #initialize tables
    x2=[] #x squared
    x3=[] #x cubed
    x4=[] #x fourth powered
    xy=[] #x*y
    x2y=[] #x squared times y
    
    #solve for terms
    for i in range(len(x)):
        x2.append(float(x[i]**2))
        x3.append(float(x[i]**3))
        x4.append(float(x[i]**4))
        xy.append(float(x[i]*y[i]))
        x2y.append(float(x[i]**2)*y[i])
        
    #initialize sums
    sum_n=0.0
    sum_x=0.0
    sum_y=0.0
    sum_x2=0.0
    sum_x3=0.0
    sum_x4=0.0
    sum_xy=0.0
    sum_x2y=0.0
    
    #solve for sums
    for i in range(len(x)):
        sum_n+=i
        sum_x+=x[i]
        sum_y+=y[i]
        sum_x2+=x2[i]
        sum_x3+=x3[i]
        sum_x4+=x4[i]
        sum_xy+=xy[i]
        sum_x2y+=x2y[i]
    
    #a, b, and c equations
    #y=ax^2+b*x+c*n
    #xy=a*x^3+b*x^2+c*x
    #x2y=a*x^4+b*x^3+c*c^2
    a=np.array([[sum_x2,sum_x,sum_n],
                [sum_x3,sum_x2,sum_x],
                [sum_x4,sum_x3,sum_x2]])
    
    b=np.array([sum_y, sum_xy, sum_x2y])
    
    U,sigma, VT= computeSVD(a)
    
    coefficients=VT.T.dot(sigma.dot(U.T.dot(b)))
    A=coefficients[0]
    B=coefficients[1]
    C=coefficients[2]
    
    tls_value=[]
    
    for i in range(0,len(x)):
        y_temp=(A*(x[i]**2))+(B*x[i])+C
        tls_value.append(y_temp)
    
    return tls_value

def calcRANSAC(array):
    x=array[:,0]
    y=array[:,1]
    
    #STEP 1: Randomly select small subset of points
    x2=np.power(x,2)
    
    #create parabolic polynomial array
    A=np.stack((x2,x,np.ones((len(x)),dtype=int)),axis=1)
    # print("shape of calcRANSAC A is ", A.shape)
    
    #create threshold for outliers vs inliers
    threshold=np.std(y)/3  #best curve when sd/3. sd/2 and sd/5 also worked okay
    
    #STEP 2: Fit model to subset points (2 points)
    ransac_model_test=ransacFit(A,y,2, threshold)
    
    ransac_solution=A.dot(ransac_model_test)
    
    #STEP 4: Repeat until have best model
    
    return ransac_solution

def ransacFit(A,y,sample_size,threshold):
    iter_max=math.inf
    iter=0
    max_inliers=0
    best_model=None
    prob_outlier=0
    prob_desired=0.95 #95 percent accurate probability
    
    data=np.column_stack((A,y))
    # print("shape of data is ", data.shape)
    
    data_size=len(data)
    # print("A is ", A)
    #STEP 3: Find all remaining points that are close to model and reject others
    #randomly iterate through data based on sample size
    while iter_max >iter:
        
        np.random.shuffle(data)
        samples=data[:sample_size,:]
        
        #create line using LS method
        #A=samples[:,:-1]  Y=samples[:,-1:]
        
        temp_matrix=samples[:,:-1]
        # print("shape of temp_matrix is ", temp_matrix.shape)
        temp_y=samples[:,-1:]
        # print("shape of temp_y is ", temp_y.shape)
        iter_model=calcLeastSquares(temp_matrix,temp_y)
        # print("iter_model " , iter_model)
        
        #count inliers within threshold
        inliers=A.dot(iter_model)
        
        err=np.abs(y-inliers.T)
        
        inlier_count=np.count_nonzero(err<threshold)
        print("Inlier Count is: ", inlier_count, "for iteration ", iter)
        
        #save model if has maximum inliers
        if inlier_count >max_inliers:
            max_inliers=inlier_count
            best_model=iter_model
            
        #Outlier probability
        prob_outlier=1-inlier_count/data_size
        
        #Update number of iterations to be less than infinity
        #based on probabilities
        iter_max =math.log(1-prob_desired)/math.log(1-(1-prob_outlier)**sample_size)
        
        iter+=1
        
    return best_model
        
    
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

#--Convert 'list' data to  int/float
age=[int(i) for i in ages]
# print("ages as integers is: ", age)

cost=[float(i) for i in charges]
# print("charges as floats is: ", cost)

A=np.column_stack((age,cost))

#---Create Covariance Matrix, find eigenvectors--

S, eig_vect, eig_val=solveCovariance(A)

#--Make Eigenvectors plottable
eig_vec1=eig_vect[:,0]
# print("eig_vec1 is ", eig_vec1)
eig_vec2=eig_vect[:,1]
# print("eig_vec2 is ", eig_vec2)

# print("x center ", np.average(age))
# print("y center ", np.average(cost))
origin=[np.mean(age),np.mean(cost)]

#plotting graph 1
fig = plt.figure(1)
plt.title('Jerry Pittman Homework #1')
plt.subplot(121)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.scatter(age,cost,c='red', label='Ages vs Insurance Cost')
plt.quiver(*origin,*eig_vec1, color=['r'],label='X Eigenvector')
plt.quiver(*origin,*eig_vec2, color=['b'],label='Y Eigenvector')
plt.legend()
plt.title('Part 1, Eigenvectors')

#----Part 2: LS, TLS, and RANSAC----
#--Calculate LS using methodology of Problem 2--
cost_ls=calcStandardLeastSquares(A)
# print("Linear Least Squares ", cost_ls)

#--Calculate TLS--
cost_tls=calcTotalLeastSquares(age,cost)

#--Calculate RANSAC--
cost_ransac=calcRANSAC(A)

# plotting of curve fitting against data
plt.subplot(122)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Part 2, compare curve fitting')
plt.scatter(age,cost,c='r',label='data points')
plt.plot(age,cost_ls, c='blue', label='Linear Least Squares')
plt.plot(age,cost_tls, c='g', label='Total Least Squares')
plt.plot(age,cost_ransac, c='m', label='RANSAC')
plt.legend()

plt.show()
plt.savefig('jpittma1_homework1_p3.png')
