#ENPM673 Spring 2022
#Section 0101
#Jerry Pittman, Jr. UID: 117707120
#jpittma1@umd.edu
#Homework #1; Problem # 3

from inspect import CO_VARKEYWORDS
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from sympy import N
from numpy import linalg as LA
import csv

def solveCovariance(cov_A):
    #S=1/(n-1)*sum[(x-mean)*(x-mean).T]
    x_minus_mean=cov_A-cov_A.mean(axis=0)
    
    # -PUT IN REPORT: cov_A-=cov_A.mean(axis=0)
    n=len(cov_A)
    norm=n-1
    
    covarianceMatrix=np.dot(x_minus_mean.T, x_minus_mean.conj())/norm
    print("covarianceMatrix is ",covarianceMatrix)
    
    eigenvalues,eigenvectors=LA.eig(covarianceMatrix)
    # print("eigenvalues are: ", eigenvalues)
    # print("eigenvectors are: ", eigenvectors)
    
    return covarianceMatrix, eigenvectors, eigenvalues

def calcStandardLeastSquares(points):
    #x=(A.T*A)inv*A.T*b
    #B=(X.T*X)inv*(X.T*Y)
    x_axis=points[:,0]
    # print("x_axis is ", x_axis)
    y_axis=points[:,1]
    
    #Line equation (y=ax + b)
    d = np.stack((x_axis, np.ones((len(x_axis)), dtype = int)), axis = 1) 
    
    d_transpose = d.transpose()
    dTd = d_transpose.dot(d)
    dTY = d_transpose.dot(y_axis)
    ls_estimate = (np.linalg.inv(dTd)).dot(dTY)
    ls_value= d.dot(ls_estimate)
    
    return ls_value

def calcLeastSquares(e,y):
    #x=(A.T*A)inv*A.T*b
    #B=(X.T*X)inv*(X.T*Y)
    
    e_transpose = e.transpose()
    eTe = e_transpose.dot(e)
    eTY = e_transpose.dot(y)
    ls_estimate = (np.linalg.inv(eTe)).dot(eTY)

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
    #b=w+sqrt(w^2+r^2)/r
    #w=sum(y-ymean)^2-sum(x-xmean)^2
    #r=2*sum*(x-xmean)*(y-ymean)
    #a=ymean-b*xmean
    
    #STEP 1: Create U matrix (difference from mean)
    x_tls=x_values
    y_tls=y_values
    
    n = len(x_tls)
    
    x_mean=np.mean(x_tls)
    y_mean=np.mean(y_tls)
    
    U=np.vstack(((x_tls-x_mean),(y_tls-y_mean))).T
    # print("U is ", U)
    # print("U shape is ", U.shape)
    
    #STEP 2: Create UTU matrix
    UTU=np.dot(U.transpose(),U)
    # print("UTU shape is ", UTU.shape)
    
    #STEP 3: Solve for coeffiecients of d=ax+b
    beta=np.dot(UTU.transpose(),UTU)
    
    #Get eigenvalues(w) and eigenvectors (v)
    w,v=LA.eig(beta)
    
    #find index of smallest eigenvalue
    index=np.argmin(w)
    #get corresponding eigenvector
    coefficients=v[:,index]
    # print("coefficients are", coefficients)
    # print("coefficients shape is ", coefficients.shape)
    
    a,b=coefficients
    D=a*x_mean+b
    
    tls_value=[]
    for i in range(0,n):
        # y_temp=D-(a*x_tls[i])
        y_temp=(D-(a*x_tls[i]))/b
        tls_value.append(y_temp)
        
    # print("tls_value ",tls_value )
    
    return tls_value

def calcRANSAC(array):
    xray=array[:,0]
    y=array[:,1]
    
    #STEP 1: Randomly select small subset of points
    
    #create line array
    arr=np.stack((xray,np.ones((len(xray)),dtype=int)),axis=1)
    # print("shape of calcRANSAC A is ", A.shape)
    
    #create threshold for outliers vs inliers
    threshold=np.std(y)/3  #best curve when sd/3. sd/2 and sd/5 also worked okay
    
    #STEP 2: Fit model to subset points (2 points)
    #---Repeat until have best model
    ransac_model_test=ransacFit(arr,y,2, threshold)
    
    ransac_solution=arr.dot(ransac_model_test)

    return ransac_solution

def ransacFit(arr_ransac,y,sample_size,threshold):
    iter_max=math.inf
    iter=0
    max_inliers=0
    best_model=None
    prob_outlier=0
    prob_desired=0.95 #95 percent accurate probability
    
    data=np.column_stack((arr_ransac,y))
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
        inliers=arr_ransac.dot(iter_model)
        
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
  
def plotter(orig, eig1, eig2, ages, charges, charges_ls, charges_tls, charges_ransac):
    #plotting Part 1
    fig = plt.figure(1)
    plt.title('Jerry Pittman Homework #1')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.scatter(ages,charges,c='red', label='Ages vs Insurance Cost')
    plt.quiver(*orig,*eig1, color=['r'],label='X Eigenvector')
    plt.quiver(*origin,*eig2, color=['b'],label='Y Eigenvector')
    plt.legend()
    plt.title('Part 1, Eigenvectors')
    
    # plotting of Part 2
    plt.figure(2)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Part 2, LS curve fitting')
    plt.scatter(ages,charges,c='r',label='data points')
    plt.plot(ages,charges_ls, c='blue', label='Linear Least Squares')
    plt.savefig('jpittma1_homework1_p3_LS.png')
    plt.legend()

    plt.figure(3)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Part 2, TLS curve fitting')
    plt.scatter(ages,charges,c='r',label='data points')
    plt.plot(ages,charges_tls, c='g', label='Total Least Squares')
    plt.savefig('jpittma1_homework1_p3_TLS.png')
    plt.legend()

    plt.figure(4)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Part 2, RANSAC curve fitting')
    plt.scatter(ages,charges,c='r',label='data points')
    plt.plot(ages,charges_ransac, c='k', label='RANSAC')
    plt.savefig('jpittma1_homework1_p3_RANSAC.png')
    plt.legend()

    plt.figure(5)
    plt.subplot(121)
    plt.title('Jerry Pittman Homework #1 Combined Plots')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.scatter(ages,charges,c='red', label='Ages vs Insurance Cost')
    plt.quiver(*orig,*eig1, color=['r'],label='X Eigenvector')
    plt.quiver(*orig,*eig2, color=['b'],label='Y Eigenvector')
    plt.legend()
    plt.title('Part 1, Eigenvectors')
    plt.subplot(122)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Part 2, Compare curve fittings')
    plt.scatter(ages,charges,c='r',label='data points')
    plt.plot(ages,charges_ls, c='blue', label='Linear Least Squares')
    plt.plot(ages,charges_tls, c='g', label='Total Least Squares')
    plt.plot(ages,charges_ransac, c='k', label='RANSAC')
    plt.legend()
    plt.savefig('jpittma1_homework1_p3_combined.png')

    plt.show()
          
#----Problem 3------
#Task: Fit line for age and insurance cost

#-----PART 1: Solve Covariance (manually) and plot eigenvectors------

#STEP 1: Get data from CSV and make useful
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

combined=np.column_stack((age,cost))
# print("combined is: ", combined)

#STEP 2: Create Covariance Matrix, find eigenvectors--
S, eig_vect, eig_val=solveCovariance(combined)

#--Make Eigenvectors plottable
eig_vec1=eig_vect[:,0]
# print("eig_vec1 is ", eig_vec1)
eig_vec2=eig_vect[:,1]
# print("eig_vec2 is ", eig_vec2)

origin=[np.mean(age),np.mean(cost)]

#-------------Part 2: LS, TLS, and RANSAC-----------
#--Calculate LS using methodology of Problem 2--
cost_ls=calcStandardLeastSquares(combined)

#--Calculate TLS--
cost_tls=calcTotalLeastSquares(age,cost)

#--Calculate RANSAC--
cost_ransac=calcRANSAC(combined)

plotter(origin, eig_vec1,eig_vec2,age, cost, cost_ls, cost_tls, cost_ransac)
