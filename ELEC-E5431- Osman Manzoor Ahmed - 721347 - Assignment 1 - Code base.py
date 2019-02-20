
# coding: utf-8

# # **ELEC-E5431 – Large-Scale Data Analysis (LSD Analysis)**
# **Home Work 1**<br>
# **Osman Manzoor Ahmed**<br>
# **721347**<br>
# 

# 
# In every algorithm the output is as following:
# 1. X Matrix computed by the algorithm
# 2. X Optimal Matrix
# 3. Difference of the above mentioned matrices
# 4. Convergence rate plot
# 5. Plot of the X optimal and X created by the algorithm
# 
# Kindly scroll down to the ends of each algorithm output in order to see the plots.

# In[13]:


# All Libraries

import pandas as pd 
from matplotlib import pyplot as plt 
from IPython.display import display, HTML
import numpy as np  
from scipy import random, linalg
from sklearn.datasets import make_spd_matrix
import math
from random import randint


# In[33]:


# ********************************* All Functions*******************************

# Gradient  Ax-b
def gradient(A,b,x):
    #gradient = np.subtract(np.dot(np.transpose(A),x),b)
    gradient = np.subtract(np.dot(A,x),b)
    #gradient = np.subtract(b,np.dot(np.transpose(A),x))
    return gradient

# Negative Gradient  b-Ax
def neg_gradient(A,b,x):
    #gradient = np.subtract(np.dot(np.transpose(A),x),b)
    gradient = np.subtract(b,np.dot(np.transpose(A),x))
    return gradient

#Error Function log || x_grad - x_opt ||
def logerror(x_grad, x_opt):
    diff = x_grad - x_opt
    norm = np.linalg.norm(diff)
    log = np.log(norm)
    return log

# Gradient Descent Method
def gradient_descent(A, b, a, k,x_opt):
    #x = np.zeros((100,1))
    n = len(b)
    x = np.zeros((n,1))
    errors = []

    #for i in range(k):
    while True:
        # Calculate gradient
        grad = gradient(A,b,x)
        # Gradient descent formula
        x = x - a * grad
        #print("In")
        if(np.linalg.norm(grad) <= pow(10,-5)):
           break
        #empirical_errors.append(empirical_risk(X,Y,w))
        errors.append(logerror(x,x_opt))
    return errors,x

# Conjugate Gradient Descent Method
def conjugate_gradient_descent(A, b, a, k,x_opt):
    #x = np.zeros((100,1))
    n = len(b)
    x = np.zeros((n,1))
    p = b
    p_prev = np.zeros((n,1))
    errors = []
    checker = 1
    gamma = 0
    grad = np.ones((n,1))
    i = 0
    r = b
    r_new = b
    #for i in range(k):
    while True:
        # Calculate gradient
        i = i+1
        if(i == 1):
            p = r
            p_prev = p
        else:
            beta = np.divide(np.power(np.linalg.norm(r_new),2),np.power(np.linalg.norm(r),2))
            p = np.add(r_new,np.dot(beta,p_prev))
            p_prev = p
 
        # conjugate Gradient formula
        x = x + a * p
        r = r_new
        r_new = neg_gradient(A,b,x)
        if(np.linalg.norm(r_new) <= pow(10,-5)):
           break
        # Calculate the gamma now
        #print("In")
        errors.append(logerror(x,x_opt))
    return errors,x

# Nesterov’s Algorithm
def nesterov_algorithm(A, b, a, L, k,x_opt):
    #x = np.zeros((100,1))
    n = len(b)
    x = np.zeros((n,1))
    errors = []
    y = x
    alpha = a
    
    #for i in range(k):
    while True:
        # Calculate gradient
        x_prev = x
        grad = gradient(A,b,y)
        x = (np.subtract(y,(1/L)*grad))
        a_prev = alpha
        alpha = (1 + math.sqrt(1 + 4 * a_prev * a_prev))/2
        
        beta = (a_prev * (1-a_prev)) / (a_prev * a_prev + alpha)
             
        y = x + (beta * (np.subtract(x,x_prev)))
        
        if(np.linalg.norm(grad) <= pow(10,-5)):
           break
        # Calculate the gamma now
        #print("In")
        errors.append(logerror(x,x_opt))
    return errors,x

# Stochastic Coordinate Descent Method
def stochastic_coordinate_descent(A, b, a, k,x_opt):
    #x = np.zeros((100,1))
    n = len(b)
    x = np.zeros((n,1))
    errors = []

    for i in range(k):
    #while True:
        #for j in range(len(x)):
        random = randint(0, n-1)
        # Calculate gradient
            
        grad = gradient(A[random,random].reshape(1,1),b[random].reshape(1,1),x[random].reshape(1,1))
        #print(grad.shape)
        # Gradient descent formula
        x[random] = x[random] - a * grad
        #print("In")
        if(np.linalg.norm(grad) <= pow(10,-5)):
           break
        #empirical_errors.append(empirical_risk(X,Y,w))
        errors.append(logerror(x,x_opt))
    return errors



# In[15]:


# Creation of Matrices A, b and X_opt(Optimal x)

# Generate a random symmetric, positive-definite matrix. Size 100*100 
matrixSize = 100
A = make_spd_matrix(matrixSize, random_state=None)

# Check if the newly created matrix A is positive- definite. Check if all its Eignevalues are positive
if(np.all(np.linalg.eigvals(A) > 0)):
    print("Success, you have a positive definite matrix")
    
# Generate matrix b size 100*1 that should be in range of Matrix A
x = np.ptp(A,axis = 0)
b = np.reshape(x, (100,1))

# Generate X_Opt as X_opt = A(inverse)b
A_inverse = np.linalg.inv(A)
X_opt = np.dot(A_inverse,b)

#Calculate value of alpha as 1/trace(A)
alpha = 1/A.trace()


# In[22]:



# **************************** Gradient Descent *******************

loggError, x_grad_hat = gradient_descent(A,b,alpha,1000,X_opt)
print("******************************************************")
print("X Matrix computed by the gradient descent")
print(x_grad_hat)
print("******************************************************")
print("X Optimal ")
print(X_opt)
print("******************************************************")
diff_gradient_descent = np.subtract(x_grad_hat,X_opt)
print("Gradient Descent, Difference between x and x_optimal")
print(diff_gradient_descent)

plt.plot(loggError)
plt.xlabel('Number of Iterations')
plt.ylabel('Log Error')
plt.title('Plot of error according to Iterations(Gradient Descent)')
plt.show()

fig, axes = plt.subplots(1, 2,figsize=(12, 4))
axes[0].plot(X_opt)
axes[0].set_title("X optimal")
axes[1].plot(x_grad_hat,color="red")
axes[1].set_title("X Created by Gradient Descent")


# In[21]:


#***************************** Conjugate Gradient Descent ************

logeError,x_conj_grad_hat = conjugate_gradient_descent(A,b,alpha,10000,X_opt)
print("******************************************************")
print("X Matrix computed by the conjugate gradient descent")
print(x_conj_grad_hat)
print("******************************************************")
print("X Optimal ")
print(X_opt)
print("******************************************************")
diff_conj_gradient_descent = np.subtract(x_conj_grad_hat,X_opt)
print("Conjugate Gradient Descent, Difference between x and x_optimal")
print(diff_conj_gradient_descent)

plt.plot(logeError)
plt.xlabel('Number of Iterations')
plt.ylabel('Log Error')
plt.title('Plot of error according to Iterations(Conjugate Gradient Descent)')
plt.show()


fig, axes = plt.subplots(1, 2,figsize=(12, 4))
axes[0].plot(X_opt)
axes[0].set_title("X optimal")
axes[1].plot(x_conj_grad_hat,color="red")
axes[1].set_title("X Created by Conjugate Gradient Descent")


# In[28]:


#***************************** Nesterov’s Algorithm ************

logError,x_nest_algo = nesterov_algorithm(A,b,0,A.trace(),10000,X_opt)
print("******************************************************")
print("X Matrix computed by the Nesterov Algorithm")
print(x_nest_algo)
print("******************************************************")
print("X Optimal ")
print(X_opt)
print("******************************************************")
diff_nest_algo = np.subtract(x_nest_algo,X_opt)
print("Nesterov Algorithm, Difference between x and x_optimal")
print(diff_nest_algo)

plt.plot(logError)
plt.xlabel('Number of Iterations')
plt.ylabel('Log Error')
plt.title('Plot of error according to Iterations(Nesterov Method)')
plt.show()

fig, axes = plt.subplots(1, 2,figsize=(12, 4))
axes[0].plot(X_opt)
axes[0].set_title("X optimal")
axes[1].plot(x_nest_algo,color="red")
axes[1].set_title("X Created by Nesterov Algorithm")


# In[34]:


#***************************** Stochastic Coordinate Descent Algorithm ************

logcoorError = stochastic_coordinate_descent(A,b,alpha,50000,X_opt)
#print(len(logError))
#print("Gradient X")
#print(x_grad)
#print(x_grad.shape)

plt.plot(logcoorError)
plt.xlabel('Number of Iterations')
plt.ylabel('Log Error')
plt.title('Plot of error according to Iterations(Stochastic Coordinate Descent Method)')
plt.show()

