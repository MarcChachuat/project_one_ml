# -*- coding: utf-8 -*-
"""functions used to preprocess the data."""

import numpy as np
from costs import *
from proj1_helpers import *
from helpers import *


def select_random(y, tX, num_samples, seed=1):
    """ 
    select randomly num_samples rows of the data and label sets 
    
    inputs:
    "y": 1D-array, sample labels 
    "tX" : 2D-array, each row containing the features of the corresponding sample
    
    outputs:
    "y2 : 1D-array, the labels of the selected samples
    "tX2": 2D-array, each row containing the features of a selected sample
    """
    # set seed
    np.random.seed(seed)
    
    #change the order of the indices
    indices = np.arange(y.shape[0])
    np.random.shuffle(indices)
    y2 = y[indices]
    tX2 = tX[indices,:]
    #select the first num_samples rows of the shuffled matrix
    if (num_samples < y2.shape[0]):
        y2 = y2[:num_samples]
        tX2 = tX2[:num_samples,:]
    return y2,tX2


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio.
    Obtain two sets : the training set and the test set ( or validation set)
    
    inputs : 
    "x" : 2D-array, data features
    "y" : 1D-array, data labels
    "ratio" : float, equal to the number of training data over the number of data 
    
    outputs : 
    "xtrain" : 2D-array, training data features
    "ytrain" : 1D- array, training data labels
    "xtest" : 2D-array, validation data features
    "ytest" : 1D-array, validation data labels
    """
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # split the data based on the given ratio: TODO
    indices=np.arange(x.shape[0])
    np.random.shuffle(indices)
    
    #upperbound for the ratio : 
    up=int(ratio * x.shape[0])
    #creation of the data sets
    xtrain=x[indices[0:up]]
    xtest=x[indices[up:]]
    ytrain=y[indices[0:up]]
    ytest=y[indices[up:]]
    #returning the result
    return xtrain,xtest,ytrain,ytest

def my_pca(x, ratio): 
    """
        This function realizes a PCA (features selection) on the matrix x.
        
        inputs : 
        "x" : 2D (m x n) matrix (m samples, n features) containing the data
        "ratio " : float in [0,1], indicates which percentage of the data the selected features should represent
        
        ouputs : 
        "U" : 2D (n x k) matrix, projector,
        "k" : int, new number of features
    """
    
    #local variables
    M = x
    [m, n] = np.shape(M)
    #substract the mean of each column
    for j in range(n):
        u = np.mean(x[:,j])
        for i in range(m):
            M[i,j] = M[i,j]-u
    
    #compute the covariance matrix 
    W = np.dot(np.transpose(M),M)
    
    #diagonalize the matrix
    eigval, eigvectors = np.linalg.eig(W)
    
    #sorte the eigenvalues in decreasing order
    indices = np.argsort(eigval)
    indices = indices[::-1]
    eigval = eigval[indices]
    eigvectors = eigvectors[:,indices]
    print( "eigenvalues sorted by decreasing order : ", eigval)
    
    #find k such that the first k first values are containing ratio percent of the information
    k=0
    ref_sum=0
    total_sum=np.sum(eigval)
    while(ref_sum < ratio*total_sum ):
        k = k+1
        ref_sum = ref_sum+eigval[k-1]
    
    # debugging
    print(" debugging : check the values :")
    print("k : ", k)
    print("partial sum of eigenvalues : ", ref_sum)
    print("total sum of eigenvalues : ", total_sum)
    print("ratio : ", str(ref_sum/total_sum))
          
    #keep the corresponding eigenvectors
    U=eigvectors[:,:k]
    return U,k
          
def build_poly(x,degree):
    """
        This function treats the features in x using polynomial basis. 
        
        input : 
        ""
    """
    
   
    
    