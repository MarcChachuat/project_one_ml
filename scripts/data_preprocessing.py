# -*- coding: utf-8 -*-
"""functions used to preprocess the data."""

import numpy as np
from costs import *
from proj1_helpers import *
from helpers import *

def logs_of_features(tx, feature_lists):
    return np.log(tx[:, feature_lists] + 1e-8)

def decompose_categorical_features(tx):
    """decompose categorical features
    
    Feature 11, 12 ,22 are categorical. Features 11, 12 are binary
    while 22 has four values.
    
    input:
        tx: tX without missing terms
    return:
        m : matrix of expened categorical features.
    """
    
    tmp11 = 1 * (tx[:, 11] > 0)    
    tmp12 = 1 * (tx[:, 12] > 0.5)
    
    tmp22_0 = tx[:, 22].copy()
    tmp22_0 = 1 * (tmp22_0 == 0)
    
    tmp22_1 = tx[:, 22].copy()
    tmp22_1 = 1 * (tmp22_1 == 1)
    
    tmp22_2 = tx[:, 22].copy()
    tmp22_2 = 1 * (tmp22_2 == 2)
    
    # tmp22_3 = tx[:, 22].copy()
    # tmp22_3 = 1 * (tmp22_3 == 3)

    m = np.c_[tmp11, tmp12, tmp22_0, tmp22_1, tmp22_2]
    return m

def missing_indicator(tx):
    return 1*(np.sum(tx == -999, axis=1) == 0)

def inver_terms(tx, features):
    return 1/(tx[:, features]+1e-8)

def mixed_features(tx, features):
    foo = np.zeros(tx.shape[0])
    for i, fi in enumerate(features):
        for j in range(i+1, len(features)):
             foo = np.c_[foo, tx[:, features[i]] * tx[:, features[j]]]
    return foo[:, 1:]



def transform_y(y):
    tmp = y.copy()
    tmp[tmp == -1]=0
    return tmp

def transform_y_back(y):
    tmp = y.copy()
    tmp[tmp==0]=-1  
    return tmp

def fill_na(tX, method=np.mean):
    """ fill NA term with method provided"""
    columns_with_missing_values = []
    n_total_features=tX.shape[1]
    for i in range(n_total_features):
        if -999 in tX[:, i]:
            columns_with_missing_values.append(i)

    filled = tX.copy()
    for col in columns_with_missing_values:
        tmp = filled[:, col]
        tmp[tmp == -999] = method(tmp[tmp != -999])
        filled[:, col] = tmp
    return filled

def build_polynomial_without_mixed_term(tx, degree=2):
    """ build polynomial terms.

    The constant terms will be added in the standardize function.
    """
    n = tx.shape[0]
    tmp = tx
    for i in range(2, degree+1):
        tmp = np.c_[tmp, tx**i]

    return tmp

# def standardize(x):
#     #getting the number of samples and the number of dimensions
#     n,d=np.shape(x)
    
#     for i in range(d):
#         #computation of the mean and the standard deviation of each dimension
#         m=np.mean(x[:,i])
#         s=np.std(x[:,i])
#         #standardizing this dimension samples
#         if s>0:
#             x[:,i]=(x[:,i]-m)/s
#         else:
#             x[:,i]=(x[:,i]-m)
#             #sending a warning
#             print("warning : the standard deviation of the dimension :", i, " is null")
#     return x


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
        "x" : 2D-array, each row corresponding to a sample (maybe preprocessed).
               one coordinate by feature 
               ( n : number of features), (m : number of samples)
        "degree" : int, the maximal degree of the polynomial basis
        
        output : 
        "phi" : 2D array, each row corresponding to a sample
                dimension m x (n(degree)+1)
    """
    
    #local variables
    m = x.shape[0]
    n = x.shape[1]
    
    # create the new matrix
    phi = np.ones((m, n*degree+1))
    
    # fill the matrix
    for i in range(m):
        for j in range(n):
            for t in range(degree):
                phi[i,(1+t)+degree*j]=x[i,j]**(t+1)
            
   
    # return the matrix 
    return phi


def clean_data(x): 
    """
        As some values where missing in the original data set, -999 have been put instead. 
        This function deals with this problem by replacing -999 by the average of the good samples values for this feature
        
        input : 
        "x" : a 2D array containing bad values
        ouput : 
        "y" : a 2D array with the same dimension where we replaced the -999 value
        
    """
    #First compute the mean for each column of values that are not -999
    m=x.shape[0]
    n=x.shape[1]
    num_samples=np.zeros(n)
    mean=np.zeros(n)
    for j in range(n):
        for i in range(m):
            if (x[i,j]!=-999):
                mean[j]=mean[j]+x[i,j]
                num_samples[j]=num_samples[j]+1
        if (num_samples[j]!=0): 
            mean[j]=mean[j]/num_samples[j]
    y=np.copy(x)
    
    for i in range(m):
        for j in range(n):
            if(y[i,j]==-999):
                y[i,j]=mean[j]
    return y


def clean_remove_data(x,threshold):
    """
        As some values where missing in the original data set, -999 have been put instead. 
        This function deals with this problem by replacing -999 by the average of the good samples values for this feature
        
        input : 
        "x" : a 2D array containing bad values
        "threshold" : float in [0,1], upper which we delete a column
        ouput : 
        "indexes" : an array containing the indices of columns to be deleted
    """
    bad=[]
    m=x.shape[0]
    n=x.shape[1]
    for j in range(n):
        if ((x[:,j]==-999).sum()/len(x[:,j])>threshold):
            bad.append(j)
    return bad  


def pre_process_logistic_training_labels(ytrain):
    """
        This function pre-process the training labels for logistic regression, 
        as the label used in the course are 0,1 and the labels used here are -1 and 1
    """
    # replace the -1 by 0
    y = np.copy(ytrain)
    y[y == -1] = 0
    return y


def post_process_logistic_predicted_labels(ypred):
    """
        This function pre-process the training labels for logistic regression, 
        as the label used in the course are 0,1 and the labels used here are -1 and 1
    """
    # replace the 0 by -1
    y = np.copy(ypred)
    y[y == 0] = -1
    return y