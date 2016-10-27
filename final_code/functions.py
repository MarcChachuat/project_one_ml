# -*- coding: utf-8 -*-
"""functions used to train the model."""

import numpy as np
from costs import *
from proj1_helpers import *
from helpers import *
from implementations import *


def train_data(xtrain,ytrain,n_regression=1,lambd=0.1,gamma=0.000001,max_iters=50,batch_size=1):
    """
        train the model selected by "n_regression", with the lambdas parameters if needed,
        over the training set xtrain, ytrain
        This function has been created so that I will be able to train a chosen set of parametered 
        models and compare them. 
        
        inputs:
        "xtrain" : 2D-array, training data features
        "ytrain" : 1D array, training data labels
        "n_regression" : int, indicates which regression function does the training
                "1" => least_squares_GD
                "2" => least_squares_SGD
                "3" => least_squares
                "4" => ridge_regression
        "lambd" : (optionnal) float, constraining paramater for the ridge regression
        "gamma" : (optionnal) float, step_size of the gradient descent
        "max_iters" : (optionnal) int, numbers of iterations in the gradient descent
        "batch_size" : (optionnal) int, size of the batch for the stochastic gradient descent method
        
        ouputs: 
        "weights" : 1D-array, the training weight for the linear model
    """
    #setting the initial vector
    initial_w=np.zeros(xtrain.shape[1])
    #print( "shape de w:", initial_w.shape)
    print("lambda for ridge regression : ", lambd)
    #applying the right regression
    if (n_regression==1):
        weights, loss= least_squares_GD(ytrain, xtrain, initial_w, max_iters, gamma)
    if (n_regression==2):
        weights, loss= least_squares_SGD(ytrain, xtrain, initial_w, batch_size, max_iters, gamma)
    if (n_regression==3): 
        weights, loss= least_squares(ytrain, xtrain)
    if (n_regression==4):
        weights, loss= ridge_regression(ytrain, xtrain, lambd)
        
    #returning the trained weights vector
    return weights
