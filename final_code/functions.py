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

def sigmoid(x, clip_range=20):
    # to avoid overflow in exponential, clip input x into a reasonable large range
    cliped_x = np.clip(x, -clip_range, clip_range)
    return 1/(1+np.exp(-cliped_x))

def regularizor_lasso(w):
    # return loss and gradient
    return np.sum(np.abs(w)), np.sign(w)

def regularizor_ridge(w):
    # return loss and gradient
    return np.sum(w**2), 2 * w

def logistic_regression_GD(y, tx, gamma, max_iters):
    return reg_logistic_regression_GD_with_init(y, tx, gamma, max_iters)

def reg_logistic_regression_GD(y, tx, gamma, max_iters, lambda_, regularizor=regularizor_ridge):
    assert(lambda_>= 0)
    return reg_logistic_regression_GD_with_init(y, tx, gamma, max_iters, lambda_=lambda_,regularizor=regularizor)

def logistic_regression_SGD(y, tx, gamma, max_iters, w0=None):
    return reg_logistic_regression_GD_with_init(y, tx, gamma, max_itersm, w0=w0)

def reg_logistic_regression_SGD(y, tx, gamma, max_iters, lambda_, regularizor=regularizor_ridge, w0=None):
    assert(lambda_>= 0)
    return reg_logistic_regression_GD_with_init(y, tx, gamma, max_iters, lambda_=lambda_, w0=w0, regularizor=regularizor)

def reg_logistic_regression_GD_with_init(y, tx, gamma, max_iters, w0=None, lambda_= 0, regularizor=regularizor_ridge):
    """ Logistic regression using Gradient descent.
    
    As loss fuction of logistic regression is a L-lipschitz function, it is better to use 
        $$gamma=1/L$$
    where $L=||tx.T @ tx||/4$ is the lipschitz constant.
    
    inputs:
        "y"        : 1D-array, containing the training values associated to each sample.
        "tx"       : nD-array, each row contains the data associated to a sample.
                       each column contains all the sample values for a feature.
        "gamma"    : float, gradient descent learning rate
        "max_iters": maximum number of iterations
        
    returns: 
        "cost"     : float,value of the cost function associated to the optimal analytical weight vector
                     obtained from logistic regression.
        "w"        : 1-D array, optimal weight code associated to the logistic regression cost function,
                     with respect to tx and y.
    """    
    def logistic_loss(y, tx, w):
        """compute the cost by negative log likelihood."""
        r,_ = regularizor(w)
        return np.sum(-np.log(1 - sigmoid(tx @ w))) - y.T @ tx @ w + lambda_ * r
    
    def logistic_gradient(y, tx, w):
        """compute the gradient of loss."""
        _, g = regularizor(w)
        return tx.T @ (sigmoid(tx @ w) - y) + lambda_ * g

    # Initialize weights
    if w0 is not None:
        w = w0
    else:
        w = np.random.randn(tx.shape[1]) * 0.1

    costs = []

    # Training
    for i in range(max_iters):
        w    = w - gamma * logistic_gradient(y, tx, w)
        cost = logistic_loss(y, tx, w)

        if i % 1000 == 0:
            print ("Losgistic Regression({bi: >8}/{ti}): loss={l: 10.15}".format(bi=i, ti=max_iters, l=cost))
        costs.append(cost)
    return w, costs

def logistic_AGDR(y, tx, gamma, max_iters, w0=None, lambda_= 0, regularizor=regularizor_ridge):
    """ Logistic regression using accelerated Gradient descent with restart.
    
    As loss fuction of logistic regression is a L-lipschitz function, it is better to use 
        $$gamma=1/L$$
    where $L=||tx.T @ tx||/4$ is the lipschitz constant.
    
    inputs:
        "y"        : 1D-array, containing the training values associated to each sample.
        "tx"       : nD-array, each row contains the data associated to a sample.
                       each column contains all the sample values for a feature.
        "gamma"    : float, gradient descent learning rate
        "max_iters": maximum number of iterations
        "w0"       : initial place
        "regularizor": types of regularizor function which returns the value regularizor term and its gradient. 
        
    returns: 
        "w"        : 1-D array, optimal weight code associated to the logistic regression cost function,
                     with respect to tx and y.
    """    
    def grad(w):
        """compute the gradient of loss."""
        _, g = regularizor(w)
        return tx.T @ (sigmoid(tx @ w) - y) + lambda_ * g
    
    def loss(w):
        """compute the cost by negative log likelihood."""
        r,_ = regularizor(w)
        return np.sum(-np.log(1 - sigmoid(tx @ w))) - y.T @ tx @ w + lambda_ * r
    
    # Initialization
    if w0 is None:
        w = np.random.randn(tx.shape[1])
    else:
        w = w0

    # initialization
    z = w
    t = 1
    
    last_cost = np.inf
    for i in range(max_iters):
        w_next = z - gamma * grad(z)

        # Restart if the new loss is larger
        if loss(w) <= loss(w_next):
            z = w
            t = 1
            w_next = z - gamma*grad(z)
        
        t_next = (1+np.sqrt(1+4*t**2))/2
        z_next = w_next + (t-1)/(t+1)*(w_next - w)
        
        # update
        z, w, t = z_next, w_next, t_next

        cost = loss(w_next)
        if i % 100 == 0:
            print ("Losgistic Regression({bi: >8}/{ti}): loss={l: 10.15}".format(bi=i, ti=max_iters, l=cost))

            if last_cost - cost < 1e-5 * abs(last_cost):
                print ("Totoal number of iterations = ", i)
                print ("Loss                        = ", cost)
                break

            last_cost = cost
        
        # The learning rate becomes smaller as the number iterations grows.    
        gamma = 1/(1/gamma + 1)

    return w, cost