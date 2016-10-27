# -*- coding: utf-8 -*-
"""functions used to train the model."""

import numpy as np
from costs import *
from proj1_helpers import *
from helpers import *

def compute_gradient(y, tx, w):
    """ compute the gradient associated to the MSE cost function"""
    e= y - np.dot(tx,w)
    num_samples=len(y)
    grad=-(1/num_samples)*np.dot(np.transpose(tx),e)
    return grad 


def least_squares_GD(y, tx, initial_w, max_iters, gamma): 
    """Gradient descent algorithm using the MSE cost function
    
    inputs : 
    "y" : 1D-array, containing the training values
    "tx" : 2D-array, each row contains the data associated to a sample.
           each column contains all the sample values for a feature
    "initial_w" : 1D-array, initial weight vector from which begins the first iteration
    "gamma" : float, step_size
    "max_iters" : int, number of iterations to be performed
    
    outputs: 
    "losses" : 1-D array of the successive values of the MSE cost function
    "ws" : 1-D array of the successive values of the weight vector
    
    """
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient and loss
        grad=compute_gradient(y, tx, w)
        loss=compute_loss(y, tx, w)
        print("loss : ", loss)
        # Update the weights
        w = w - gamma*grad
        # Store the new weight and the loss associated to the previous weight
        ws.append(np.copy(w))
        #print(np.copy(w))
        losses.append(loss)
        # Print the new weight and the previous loss.
        print("Gradient Descent iteration : ", str(n_iter), " done \n")
    # Return the arrays containing the losses and weight vectors
    w_final=ws[len(ws)-1]
    #return losses, ws
    return w_final



def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    minibatch_e=y-np.dot(tx,w)
    stoch_grad= -np.dot(np.transpose(tx),minibatch_e)
    return stoch_grad


def least_squares_SGD(
        y, tx, initial_w, batch_size=1, max_iters=50, gamma=0.000001):
    """ Stochastic gradient descent algorithm using the MSE COST function
    
    inputs : 
    "y" : 1D-array, containing the training values
    "tx" : 2D-array, each row contains the data associated to a sample.
           each column contains all the sample values for a feature.
    "initial_w" : 1D-array, initial weight vector from which begins the first iteration
    "gamma" : float, step_size
    "batch_size": size of the batch at each iteration
    "max_iters" : int, maximal number of iterations to be performed
    
    outputs:  
    "losses" : 1-D array, contains the successive values of the MSE cost function
    "ws" : 1-D array, contains the successive values of the weight vector
    
    """
    # max_epochs not used ? 
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w=initial_w
    # counting the iterations 
    n_iter=0
    # for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        # incrementing the count of iterations
        n_iter=n_iter+1
        # Compute the stochastic gradient and the loss
        stoch_grad= -(1.0/batch_size)*compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss= compute_loss(y, tx, w)
        # Update the weight
        w=w-gamma*stoch_grad
        # Store the new weight and the previous loss
        losses.append(loss)
        ws.append(w)
        print(" Stochastic Gradient Descent iteration : ", str(n_iter), " done \n")
        print( " value of the current loss : ", str(loss)) 
    w_final=ws[len(ws)-1]
    #return losses, ws
    return w_final

def least_squares(y, tx):
    """calculate the least squares analytic solution.
    
    inputs:
    "y" : 1D-array, containing the training values associated to each sample.
    "tx" : 2D-array, each row contains the data associated to a sample.
           each column contains all the sample values for a feature.
    
    outputs: 
    "mse" : float, value of the MSE cost function applied to y, tx and the optimal weight w
    "w" : 1-D array, optimal weight for the mse cost function with respect to y and tx.
    
    """
    # compute the grammarian
    gram=np.dot(np.transpose(tx),tx)
    print ("shape du grammarian: ", gram.shape)
    # compute the optimal weight
    w=np.dot(np.dot(np.linalg.inv(gram),np.transpose(tx)),y)
    print ("shape du optimal weight : ", w.shape)
    # compute the associated loss
    mse=compute_loss(y, tx, w)
    #return mse,w
    return w

def ridge_regression(y, tx, lamb):
    """ implement ridge regression.
    
    inputs:
    "y" : 1D-array, containing the training values associated to each sample.
    "tx" : 2D-array, each row contains the data associated to a sample.
           each column contains all the sample values for a feature.
    "lamb" : float, coefficient associated to the penalization of the weight vector
    
    ouputs : 
    "rrcost" : float,  value of the cost function associated to the optimal analytical weight vector
    for the ridge regression
    "w_ridge": 1-D array, optimal weight code associated to the ridge regression cost function,
    with respect to tx and y. 
    
    """
    #define an auxiliary variable
    lamb_aux=2*len(y)*lamb
    # analytical solution
    gram= np.dot(np.transpose(tx),tx)
    w_ridge= np.dot(np.dot(np.linalg.inv(gram+lamb_aux*np.identity(gram.shape[0])),np.transpose(tx)),y)
    # calculate the error (cost function)
    rr_cost= compute_loss(y,tx,w_ridge)+lamb*(np.linalg.norm(w_ridge)**2)
    # return the cost and the optimal weight
    #return rr_cost, w_ridge
    return w_ridge



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
    
    #applying the right regression
    if (n_regression==1):
        weights= least_squares_GD(ytrain, xtrain, initial_w, max_iters, gamma)
    if (n_regression==2):
        weights= least_squares_SGD(ytrain, xtrain, initial_w, batch_size, max_iters, gamma)
    if (n_regression==3): 
        weights= least_squares(ytrain, xtrain)
    if (n_regression==4):
        weights= ridge_regression(ytrain, xtrain, lambd)
        
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
    # randomly initialize the weight vector
    def logistic_loss(y, tx, w):
        """compute the cost by negative log likelihood."""
        r,_ = regularizor(w)
        return np.sum(-np.log(1 - sigmoid(tx @ w)) - y.T @ tx @ w) + lambda_ * r
    
    def logistic_gradient(y, tx, w):
        """compute the gradient of loss."""
        _, g = regularizor(w)
        return tx.T @ (sigmoid(tx @ w) - y) + lambda_ * g

    w = np.random.randn(tx.shape[1]) * 0.1
    if w0 is not None:
        w = w0
    costs = []
    
    for i in range(max_iters):
        w = w - gamma * logistic_gradient(y, tx, w)
        cost = logistic_loss(y, tx, w)
        if i % 1000 == 0:
            print ("Losgistic Regression({bi}/{ti}): loss={l}".format(bi=i, ti=max_iters, l=cost))
        costs.append(cost)
    return w, costs