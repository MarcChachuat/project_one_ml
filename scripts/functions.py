# -*- coding: utf-8 -*-
"""functions used to train the model."""

import numpy as np
from costs.py import *

def compute_gradient(y, tx, w):
    """ compute the gradient associated to the MSE cost function"""
    e= y - np.dot(tx,w)
    num_samples=len(y)
    grad=-(1./num_samples)*np.dot(np.transpose(tx),e)
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
        # Update the weights
        w=w-gamma*grad
        # Store the new weight and the loss associated to the previous weight
        ws.append(np.copy(w))
        losses.append(loss)
        # Print the new weight and the previous loss.
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    # Return the arrays containing the losses and weight vectors
    return losses, ws


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    minibatch_e=y-np.dot(tx,w)
    stoch_grad= -np.dot(np.transpose(tx),minibatch_e)
    return stoch_grad


def least_squares_SGD(
        y, tx, initial_w, batch_size, max_iters, gamma):
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
    # for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,max_iters):
        #Compute the stochastic gradient and the loss
        stoch_grad= -(1.0/batch_size)*compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        loss= compute_loss(y, tx, w)
        # Update the weight
        w=w-gamma*stoch_grad
        # Store the new weight and the previous loss
        losses.append(loss)
        ws.append(w)
    return losses, ws


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
    # compute the optimal weight
    w=np.dot(np.dot(np.linalg.inv(gram),np.transpose(tx)),y)
    # compute the associated loss
    mse=compute_loss(y, tx, w)
    return mse,w


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
    # compute the analytical solution
    gram= np.dot(np.transpose(tx),tx)
    w_ridge= np.dot(np.dot(np.linalg.inv(gram+lamb*np.identity(gram.shape[0])),np.transpose(tx)),y)
    # calculate the error (cost function)
    rr_cost= compute_loss(y,tx,w_ridge)+lamb*(np.linalg.norm(w_ridge)**2)
    # return the cost and the optimal weight
    return rr_cost, w_ridge
