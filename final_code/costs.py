# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np
import math

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return (1/2)*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """
    For polynomial and ridge regression
    Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

def calculate_logistic_loss(y, tx, w):
    """ 
    For logistic regression :
    compute the cost by negative log likelihood.
    """
    #number of samples : 
    n = tx.shape[0]
    #accumulator for the cost :
    cost = 0
    for i in range(n):
        if (np.dot(tx[i,:],w) > 100):
            cost = cost + np.dot(tx[i,:],w)*(1-y[i])
        else:
            cost=cost+math.log(1+math.exp(np.dot(tx[i,:],w)),math.e)-y[i]*np.dot(tx[i,:],w)
    return cost

def measure_error(y, ypred):
    """
        This function measures the error between a label vector y and a predictions vector ypred
    """
    # get the number of samples
    n = len(y)
    # initialize the count
    count = 0.
    # increment it
    for i in range(n): 
        if(y[i] != ypred[i]):
            count = count + 1
            
    return count/n