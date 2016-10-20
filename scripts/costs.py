# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


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
        cost=cost+math.log(1+math.exp(np.dot(tx[i,:],w)),math.e)-y[i]*np.dot(tx[i,:],w)
    return cost