# implementations.py
# This file contains the 6 basic methods of the course, and the subroutines used by these methods

import numpy as np
from costs import *
from proj1_helpers import *
from helpers import *
from implementations import *

############## 1°) Least squares Gradient Descent ################################################################################


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
    losses.append(compute_loss(y, tx, w))
    
    for n_iter in range(max_iters):
        # Compute gradient 
        grad=compute_gradient(y, tx, w)
        
        # Update the weights
        w = w - gamma*grad
        
        # compute the new loss
        loss = compute_loss(y, tx, w)
        #print("loss : ", loss)
        
        # Store the new weight and the new loss associated to the previous weight
        ws.append(np.copy(w))
        #print(np.copy(w))
        losses.append(loss)
        
        # Print the new weight and the previous loss.
        #print("Gradient Descent iteration : ", str(n_iter), " done \n")
    # Return the arrays containing the losses and weight vectors
    w_final=ws[len(ws)-1]
    
    #return losses, ws
    return w_final, losses[-1]


############## 2°) Least squares Stochastic Gradient Descent #####################################################################


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    minibatch_e=y-np.dot(tx,w)
    stoch_grad= -np.dot(np.transpose(tx),minibatch_e)
    return stoch_grad


def least_squares_SGD(
        y, tx, initial_w, max_iters, gamma):
    """ Stochastic gradient descent algorithm using the MSE COST function
    
    inputs : 
    "y" : 1D-array, containing the training values
    "tx" : 2D-array, each row contains the data associated to a sample.
           each column contains all the sample values for a feature.
    "initial_w" : 1D-array, initial weight vector from which begins the first iteration
    "max_iters" : int, maximal number of iterations to be performed
    "gamma" : float, step_size
    
    outputs:  
    "losses" : 1-D array, contains the successive values of the MSE cost function
    "ws" : 1-D array, contains the successive values of the weight vector
    
    """
     
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    
    # Initialize the weight vector
    w = initial_w
    losses.append(compute_loss(y, tx, w))
    
    # Initialize the iteration counter 
    n_iter = 0
    
    # Batch size
    batch_size = 1
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, max_iters):
        
        # incrementing the count of iterations
        n_iter=n_iter+1
        
        # Compute the stochastic gradient 
        stoch_grad= -(1.0/batch_size)*compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        
        
        # Update the weight
        w = w - gamma*stoch_grad
        
        # compute the new loss
        loss = compute_loss(y, tx, w)
        
        # Store the new weight and the previous loss
        losses.append(loss)
        ws.append(w)
        print(" Stochastic Gradient Descent iteration : ", str(n_iter), " done \n")
        print( " value of the current loss : ", str(loss)) 
    
    # final weight
    w_final = ws[len(ws)-1]
    
    # return the final weight and the final loss
    return w_final, losses[-1]


############## 3°) Analytical least squares #####################################################################################


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
    gram = np.dot(np.transpose(tx),tx)

    # compute the optimal weight
    #w=np.dot(np.dot(np.linalg.inv(gram),np.transpose(tx)),y)
    w = np.linalg.solve(gram,np.dot(np.transpose(tx),y))
    
    # compute the loss
    loss = compute_loss(y, tx, w)
    
    # return the optimal weight vector and the loss
    return w, loss


############## 4°) Ridge regression ##############################################################################################


def ridge_regression(y, tx, lambda_):
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
    lamb_aux=2*len(y)*lambda_

    # analytical solution
    gram = np.dot(np.transpose(tx),tx)

    A = gram + lamb_aux*np.identity(gram.shape[0])
    b = np.dot(np.transpose(tx),y)
    #w_ridge= np.dot(np.dot(np.linalg.inv(gram+lamb_aux*np.identity(gram.shape[0])),np.transpose(tx)),y)
    w_ridge = np.linalg.solve(A,b)
    
    # compute the error (cost function)
    rr_cost= compute_loss(y,tx,w_ridge)+lambda_*(np.linalg.norm(w_ridge)**2)
    
    # return the optimal weight and the loss
    return w_ridge, rr_cost


############## 5°) Logistic regression ###########################################################################################


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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
        logistic regression function using gradient descent
        signature corresponding to the requirements of the teachers
        just call an other logistic regression function
    """
    
    w, costs = reg_logistic_regression_GD_with_init(y, tx, gamma, max_iters, w0=initial_w)
    return w, costs[-1]


############## 6°) Regularized Ridge regression


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
        logistic regression function using gradient descent
        signature corresponding to the requirements of the teachers
        just call an other logistic regression function
    """
    
    w, costs = reg_logistic_regression_GD_with_init(y, tx, gamma, max_iters, w0=initial_w, lambda_= lambda_)
    return w, costs[-1]