#run.py

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from functions import *

from functions import *
from implementations import *
from helpers import *
from proj1_helpers import *
from costs import *
from data_preprocessing import *

#################################################################################################################################
############################ Part I : Load the training data into feature matrix, class labels, and event ids: ##################

from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv' 
# download train data 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print("loading of the data : done")

#################################################################################################################################
############################ Part II : Linear Regression ########################################################################

print("Linear regression models ...")

########## A) Data Preprocessing 

##### Select some data
num_samples=5000
seed=3
y_lin, tX_lin = select_random(y, tX, num_samples, seed)
print ("random selection of samples : done")
print( "number of samples : ", y.shape[0])

##### Split the data
xtrain1,xvalid1,ytrain,yvalid = split_data(tX_lin, y_lin, 0.7)
print( "splitting of the data : done ")
print("number of training samples :", xtrain1.shape[0])

##### Standardize the data
xtrain1bis = standardize(xtrain1)
xvalid1bis = standardize(xvalid1)

##### Reduce the dimension : Perform a PCA on the training data

# percentage of information we keep during the PCA
ratio_pca=0.8

# find the projector
U, k = my_pca(xtrain1bis, ratio_pca)
print("PCA done ")

# projecting the data 
xtrain2=np.dot(xtrain1bis,U)
xvalid2=np.dot(xvalid1bis,U)


########## B) Comparison of the polynomial ridge regression

# range for the parameters
lambdas = np.logspace(-3, 3, 500) 
degree=7

# initial best parameters
degree_ref=0
w_ref = 0
lambd_ref=0
score_ref = float("inf")

# loop over the degrees
for t in range(1,degree):
    
    # build polynomial basis of the current degree
    xtrain_rr = build_polynomial_without_mixed_term(xtrain2, t)
    xvalid_rr = build_polynomial_without_mixed_term(xvalid2, t)
    
    # loop over the lambdas
    for lambd_ in lambdas:
        
        # train the model for the ridge regression given the current lambda
        weights = train_data(xtrain_rr, ytrain, n_regression=4,lambd=lambd_)

        # compute its error on the validation set
        yvalid_pred = predict_labels(weights, xvalid_rr)
        score = 0
        for i in range(yvalid.shape[0]):
            if (yvalid_pred[i] != yvalid[i]):
                score = score + 1
                
        #if its score is better than the best current one we keep the (weights, lambda couple)
        if score<score_ref:
            # updating the best parameters and errors
            w_ref=weights
            score_ref = score
            lambd_ref = lambd_
            degree_ref = t
            
        # print the result for this iteration    
        print("degree : ", str(t),  ", study of the parameter :", str(lambd_), " done. Error associated : ", str(score_ref/yvalid.shape[0]))

        
########## C) Results of the comparison        

# Best parameters
degree_best_rr = degree_ref
weights_rr = w_ref
err_rr = score_ref/yvalid.shape[0]

# Plot the results 
print("Finding of the best ridge regression parameters : done")
print("Best lambda parameter : " , str(lambd_ref))
print("Best degree : ", str(degree_best_rr))
print(" Associated error (in percent of the validation set) : " , str(err_rr))
print("Associated weight vector : " , str(weights_rr))

################################################################################################################################
############################ Part III : Logistic regression  ###################################################################

