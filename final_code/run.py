#run.py

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from helpers import *
from proj1_helpers import *
from costs import *
from data_preprocessing import *
from implementations import *

#################################################################################################################################
############################ Part I : Load the training data into feature matrix, class labels, and event ids: ##################

from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv' 
# download train data 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
print("loading of the data : done")

##### Select some data
num_samples=10000
seed=3
y, tX= select_random(y, tX, num_samples, seed)
print ("random selection of samples : done")
print( "number of samples : ", y.shape[0])

##### Split the data
xtrain1,xvalid1,ytrain,yvalid = split_data(tX, y, 0.7)
print( "splitting of the data : done ")
print("number of training samples :", xtrain1.shape[0])


#################################################################################################################################
############################ Part II : Linear Regression ########################################################################

print("Linear regression models ...")

########## A) Data Preprocessing 

# Replace the missing values by the mean over other samples
xtrain1bis = fill_na(xtrain1, method=np.mean)
xvalid1bis = fill_na(xvalid1, method=np.mean)

##### Standardize the data
xtrain1ter = my_standardize(xtrain1bis)
xvalid1ter = my_standardize(xvalid1bis)

##### Reduce the dimension : Perform a PCA on the training data

# percentage of information we keep during the PCA
ratio_pca=0.9

# find the projector
U, k = my_pca(xtrain1ter, ratio_pca)
print("PCA done ")

# projecting the data 
xtrain2=np.dot(xtrain1ter,U)
xvalid2=np.dot(xvalid1ter,U)


########## B) Comparison of the polynomial ridge regression

# range for the parameters
lambdas = np.logspace(-3, 6, 500) 
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
            w_ref = weights
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


    # TO DO 
    
    # NOTE FOR HE  : I have already select random samples among all the tX,y and created the training and validation set
    # see the Part One 
    # As a recall : 
    # xtrain1 => training part of the reduced tX
    # xvalid1 => validation part of the reduced tX
    # ytrain1 => training part of the reduced y
    # yvalid1 => validation part of the reduced y
    # WARNING : these data aren't cleaned neither standardized
    
    # I will need you to call your best percentage of error on the validation set : "err_log"
err_log = 1

################################################################################################################################
############################ Part IV : Selection of the best parametered model ##################################################


# comparison of the two percentage of error 
if (err_rr < err_log):
    best_model = "polynomial ridge"
else:
    best_model = "logistic"

################################################################################################################################
############################ Part V : Prediction using the best parametered model ###############################################

###### A) Get the data test path
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

##### B) Get the output path

# output path 
OUTPUT_PATH = '../results/result_final2.csv' 

##### C) Make the submission

if (best_model == "polynomial ridge"):
    
    # predict using polynomial ridge regression model
    print(" best model : polynomial ridge regression " )
    
    ### preprocess the submission data for the ridge regression: 
    
    # 1) clean the test data 
    xtest = fill_na(tX_test, method=np.mean)
    
    # 2) standardize
    xtest1 = my_standardize(xtest)

    
    # 3) project 
    xtest2 = np.dot(xtest1, U)
    
    # 4) build polynomial basis
    xtest_rr = build_polynomial_without_mixed_term(xtest2, degree_best_rr)
    
    
    # 5) predict the labels and save the prediction in a csv file 
    y_pred = predict_labels(weights_rr, xtest_rr)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    
else:
    
    # predict using logistic regression model
    print( "best model : logistic regression model ")
    
    # TO DO : He   