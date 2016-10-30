#run.py

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from helpers import *
from proj1_helpers import *
from costs import *
from data_preprocessing import *
import os
from implementations import *

#################################################################################################################################
############################ Part I : Load the training data into feature matrix, class labels, and event ids: ##################
DATA_TRAIN_PATH = '../data/train.csv' 

# download train data / cache data
if os.path.exists('../data/y.npy') and os.path.exists('../data/tX.npy') and os.path.exists('../data/ids.npy'):
    y   = np.load('../data/y.npy'  )
    tX  = np.load('../data/tX.npy' )
    ids = np.load('../data/ids.npy')
else:
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    np.save('../data/y',     y)
    np.save('../data/tX',   tX)
    np.save('../data/ids', ids)

print("loading of the data : done")

##### Select some data
num_samples=30000
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
# ##### Reduce the dimension : Perform a PCA on the training data
xtrain1ter = my_standardize(xtrain1bis)
xvalid1ter = my_standardize(xvalid1bis)

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

# Classify features based on their properties
Features_Good            = [7, 10, 14, 15, 17, 18, 20]
Features_with_outlier    = [3, 8, 19, 23, 26, 29]
Features_skewed          = [1, 2, 5, 9, 13, 16, 21]
Features_missing_entry   = [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
Features_categorical     = [11, 12, 22]
Features_non_categorical = [x for x in range(30) if x not in Features_categorical]
Features_using_log       = np.union1d(Features_with_outlier, Features_skewed)

# Setups for logistic regression
degree     = 3
iterations = 10000

# Types of regularizor used
regularizor = regularizor_lasso
lambda_     = 1

# Transform y to take value in {0, 1} 
transformed_y = transform_y(ytrain)

def feature_engineering(xtrain1, input_mean_x=None, input_std_x=None):
    print ("Begin feature engineering...")
    # For non categorical features, build polynomials
    # Replace the missing values by the mean over other samples
    xtrain1bis = fill_na(xtrain1, method=np.median)

    # create a feature to indicate if some of the fields of a datapoint is missing
    missing_indicator_tX = missing_indicator(xtrain1)
    # take logs of some features whose value is non-negative
    log_tX               = logs_of_features (xtrain1bis, Features_using_log)
    # decompose categorical features into binary ones
    decomposed_tX        = decompose_categorical_features(xtrain1bis)
    # mixed polynomial terms
    mixed_tX             = mixed_features(xtrain1bis, Features_non_categorical)

    # polynomial terms of non categorical features
    poly_tX     = build_polynomial_without_mixed_term(xtrain1bis[:, Features_non_categorical], degree=degree)
    # polynomial terms of logs of features
    log_poly_tX = build_polynomial_without_mixed_term(log_tX    , degree=degree)
    # polynomial terms of inverse of features
    # inv_poly_tX = build_polynomial_without_mixed_term(inverse_tX, degree=degree)

    # Build a design matrix
    print ("Concatenating features...")
    design_matrix = np.c_[poly_tX, decomposed_tX, missing_indicator_tX, log_poly_tX, mixed_tX]

    print ("Standardizing features...")
    logistic_tX, mean_x, std_x = standardize(design_matrix, input_mean_x, input_std_x)
    return logistic_tX, mean_x, std_x

logistic_tX, mean_x, std_x = feature_engineering(xtrain1)

# The loss function of logistic regression is a L-lipschitz function, thus the 
# learning rate gamma can be determined by 1/L.
L = np.real(abs(np.linalg.eigvals(logistic_tX.T @ logistic_tX)).max())

# Output settings for this problem
print("--------------------- Setups of logistic regression-------------------------")
print("Accelerated Gradient Descent with Restart is used for solving Higgs Problem.")
print("Learning rate used in Logistic regression = ", 1/L)
print("lambda used in Logistic regression        = ", lambda_)
print("degree of polynomial used in modeling     = ", degree)
print("maximum number of iterations              = ", iterations)
print("Size of trainging data size               = ", len(transformed_y))
print("Number of features used                   = ", logistic_tX.shape[1])

if regularizor == regularizor_lasso:
    print("the regularizor used here is Lasso")
else:
    print("the regularizor used here is Ridge")

print("--------------------- Begin training -------------------------")

# solve the optimization problem with logistic regression
weights_lg, losses = logistic_AGDR(transformed_y, logistic_tX, gamma=1/L, \
                   max_iters = iterations, lambda_=lambda_, regularizor=regularizor)

def prediction_accuracy(y, y_pred):
    return np.mean(y == y_pred)

# Calculate training error of logistic regression.
logistic_pred_y = predict_labels(weights_lg, logistic_tX)
training_accuracy = prediction_accuracy(ytrain, logistic_pred_y)
print ("--------------Performance of Logistic regression-------------")
print ("Training error of Logistic regression = ", training_accuracy)
err_log = losses[-1] / len(transformed_y)
err_rr = 1
# Performance on validation set 
validation_tx, _, _ = feature_engineering(xvalid1, mean_x, std_x)

print ("Validation accuracy                   = ", prediction_accuracy(yvalid, predict_labels(weights_lg, validation_tx)))
print ("------------------------------------------------------------")
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
print ("Loading Test data")
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
    xtest1, _, _ = my_standardize(xtest)
    
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
    
    ### preprocess the submission data for the ridge regression: 
    xtest_lg, _, _ = feature_engineering(tX_test, mean_x, std_x)
    
    # 5) predict the labels and save the prediction in a csv file 
    y_pred = predict_labels(weights_lg, xtest_lg)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    