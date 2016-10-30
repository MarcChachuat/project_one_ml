# readme.md

Pattern Classification And Machine Learning
Project One
Group 16


Content : 
1°) Our strategy
2°) Description of the folder
3°) The run.py in detail



1°) Our strategy

	a) the global strategy

For this project, our idea was to first compare the efficiency of the different models seen in the course, and then to use the best one for the prediction. 
In a first time, we tested manually the performance of the linear regression, of the ridge regression and of the logistic regression, trained over a training set and tested over a validation set, for different parameters. As ridge regression (repectively : regularized logistic regression) seemed more performant than least-squares method (respectively : logistic regression), and as the second one is a particular case of the first one, we restricted our study to two models : 
- the ridge regression
- the regularized logistic regression

For each of these two models, we wanted to obtain the best feasible performance, which implied to : 
- define how to systematically measure the performance
- pre-process the data
- select the best parameter for each model  

Thus, we had for each model : 
- an optimal set of parameters
- a measure of performance associaed to this set of parameter

Finally, we kept the model with the best performance (associated to its optimal parameters) and use it to predict the labels. 

	b) how to measure a performance ?
	
For this project, we had to predict a binary label, mainly using linear or logistic model. 
We could have used the mean square error to measure the performance of the model, but we were only focused on the sign of our prediction. 
And as we used polynomial basis, the values where quite large.
Therefore, the mean square error wasn't very accurate (in the sense that it is not a bad thing to be far from our target, if we have the same sign).

Thus, we decided to define the performance of a parametred model, trained over a training set as the score of its prediction over the validation set (in %). 
By choosing the score over the validation set rather than over the training set, we intended to decrease the risk of overfitting   

	c) the data pre-processing
	
To guarantee a good performance (for a reasonnable running time), it was essential to pre-process the data. 
To do so, we disposed of a few tools seen in the course : standardization, polynomial basis building, restrict outliers by taking their logrithms and indicating missing values. 

But these methods should be applied to 'clean' data. 
In our data set, a lot of data were missing (and replaced by -999), what could be problematic. 

We decided to deal with this issue in three steps : 
- first : remove the -999 and replace them by the mean/median of the feature over the non missing data.
- second : Yet, this had a drawback : if there was some information in the fact that a data was missing, we didn't take this information into account. 
Therefore, we tried to replace this information for some columns by a new column of binaries indicating if there was a missing value of not.
- third : some of the features have outliers and their peaks are not centered in the middle. We will apply logs to these features so that the outliers will not influence too much and the distribution of data it is not skewed.
Combinating these three tricks gave us satisfiying results. 

After the cleaning of the data, we standardized the data, as seen in course. 
But the dimension of our data set was very large and it was problematic for using polynomial basis (or for running time concerns).

Then, we implemented our own PCA, to keep only a few dimensions without loosing too much information. 
(the principle of PCA being to project over the linear subspace associated to the main eigenvalues of the covariance matrix)

Finally, we could use polynomial basis to let the prediction depend not only of the features but also of their powers. 

	d) the parameter selection for each model

The two models (ridge and regularized logistic regression) we used for our prediction need some parameters : 
- the regularization parameter
- the degree of the polynomial basis
- training datasets

Rather than randomly trying parameters, we inspired ourselves from the grid search and performed a grid search over the set of parameters (lambda, degree).
For each model, for each parameter (lambda,degree) over a large range, we obtain a score (% of accuracy) over the validation set when the model is trained over the training set. 
For each model, we finally keep the couple (lambda,degree) associated to the best score 

In order to have a reasonable training data sets, we plot a learning curve of the size of learning datapoints and choose 25000 to be size of training datasets for logistic regression.

	e) optimization
Inorder to speedup the gradient descent, *Accelerated Gradient Descent with Restart* is implemented and used. This method gives much better convergence rate. The "line search" method is not used here because we find it doesn't work as well as we expected. Unlinke Newton/Quasi Newton method, AGDR didn't require oracla information about derivative of loss function. Besides, AGDR use less memory than Newton's method.

	e) prediction
	
Then, by comparing the two scores over the validation set, we know which model is the most performant.
We use it for the final prediction.




2°) Description of the folder

	-run.py : executable file, contains the process that lead to the creation of our submission csv file.
	
	-implementations.py : contains the six basics methods seen in the course and their main subroutines.
	
	-functions.py : contains auxiliary functions, mainly a function that calls linear method in a standardized way
	
	-helpers.py : a file provided by the teachers containing helpful methods
	
	-proj1_helpers.py : an other file provided by the teachers, more specific to this project
	
	-costs.py : contains the cost functions provided by the teachers and additional cost functions we added during the project.
	
	-data_preprocessing.py : contains the methods dealing with the preprocessing of the data