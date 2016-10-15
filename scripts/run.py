#run.py

# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from functions import *


from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv' 
# TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

################################# TEST OF THE FUNCTIONS ##############################################################

#### Data split:  test 

xtrain,xvalid,ytrain,yvalid=split_data(tX, y, 0.8)

print( " splitting of the data : done ")

### Training of the model : test of least_squares_GD

weights = train_data(xtrain,ytrain,n_regression=1,gamma=0.05,max_iters=20)

print("Training of the model : done")