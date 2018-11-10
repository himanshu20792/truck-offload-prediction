#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm

#Importing dataset
dataset = pd.read_csv('ReceivingTimes.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 42)

from sklearn.svm import SVR
regressor=SVR(kernel='linear',degree=1)
regressor.fit(xtrain, ytrain)

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
regressor.get_params()

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
C = [0.001, 0.01, 0.1,1,10]
# Number of features to consider at every split
kernel = ['linear', 'poly']
# Maximum number of levels in tree
epsilon = [0.001, 0.01, 0.1,1,10]
gamma = [0.001, 0.01, 0.1,1,10]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.svm import SVR
rf = SVR()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)
test_predictions_hp= rf_random.predict(xtest)
rf_random.best_params_

