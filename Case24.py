#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler


#Pre-model Simple Baseline score calculation
# Function to calculate mean absolute error
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

baseline_guess = np.median(y)
print('The baseline guess is a score of %0.2f' % baseline_guess)
print("Baseline Performance on the test set: MAE = %0.4f" % mae(ytest, baseline_guess))


#Importing dataset
dataset = pd.read_csv('ReceivingTimes.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 42)


#feature scaling
sc_X = MinMaxScaler(feature_range=(0,1))
sc_X.fit(xtrain)
xtrain=sc_X.transform(xtrain)
xtest=sc_X.transform(xtest)
from numpy import array
from numpy import reshape
ytrain = np.array(ytrain).reshape((-1,))
ytest = np.array(ytest).reshape((-1,))

#Baseline model performance
from sklearn.svm import SVR
rf = SVR(kernel='linear', degree=1)
rf.fit(xtrain, ytrain)
ytestpred= rf.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,ytestpred)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, ytestpred)
mae = np.mean(abs(ytestpred - ytest))
print(mae)
errors = abs(ytestpred - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Searching best parameters
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
C = [0.001, 0.01, 0.1,1,10]
# Number of features to consider at every split
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
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
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results1 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results1.head(10)

rf_random.best_params_

test_predictions_hp1= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp1)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp1)
mae = np.mean(abs(test_predictions_hp1 - ytest))
print(mae)
errors = abs(test_predictions_hp1 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Searching best parameters
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
C = [5,10,20]
# Number of features to consider at every split
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
# Maximum number of levels in tree
epsilon = [0.001,0.01, 0.1,1]
gamma = [0.1,0.001,1,10,50]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results2 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results2.head(10)

rf_random.best_params_

test_predictions_hp2= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp2)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp2)
mae = np.mean(abs(test_predictions_hp2 - ytest))
print(mae)
errors = abs(test_predictions_hp2 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Iteration4
# Number of trees in random forest
C = [50,100,150,200]
# Number of features to consider at every split
kernel = ['linear','rbf']
# Maximum number of levels in tree
epsilon = [0.1,1,10,20,50]
gamma = [0.001,0.01,0.1,1,10,50,100]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results4 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results4.head(10)

test_predictions_hp4= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp4)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp4)
mae = np.mean(abs(test_predictions_hp4 - ytest))
print(mae)
errors = abs(test_predictions_hp4 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Iteration5
# Number of trees in random forest
C = [30,40,50,100,150]
# Number of features to consider at every split
kernel = ['linear','rbf']
# Maximum number of levels in tree
epsilon = [0.1,1,10,20,30,40,50]
gamma = [0.001,0.01,0.1,1,10,50,100]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results5 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results5.head(10)

test_predictions_hp5= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp5)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp5)
mae = np.mean(abs(test_predictions_hp5 - ytest))
print(mae)
errors = abs(test_predictions_hp5 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Iteration6
# Number of trees in random forest
C = [40,50,100,150,200]
# Number of features to consider at every split
kernel = ['linear','rbf']
# Maximum number of levels in tree
epsilon = [10,20,30,40,50,100]
gamma = [0.001,0.01,0.1,1,10,50,100]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results6 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results6.head(10)

test_predictions_hp6= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp6)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp6)
mae = np.mean(abs(test_predictions_hp6 - ytest))
print(mae)
errors = abs(test_predictions_hp6 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Iteration7
# Number of trees in random forest
C = [90,95,100,105,110,120]
# Number of features to consider at every split
kernel = ['linear','rbf']
# Maximum number of levels in tree
epsilon = [10,12,15,18,20,22,24]
gamma = [0.001,0.01,0.1,1,10,50,100]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results7 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results7.head(10)

test_predictions_hp7= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp7)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp7)
mae = np.mean(abs(test_predictions_hp7 - ytest))
print(mae)
errors = abs(test_predictions_hp7 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Iteration7
# Number of trees in random forest
C = [90,95,100,105,110,120]
# Number of features to consider at every split
kernel = ['linear','rbf']
# Maximum number of levels in tree
epsilon = [10,12,15,18,20,22,24]
gamma = [0.001,0.01,0.1,1,10,50,100]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results7 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results7.head(10)

test_predictions_hp7= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp7)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp7)
mae = np.mean(abs(test_predictions_hp7 - ytest))
print(mae)
errors = abs(test_predictions_hp7 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Iteration8
# Number of trees in random forest
C = [80,85,90,95,100,105]
# Number of features to consider at every split
kernel = ['linear','rbf']
# Maximum number of levels in tree
epsilon = [15,18,20,22,24,26,28]
gamma = [0.001,0.01,0.1,1,10,20,30,40,50,100]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results8 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results8.head(10)

test_predictions_hp8= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp8)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp8)
mae = np.mean(abs(test_predictions_hp8 - ytest))
print(mae)
errors = abs(test_predictions_hp8 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Iteration9
# Number of trees in random forest
C = [80,85,90,91,92,95,100,105]
# Number of features to consider at every split
kernel = ['linear','rbf']
# Maximum number of levels in tree
epsilon = [10,11,12,13,14,15,18,20,22,24,26,28]
gamma = [0.001,0.01,0.1,1,10,20,30,40,50,100]

# Create the random grid
random_grid = {'C': C,
               'kernel': kernel,
               'epsilon': epsilon,
               'gamma': gamma}

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results9 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results9.head(10)

test_predictions_hp9= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp9)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp9)
mae = np.mean(abs(test_predictions_hp9 - ytest))
print(mae)
errors = abs(test_predictions_hp9 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)
rf_random.best_estimator_
rf_random.best_params_
