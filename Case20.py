#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm

#Importing dataset
X = pd.read_csv('ReceivingTimes.csv')
X.head(5)

# Descriptive statistics for each column
X.describe()

# Labels are the values we want to predict
y = np.array(X['Total time'])

# Remove the labels from the features
# axis 1 refers to the columns
X= X.drop('Total time', axis = 1)

# Saving feature names for later use
X_list = list(X.columns)
# Convert to numpy array
X = np.array(X)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(xtrain, ytrain)

# Use the forest's predict method on the training data
train_predictions = rf.predict(xtrain)
#Mean sq error
from sklearn.metrics import mean_squared_error
mean_squared_error(ytrain, train_predictions)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytrain, train_predictions)

# Use the forest's predict method on the test data
test_predictions = rf.predict(xtest)
#Mean sq error
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest, test_predictions)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions)

# Look at parameters used by our current forest
print('Parameters currently in use:\n')
rf.get_params()

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)
test_predictions_hp= rf_random.predict(xtest)
rf_random.best_params_

#We know BASE MODEL PERFORMANCE

#Now we evaluate model performance with Hyperparameters
from sklearn.metrics import mean_squared_error
#R-Sq
from sklearn.metrics import r2_score

def evaluate(model, xtest, ytest):
    predictions = model.predict(xtest)
    errors = abs(predictions - ytest)
    mape = 100 * np.mean(errors / ytest)
    accuracy = 100 - mape
    MSE = mean_squared_error(ytest, predictions)
    R2 = r2_score(ytest, predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} mins.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('MSE = {:0.2f}.'.format(MSE))
    print('R2 = {:0.2f}%.'.format(R2))
    
    return accuracy

best_random = rf_random.best_estimator_
print('Parameters currently in use:\n')
best_random.get_params()
random_accuracy = evaluate(best_random, xtest, ytest)

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 30, 40, 50],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 3,4],
    'n_estimators': [100, 300, 500, 800]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, verbose = 2)

# Fit the grid search to the data
grid_search.fit(xtrain, ytrain)
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, xtest, ytest)


##2nd Grid Search
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid_2 = {
    'bootstrap': [True],
    'max_depth': [10, 30, 40, 50],
    'max_features': [2, 3,4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2,3,4],
    'n_estimators': [10, 50, 100, 200]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search_2 = GridSearchCV(estimator = rf, param_grid = param_grid_2, 
                          cv = 3, verbose = 2)

# Fit the grid search to the data
grid_search_2.fit(xtrain, ytrain)
grid_search_2.best_params_

best_grid_2 = grid_search_2.best_estimator_
grid_accuracy = evaluate(best_grid_2, xtest, ytest)    

##3rd Grid Search
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid_3 = {
    'bootstrap': [True],
    'max_depth': [30, 40, 50, 70],
    'max_features': [3,4],
    'min_samples_leaf': [4,5,6],
    'min_samples_split': [3,4,5],
    'n_estimators': [5,10,20,30,50]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search_3 = GridSearchCV(estimator = rf, param_grid = param_grid_3, 
                          cv = 3, verbose = 2)

# Fit the grid search to the data
grid_search_3.fit(xtrain, ytrain)
grid_search_3.best_params_

best_grid_3 = grid_search_3.best_estimator_
grid_accuracy = evaluate(best_grid_3, xtest, ytest)    
