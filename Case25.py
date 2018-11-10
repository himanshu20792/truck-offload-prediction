#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm

#Importing dataset
X = pd.read_csv('ReceivingTimes.csv')
y = np.array(X['Total time'])
X =X.drop('Total time',axis=1)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor()
mlp.fit(xtrain,ytrain)

ytest = np.array(ytest).reshape(-1,1)
predictions = mlp.predict(xtest)

from sklearn.metrics import mean_squared_error
mean_squared_error(ytest, predictions)
from sklearn.metrics import r2_score
r2_score(ytest, predictions)
mae = np.mean(abs(predictions - ytest))
print(mae)
errors = abs(predictions - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)

#Searching best parameters
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
hidden_layer_sizes = [100,200,300]
# Number of features to consider at every split
activation = ['identity','logistic', 'tanh', 'relu']
# Maximum number of levels in tree
solver = ['lbfgs', 'sgd', 'adam']
alpha = [0.0001, 0.001, 0.1,0.00001]

# Create the random grid
random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'activation': activation,
               'solver': solver,
               'alpha': alpha}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.neural_network import MLPRegressor
rf = MLPRegressor()
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
hidden_layer_sizes = [300,350,400,450]
# Number of features to consider at every split
activation = ['tanh', 'relu']
# Maximum number of levels in tree
solver = ['lbfgs', 'sgd', 'adam']
alpha = [0.0001, 0.001, 0.1,0.00001,1]

# Create the random grid
random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'activation': activation,
               'solver': solver,
               'alpha': alpha}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.neural_network import MLPRegressor
rf = MLPRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
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


mlp.get_params

# Number of trees in random forest
hidden_layer_sizes = [(200,),(300,),(350,)]
# Number of features to consider at every split
activation = ['relu']
# Maximum number of levels in tree
solver = ['lbfgs', 'sgd', 'adam']
alpha = [0.0001, 0.001, 0.1,0.00001,1]

# Create the random grid
random_grid = {'hidden_layer_sizes': hidden_layer_sizes,
               'activation': activation,
               'solver': solver,
               'alpha': alpha}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
from sklearn.neural_network import MLPRegressor
rf = MLPRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
# Fit the random search model
rf_random.fit(xtrain, ytrain)

# Get all of the cv results and sort by the test performance
random_results3 = pd.DataFrame(rf_random.cv_results_).sort_values('mean_test_score', ascending = False)
random_results3.head(10)

rf_random.best_params_

test_predictions_hp3= rf_random.predict(xtest)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,test_predictions_hp3)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions_hp3)
mae = np.mean(abs(test_predictions_hp2 - ytest))
print(mae)
errors = abs(test_predictions_hp2 - ytest)
mape = 100 * np.mean(errors / ytest)
accuracy = 100 - mape
print(mape)
print(accuracy)
