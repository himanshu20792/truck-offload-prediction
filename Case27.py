#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble

#Importing dataset
dataset = pd.read_csv('ReceivingTimes.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.ensemble import GradientBoostingRegressor
rf = GradientBoostingRegressor()
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
max_depth = [3, 4, 5]
# Maximum number of levels in tree
n_estimators = [100, 150, 200,250]
learning_rate = [0.001, 0.01, 0.1,1,10]

# Create the random grid
random_grid = {'max_depth': max_depth,
               'n_estimators': n_estimators,
               'learning_rate': learning_rate,
              }

# Use the random grid to search for best hyperparameters
# First create the base model to tune
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3, verbose=2, scoring = 'neg_mean_squared_error', return_train_score = True, random_state=42)
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
max_depth = [2, 3, 4, 5]
# Maximum number of levels in tree
n_estimators = [100, 150,175, 200,250]
learning_rate = [0.001, 0.01,0.05,0.03, 0.1,0.005]

# Create the random grid
random_grid = {'max_depth': max_depth,
               'n_estimators': n_estimators,
               'learning_rate': learning_rate,
              }


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
