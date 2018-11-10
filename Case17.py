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
#FIT THE MODEL
regressor.fit(xtrain,ytrain)
train_pred=regressor.predict(xtrain)
from sklearn.metrics import mean_squared_error
mean_squared_error(ytrain, train_pred)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytrain, train_pred)

# Use SVR's predict method on the test data
test_pred=regressor.predict(xtest)
mean_squared_error(ytest, test_pred)
r2_score(ytest, test_pred)
    errors = abs(test_pred - ytest)
    mape = 100 * np.mean(errors / ytest)
    accuracy = 100 - mape
    print('Average Error: {:0.4f} mins.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))


#Applying k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = xtrain, y = ytrain, cv = 10)
accuracies.mean()
accuracies.std()

#Applying grid search - To find best models and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel':['linear']},
               {'C': [1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.5,0.1,0.01,0.001]}] 
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = 'r2', cv = 10)
grid_search = grid_search.fit(xtrain, ytrain)
best_accuracy = grid_search.best_score_


#Parameter tuning
regressor=SVR(kernel='rbf',epsilon=1.0)
regressor.fit(xtrain,ytrain)
pred=regressor.predict(xtest)
print(regressor.score(xtest,ytest))
print(r2_score(ytest,pred))
mean_squared_error(ytest, pred)
