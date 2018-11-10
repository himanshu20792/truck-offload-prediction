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
rf_plain = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf_plain.fit(xtrain, ytrain)

# Use the forest's predict method on the training data
train_predictions = rf.predict(xtrain)
#Mean sq error
from sklearn.metrics import mean_squared_error
mean_squared_error(ytrain, train_predictions)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytrain, train_predictions)

# Use the forest's predict method on the test data
test_predictions = rf_plain.predict(xtest)
#Mean sq error
from sklearn.metrics import mean_squared_error
mean_squared_error(ytest, test_predictions)
#R-Sq
from sklearn.metrics import r2_score
r2_score(ytest, test_predictions)
    errors = abs(test_predictions - ytest)
    mape = 100 * np.mean(errors / ytest)
    accuracy = 100 - mape
    print('Average Error: {:0.4f} mins.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
