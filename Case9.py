#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm

#Importing dataset
features = pd.read_csv('ReceivingTimes.csv')
features.head(5)

# Descriptive statistics for each column
features.describe()

# Labels are the values we want to predict
labels = np.array(features['Total time'])

# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('Total time', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(train_features, train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

#Mean sq error
from sklearn.metrics import mean_squared_error
mean_squared_error(test_labels, predictions)

#R-Sq
from sklearn.metrics import r2_score
r2_score(test_labels, predictions)

#Visualizing test set results
plt.scatter(test_labels, predictions,color='red')
plt.title("No. of rolls VS Time Taken (Test set)")
plt.xlabel("No. of rolls")
plt.ylabel("Time taken")
plt.show()

# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
