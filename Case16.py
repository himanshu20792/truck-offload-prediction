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

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(train_features,train_labels)

predictions = mlp.predict(test_features)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(test_labels,predictions))

from sklearn.metrics import mean_squared_error
mean_squared_error(test_labels, predictions)

from sklearn.metrics import r2_score
r2_score(test_labels, predictions)


