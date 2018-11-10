#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

X = pd.read_csv('ReceivingTimes.csv')
y = np.array(X['Total time'])
X =X.drop('Total time',axis=1)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.3, random_state = 42)

#feature scaling
sc_X = StandardScaler()
sc_X.fit(xtrain)
xtrain=sc_X.transform(xtrain)
xtest=sc_X.transform(xtest)
from numpy import array
from numpy import reshape
ytrain = np.array(ytrain).reshape((-1,))
ytest = np.array(ytest).reshape((-1,))

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



