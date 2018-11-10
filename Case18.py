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

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X, y)
predictions = clf.predict(X)

from sklearn.metrics import mean_squared_error
mean_squared_error(y, predictions)
from sklearn.metrics import r2_score
print(r2_score(y,predictions))