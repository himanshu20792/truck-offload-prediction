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
y = dataset.iloc[:,1].values

#Splitting training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 2)

"""#Fitting SLR into training set
from sklearn.linear_model import LinearRegression #Imported Linear regression Class from sklearn.linear_model library
regressor = LinearRegression() #Making an object of LinearRegression class
regressor.fit(X_train,y_train)
y_pred_train = regressor.predict(X_train)"""


"print(regressor.intercept_, regressor.coef_)

"# Have a look at R sq to give an idea of the fit 
"print('R sq: ',regressor.score(X_train,y_train))
"# and so the correlation is..
"print('Correlation: ', math.sqrt(regressor.score(X_train,y_train)))"


#Fitting SLR using OLS using Statsmodel
"""X_train = sm.add_constant(X_train) #For adding constant term to regression model"""
model1 = sm.OLS(y_train, X_train)
model2=model1.fit()
""""results = model.fit()"""
model2.summary()
from sklearn.metrics import mean_squared_error
y_pred_train = model2.predict(X_train)
mean_squared_error(y_train, y_pred_train)

"""#Predicting the test results
X_test = sm.add_constant(X_test)
y_pred_test = model2.predict(X_test)

#Visualizing training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("No. of rolls VS Time Taken (Training set)")
plt.xlabel("No. of rolls")
plt.ylabel("Time taken")
plt.show()    


#Visualizing test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("No. of rolls VS Time Taken (Test set)")
plt.xlabel("No. of rolls")
plt.ylabel("Time taken")
plt.show()



