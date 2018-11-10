import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.linear_model import LinearRegression
import math   # yep! going to a bit of maths later!!
from scipy import stats as st # and some stats
import statsmodels.api as sm

#Importing dataset
dataset = pd.read_csv('ReceivingTimes.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Visualizing raw data
plt.scatter(X,y,color='red')
plt.title("No. of rolls VS Time Taken (Raw data)")
plt.xlabel("No. of rolls")
plt.ylabel("Time taken")
plt.show() 

'''#Splitting training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 2)
'''

#adding Polynomial for better fitting of data
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree= 2)
poly_X = poly.fit_transform(X)
"""poly_X = sm.add_constant(poly_X) #For adding constant term to regression model"""
model1 = sm.OLS(y, poly_X)
model2=model1.fit()
"""results = model2.fit()"""
model2.summary()
from sklearn.metrics import mean_squared_error
y_pred = model2.predict(poly_X)
mean_squared_error(y, y_pred)


#ploting the data  
plt.scatter(X,y)
plt.plot(X,poly_regression.predict(poly_features))
plt.title("PolyNomial Regression Experiance Vs Salary with degree 2 ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show()

# higher degree
# Adding Polynominals to the hypothesis 
poly = PolynomialFeatures(degree= 3)
poly_features = poly.fit_transform(X)
poly.fit(X,y)
poly_regression = LinearRegression()
poly_regression.fit(poly_features,y)
#ploting the data  for polynomial regression 
plt.scatter(X,y)
plt.plot(X,poly_regression.predict(poly_features))
plt.title("PolyNomial Regression Experiance Vs Salary with degree 3 ")
plt.xlabel("Experiance ")
plt.ylabel("Salary data ")
plt.show()