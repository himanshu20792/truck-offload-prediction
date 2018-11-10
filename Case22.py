import statsmodels.api as sm
import pylab
import pandas as pd
import statistics as st
import numpy as np

dataset = pd.read_csv('ReceivingTimes.csv')
X_1=dataset.iloc[:,0]
X_2=dataset.iloc[:,1]
X_3=dataset.iloc[:,2]
X_4=dataset.iloc[:,3]

sm.qqplot(X_1, loc = st.mean(X_1), scale = np.std(X_1), line = 's')
sm.qqplot(X_2, loc = st.mean(X_2), scale = np.std(X_2), line = 's')
sm.qqplot(X_3, loc = st.mean(X_2), scale = np.std(X_3), line = 's')
sm.qqplot(X_4, loc = st.mean(X_2), scale = np.std(X_4), line = 's')
