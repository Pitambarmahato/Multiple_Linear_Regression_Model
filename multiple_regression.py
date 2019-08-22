# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 20:53:39 2019

@author: pitam
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
#Importing dataset
df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding Dummy Variable Trap
X = X[:, 1:]

#Splitting Dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Multiple Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test results
y_pred = regressor.predict(X_test)

#Implementing Backwart Elimination model
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Eliminating the Highest p-value and then again fit the model
X_opt = X[:, [0, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()  #This summary shows that marketing spend is slightly greater than that of the 50% of level of significance but so we can say that marketing spend and the r & d spend is the optimal model for the multiple regression but we have to remove the marketing spend because its value is also greater than our level of significance.

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() #This model is the optimal model R&D spend gives the optimal model for our dataset.


#we can get the optimal result by using the R&D Spend column only for our given dataset.
