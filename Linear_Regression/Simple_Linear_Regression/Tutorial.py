# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:24:54 2017

@author: stco224
"""

#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset X and y
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values


#Break the data into train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)

#Fit the data using simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Get predicted values and the score
y_pred = regressor.predict(X_test)

#Visualize train set
plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Visualize test set
plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
