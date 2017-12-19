# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:19:50 2017

@author: stco224
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
#Make sure X is a matrix Size (10,1)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

