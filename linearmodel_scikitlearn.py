#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 20:37:09 2018

@author: lnovitz
"""
#BELOW CODE IS FROM EXAMPLE 1-1 in "HANDS ON MACHINE LEARNING WITH SCIKITLEARN AND TENSOR FLOW" BY AURELIEN GERON. I plan to use this as a template for building future models.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

#Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands = ',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimitter='\t', encoding = 'latin1', na_values="n/a")

#Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#visualize the data
country_stats.plot(kind = 'scatter', x = "GDP per capita", y = 'Life satisfaction')
plt.show()

#Select a linear model
model = sklearn.linear_model.LinearRegression()

#Train the model
model.fit(X,y)

#Make a prediction for Cyprus
X_new = [[22587]] #Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[5.96242338]]