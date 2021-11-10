# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

xtrain=np.load("xtrain.npy")
ytrain=np.load("ytrain.npy")
#
# xtrain = normalize(xtrain, norm="max")
# ytrain = normalize(ytrain, norm="max")

# plt.scatter(xtrain[:,13],ytrain[:,0])
#
# plt.show()

xtest=np.load("xtest.npy")

df = pd.DataFrame(xtrain)
df['y'] = ytrain


corr_matrix = df.corr(method="pearson")
#sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")
#plt.show()

print(corr_matrix["y"])

X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.25, random_state=1)



Xtrain = np.reshape(xtrain,(4,25,20))
Ytrain = np.reshape(ytrain,(4,25,1))

# for i in range(0,10):
#     i *= 10
#     X[i,i:i+10,:] = xtrain[i:i+10,:]
# Xtrain =


score_linear = []
score_ridge = []
score_lasso = []
score_squares = []

for i in range(4):
    
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    
            
        


    linear = LinearRegression().fit(X_train, y_train)
    ridge = Ridge().fit(X_train, y_train)
    lasso = Lasso().fit(X_train, y_train)

    y_linear = linear.predict(X_test)
    y_ridge = ridge.predict(X_test)
    y_lasso = lasso.predict(X_test)

    e_linear = y_linear - y_test
    e_ridge = y_ridge - y_test
    e_lasso = y_lasso - y_test

    et_linear = e_linear.transpose()
    et_ridge = e_ridge.transpose()
    et_lasso = e_lasso.transpose()

    sse_linear = np.matmul(et_linear, e_linear)
    sse_ridge = np.matmul(et_ridge, e_ridge)
    sse_lasso = np.matmul(et_lasso, e_lasso)

    mean_squares_linear = np.abs((y_linear - y_test)**2)
    mean_squares_ridge = np.abs((y_ridge - y_test)**2)
    mean_squares_lasso = np.abs((y_lasso - y_test)**2)

    score_linear.append(sse_linear)#linear.score(X_test, y_test))
    score_ridge.append(sse_ridge)#ridge.score(X_test, y_test))
    score_lasso.append(sse_lasso)#lasso.score(X_test, y_test))
    #score_squares.append(mean_squares)

avg_linear = np.average(np.array(score_linear))
avg_ridge = np.average(np.array(score_ridge))
avg_lasso = np.average(np.array(score_lasso))

print(f"Linear: {avg_linear}\nRidge: {avg_ridge}\nLasso: {avg_lasso}")

# plt.plot(y_linear)
# plt.plot(y_ridge)
# plt.plot(y_lasso)
# plt.legend("Liner", "Ridge", "Lasso")
# plt.show()
