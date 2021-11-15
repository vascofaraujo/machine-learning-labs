# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import normalize, StandardScaler


import matplotlib.pyplot as plt
import seaborn as sns

import eval_scores as eval

xtrain = np.load("xtrain.npy")
ytrain = np.load("ytrain.npy")
print("Before data split:\n")
print(f"xtrain shape: {xtrain.shape}\nytrain shape: {ytrain.shape}\n\n")

xtest = np.load("xtest.npy")


df = pd.DataFrame(xtrain)
df['y'] = ytrain


corr_matrix = df.corr(method="pearson")
sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("pearson correlation")
#plt.show()

#print(corr_matrix["y"])

# I don't see much coorelated columns so no need to reduce the number of features





#### K Fold validation ######

# k folds
k=10

cv = KFold(n_splits=k, shuffle=True)


#%%
#Linear

lin_mse_list = []
all_mse = []

for train, test in cv.split(xtrain, ytrain):

    y_test = ytrain[test]

    linear_model = LinearRegression().fit(xtrain[train], ytrain[train])

    y_linear = linear_model.predict(xtrain[test])

    lin_mse = eval.scores(y_test,y_linear,"r")

    lin_mse_list.append(lin_mse)

lin_mse_10avg = np.average(lin_mse_list)
all_mse.append(lin_mse_10avg)
print(f"MSE_Linear: {lin_mse_10avg}")



#%%

#Ridge

ridge_avg_list = []

for a in np.linspace(0,0.99,100):

    ridge_mse_list = []

    for train, test in cv.split(xtrain, ytrain):

            y_test = ytrain[test]

            ridge = Ridge(alpha=a)

            ridge.fit(xtrain[train], ytrain[train])

            y_ridge = ridge.predict(xtrain[test])

            ridge_mse = eval.scores(y_test,y_ridge,"r")

            ridge_mse_list.append(ridge_mse)

    ridge_mse_10avg = np.average(ridge_mse_list)

    ridge_avg_list.append(ridge_mse_10avg)

    #print(f"(Alpha= {a}) MSE_ridge: {ridge_mse_10avg}")

best_ridge_alpha = np.argmin(ridge_avg_list) / 100
best_mse = np.min(ridge_avg_list)
all_mse.append(best_mse)
print("")
print("Ridge Best Result:")
print(f"(Alpha= {best_ridge_alpha}) MSE_ridge: {best_mse}")

#%%

# #Lasso

start = 0.001
stop = 0.1
num = 100

lasso_avg_list = []

for a in np.linspace(start, stop, num):

    lasso_mse_list = []

    for train, test in cv.split(xtrain, ytrain):

            y_test = ytrain[test]

            lasso = Lasso(alpha=a)

            lasso.fit(xtrain[train], ytrain[train])

            y_lasso = lasso.predict(xtrain[test])

            y_lasso = y_lasso.reshape(k,1)

            lasso_mse = eval.scores(y_test,y_lasso,"r")

            lasso_mse_list.append(lasso_mse)

    lasso_mse_10avg = np.average(lasso_mse_list)

    lasso_avg_list.append(lasso_mse_10avg)

    #print(f"(Alpha= {a}) MSE_lasso: {lasso_mse_10avg}")



best_lasso_alpha = start + ((np.argmin(lasso_avg_list) / num) * (stop - start + 1))
best_mse = np.min(lasso_avg_list)
all_mse.append(best_mse)
print("")
print("Lasso Best Result:")
print(f"(Alpha= {best_lasso_alpha}) MSE_lasso: {best_mse}")


best_method = all_mse.index(min(all_mse))

if best_method == 0:
    print("\nChoosing Linear Regression")

    linear_model = LinearRegression().fit(xtrain, ytrain)
    np.save("test_predictions.npy", linear_model.predict(xtest))

elif best_method == 1:
    print("\nChoosing Ridge")

    ridge_model = Ridge(alpha=best_ridge_alpha)
    ridge_model.fit(xtrain, ytrain)
    np.save("test_predictions.npy", ridge_model.predict(xtest))

elif best_method == 2:
    print("\nChoosing Lasso")

    lasso_model = Lasso(alpha=best_lasso_alpha)
    lasso_model.fit(xtrain, ytrain)
    np.save("test_predictions.npy", lasso_model.predict(xtest))


predictions = np.load("test_predictions.npy")
print(f"Test predictions: {predictions}")
