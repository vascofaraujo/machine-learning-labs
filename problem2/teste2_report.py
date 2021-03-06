import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import normalize, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest



import matplotlib.pyplot as plt
import seaborn as sns

import eval_scores as eval


xtrain = np.load("Xtrain_Regression_Part2.npy")
ytrain = np.load("Ytrain_Regression_Part2.npy")
xtest = np.load("Xtest_Regression_Part2.npy")


# print("Before data split:\n")
# print(f"xtrain shape: {xtrain.shape}\nytrain shape: {ytrain.shape}\nxtest shape: {xtest.shape}\n\n")
#
# for i in range(20):
#     plt.scatter(xtrain[:,i], ytrain[:,0])
# plt.title("Scatter plot of all 20 dimensions across all 100 data points")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()


# Using Isolated Forests to detect outliers
df = pd.DataFrame(xtrain)
df['y'] = ytrain


possible_outliers_percentage = np.arange(0.1,0.3,0.001)
print(possible_outliers_percentage)

for outliers_percentage in possible_outliers_percentage:
    xtrain = np.load("Xtrain_Regression_Part2.npy")
    ytrain = np.load("Ytrain_Regression_Part2.npy")
    xtest = np.load("Xtest_Regression_Part2.npy")

    xtrain = StandardScaler().fit_transform(xtrain)


    df = pd.DataFrame(xtrain)
    df['y'] = ytrain

    isolated_forest = IsolationForest(contamination=outliers_percentage)

    iso_forest = isolated_forest.fit(df)

    iso_outliers = isolated_forest.predict(df)


    outliers = df[iso_outliers == -1].index.tolist()

    #print(f"Outliers calculated using Isolated Forest; {outliers}")

    for idx in outliers:
        df = df.drop(index=idx)

    xtrain = df.loc[:, df.columns!='y'].to_numpy()
    ytrain = df.loc[:, df.columns=='y'].to_numpy()

    #print(xtrain.shape, ytrain.shape)

    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size = 0.2, random_state=1)



    # Linear
    linear_model = LinearRegression().fit(X_train, y_train)

    y_linear = linear_model.predict(X_test)

    lin_mse = eval.scores(y_test,y_linear,"r")


    # Ridge
    ridge_model = Ridge()

    ridge_model.fit(X_train, y_train)

    y_ridge = ridge_model.predict(X_test)

    ridge_mse = eval.scores(y_test,y_ridge,"r")

    # Lasso
    lasso_model = Lasso()

    lasso_model.fit(X_train, y_train)

    y_lasso = lasso_model.predict(X_test)

    y_lasso = y_lasso.reshape(y_lasso.shape[0], 1)

    lasso_mse = eval.scores(y_test,y_lasso,"r")


    print(f"\nOutlier percentage: {outliers_percentage}, linear: {lin_mse}\n ridge: {ridge_mse}\nlasso: {lasso_mse}")


    k=5

    cv = KFold(n_splits=k)

    lin_mse_list = []
    for train, test in cv.split(xtrain, ytrain):

        y_test = ytrain[test]

        y_linear = linear_model.predict(xtrain[test])

        lin_mse = eval.scores(y_test,y_linear,"r")

        lin_mse_list.append(lin_mse)
    lin_mse_10avg = np.average(lin_mse_list)
    print(f"MSE with KFold: {lin_mse_10avg}")





#########################################################
# Predictions
best_outlier_percentage = 0.29800000000000015

xtrain = np.load("Xtrain_Regression_Part2.npy")
ytrain = np.load("Ytrain_Regression_Part2.npy")
xtest = np.load("Xtest_Regression_Part2.npy")

xtrain = StandardScaler().fit_transform(xtrain)


df = pd.DataFrame(xtrain)
df['y'] = ytrain

isolated_forest = IsolationForest(contamination=outliers_percentage)

iso_forest = isolated_forest.fit(df)

iso_outliers = isolated_forest.predict(df)


outliers = df[iso_outliers == -1].index.tolist()

for idx in outliers:
    df = df.drop(index=idx)

xtrain = df.loc[:, df.columns!='y'].to_numpy()
ytrain = df.loc[:, df.columns=='y'].to_numpy()


# Linear
linear_model = LinearRegression().fit(xtrain, ytrain)

np.save("test_predictions.npy", lasso_model.predict(xtest))
