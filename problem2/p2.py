# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import normalize, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from scipy import stats



import matplotlib.pyplot as plt
import seaborn as sns

import eval_scores as eval


xtrain = np.load("Xtrain_Regression_Part2.npy")
ytrain = np.load("Ytrain_Regression_Part2.npy")
xtest = np.load("Xtest_Regression_Part2.npy")

maxy = np.max(ytrain)
miny = np.min(ytrain)

span = abs(maxy - miny)

print(f"Outcome value span: {span}")
print("Before data split:\n")
print(f"xtrain shape: {xtrain.shape}\nytrain shape: {ytrain.shape}\nxtest shape: {xtest.shape}\n\n")




# for i in range(0,20):
#     plt.figure()
#     plt.scatter(range(0,100),xtrain[:,i])

# Using Inter Quartile range to classify outliers



######################################################################################################

# indexes = []
# for f in range(0,20):
#     z_scores = stats.zscore(xtrain[:,f])
#     low_indexes = np.where(z_scores < -2.65)
#     indexes = np.append(indexes, low_indexes[0])
#     high_indexes = np.where(z_scores > 2.65)
#     indexes = np.append(indexes, high_indexes[0])

# indexes = np.unique(indexes)
# indexes = np.sort(indexes)[::-1]

# indexes = indexes.astype(int)

# ind_size = len(indexes)
# for ind in range(0,ind_size):
#     xtrain = np.delete(xtrain, (indexes[ind]),axis=0)
#     ytrain = np.delete(ytrain, (indexes[ind]),axis=0)
    
# xtrain_size = len(xtrain)

# print(f"Size of xtrain after trimming: {xtrain_size}")

###########################################################################################################
    
# percentile_75 = np.percentile(ytrain, 75)
# percentile_25 = np.percentile(ytrain, 25)

# print(f"25%: {percentile_25}, 75%: {percentile_75}\n\n")

# IQR = percentile_75 - percentile_25

# outlier_thresh = 1.5

# upper_bound = (outlier_thresh*IQR) + percentile_75
# lower_bound = percentile_25 - (outlier_thresh*IQR)


# print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}\n\n")

# df = pd.DataFrame(xtrain)
# df['y'] = ytrain

# print("Potencial outliers smaller than lower bound:")
# smaller_indexes = df.index[df['y']<lower_bound].tolist()
# print(smaller_indexes)

# print("Potencial outliers bigger than upper bound:")
# bigger_indexes = df.index[df['y']>upper_bound].tolist()
# print(bigger_indexes)

# # Remove outliers calculated using the IQR method
# for idx in smaller_indexes:
#     df = df.drop(index=idx)
# for idx in bigger_indexes:
#     df = df.drop(index=idx)
    
    
# Standardizie Data

#xtrain = StandardScaler().fit_transform(xtrain)


# pca = PCA(n_components=10)
# # With 13 features we can explain 79% of the data variance
#
# pca_components = pca.fit_transform(x_normalized)
#
# print(np.sum(pca.explained_variance_ratio_))
#
# pca_df = pd.DataFrame(pca_components)
# pca_df['y'] = ytrain
#
# print(pca_df.head())
#
# ytrain = pca_df.loc[:, pca_df.columns=='y'].to_numpy()
# xtrain = pca_df.loc[:, pca_df.columns!='y'].to_numpy()
#
# print(xtrain.shape, ytrain.shape)



#### K Fold validation ######

# k folds
k=4

cv = KFold(n_splits=k, shuffle=True)

all_mse = []


#%%
#Linear

linear_model = LinearRegression().fit(xtrain, ytrain)

y_linear = linear_model.predict(xtrain)

indexes = []

plt.scatter(range(0,2),(ytrain[78],ytrain[85]), color='blue')
plt.scatter(range(0,2),(y_linear[78],y_linear[85]), color = 'red')

indexes = np.where(abs(ytrain - y_linear)>5)
indexes = indexes[0]
print(indexes)
indexes = np.unique(indexes)
indexes = np.sort(indexes)[::-1]

indexes = indexes.astype(int)

ind_size = len(indexes)
for ind in range(0,ind_size):
    xtrain = np.delete(xtrain, (indexes[ind]),axis=0)
    ytrain = np.delete(ytrain, (indexes[ind]),axis=0)
    
xtrain_size = len(xtrain)

print(f"Size of xtrain after trimming: {xtrain_size}")


lin_mse_list = []


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

start = 0
stop = 1
num = 100

# ridge_avg_list = []

# for a in np.linspace(start,stop,num):

#     ridge_mse_list = []

#     for train, test in cv.split(xtrain, ytrain):

#             y_test = ytrain[test]

#             ridge = Ridge(alpha=a, normalize=True)

#             ridge.fit(xtrain[train], ytrain[train])

#             y_ridge = ridge.predict(xtrain[test])

#             ridge_mse = eval.scores(y_test,y_ridge,"r")

#             ridge_mse_list.append(ridge_mse)

#     ridge_mse_10avg = np.average(ridge_mse_list)

#     ridge_avg_list.append(ridge_mse_10avg)

#     #print(f"(Alpha= {a}) MSE_ridge: {ridge_mse_10avg}")

# best_ridge_alpha = start + ((np.argmin(ridge_avg_list) / num) * (stop - start))
# best_mse = np.min(ridge_avg_list)
# all_mse.append(best_mse)
# print("")
# print("Ridge Best Result:")
# print(f"(Alpha= {best_ridge_alpha}) MSE_ridge: {best_mse}")




train_set=np.column_stack((xtrain, ytrain))

ridge = Ridge(alpha=0.13, normalize=True)

ridge.fit(xtrain, ytrain)

y_ridge = ridge.predict(xtrain)



indexes = []

indexes = np.where(abs(ytrain - y_ridge)>5)
indexes = indexes[0]
print(indexes)
indexes = np.unique(indexes)
indexes = np.sort(indexes)[::-1]

indexes = indexes.astype(int)

ind_size = len(indexes)
for ind in range(0,ind_size):
    xtrain = np.delete(xtrain, (indexes[ind]),axis=0)
    ytrain = np.delete(ytrain, (indexes[ind]),axis=0)
    
xtrain_size = len(xtrain)

print(f"Size of xtrain after trimming: {xtrain_size}")





ridge_avg_list1 = []

for b in np.linspace(start,stop,num):

    ridge_mse_list1 = []

    for train, test in cv.split(xtrain, ytrain):

            y_test = ytrain[test]

            ridge1 = Ridge(alpha=b, normalize=True)

            ridge1.fit(xtrain[train], ytrain[train])

            y_ridge1 = ridge1.predict(xtrain[test])

            ridge_mse1 = eval.scores(y_test,y_ridge1,"r")

            ridge_mse_list1.append(ridge_mse1)

    ridge_mse_10avg1 = np.average(ridge_mse_list1)

    ridge_avg_list1.append(ridge_mse_10avg1)

    #print(f"(Alpha= {b}) MSE_ridge: {ridge_mse_10avg1}")

best_ridge_alpha1 = start + ((np.argmin(ridge_avg_list1) / num) * (stop - start))
best_mse1 = np.min(ridge_avg_list1)
all_mse.append(best_mse1)
print("")
print("Ridge Best Result:")
print(f"(Alpha= {best_ridge_alpha1}) MSE_ridge: {best_mse1}")


#xtrain1=np.reshape(trimmed_set[0], (noOutliers[0].size, 1))
#ytrain1=np.reshape(trimmed_set[1], (noOutliers[1].size, 1))


#%%

#Lasso

start = 0.001
stop = 0.1
num = 100

lasso_avg_list = []

for a in np.linspace(start, stop, num):

    lasso_mse_list = []

    for train, test in cv.split(xtrain, ytrain):

            y_test = ytrain[test]

            lasso = Lasso(alpha=a, normalize=True)

            lasso.fit(xtrain[train], ytrain[train])

            y_lasso = lasso.predict(xtrain[test])


            y_lasso = y_lasso.reshape(len(y_lasso),1)

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


# best_method = all_mse.index(min(all_mse))

# if best_method == 0:
#     print("\nChoosing Linear Regression")

#     linear_model = LinearRegression().fit(xtrain, ytrain)
#     np.save("test_predictions.npy", linear_model.predict(xtest))

# elif best_method == 1:
#     print("\nChoosing Ridge")

#     ridge_model = Ridge(alpha=best_ridge_alpha)
#     ridge_model.fit(xtrain, ytrain)
#     np.save("test_predictions.npy", ridge_model.predict(xtest))

# elif best_method == 2:
#     print("\nChoosing Lasso")

#     lasso_model = Lasso(alpha=best_lasso_alpha)
#     lasso_model.fit(xtrain, ytrain)
#     np.save("test_predictions.npy", lasso_model.predict(xtest))


# predictions = np.load("test_predictions.npy")
# print(f"Test predictions: {predictions}")

#%%

# poly_predictions = np.zeros((100, 20))

# all_poly_models = []
# for a in range(0,20):

#     poly_model = np.poly1d(np.polyfit(xtrain[:,a], ytrain[:,0], 3))

#     all_poly_models.append(poly_model)



# poly_model = np.poly1d(np.mean(all_poly_models, axis=0))
# for train, test in cv.split(xtrain, ytrain):


#     y_test = ytrain[test]

#     x_test = np.average(xtrain[test], axis=1)

#     pca = PCA(n_components=1)

#     x_test = pca.fit_transform(xtrain[test])

#     y_poly = poly_model(x_test)

#     y_poly = y_poly.reshape(k,1)

#     print(y_test.shape, y_poly.shape)

#     poly_mse = eval.scores(y_test,y_poly,"r")

#     print(poly_mse)

poly = PolynomialFeatures(degree=2)
poly_variables = poly.fit_transform(xtrain)

poly_var_train, poly_var_test, res_train, res_test = train_test_split(poly_variables, ytrain, test_size = 0.3, random_state = 4)

poly_model = LinearRegression().fit(poly_var_train, res_train)

poly_predictions = poly_model.predict(poly_var_test)

poly_mse = eval.scores(res_test,poly_predictions,"r")


print(f"Score: {poly_mse}")


myline = np.linspace(-2, 2, 30)
for i in range(0,1):
    plt.scatter(myline, res_test)
    plt.plot(myline , poly_predictions, color="red")
    plt.show()
