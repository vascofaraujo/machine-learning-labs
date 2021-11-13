# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 12:41:20 2021

@author: João
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import sklearn.preprocessing as sklpp
from sklearn.model_selection import KFold

from sklearn.inspection import permutation_importance


#%%
# ---------------------------------------------------
#                   Part 1.1 - Data
#----------------------------------------------------

xtrain = np.load("../../../Xtrain_Classification_Part2.npy")
xtest = np.load("../../../Xtest_Classification_Part2.npy")
ytrain = np.load("../../../Ytrain_Classification_Part2.npy")

xtrain_len = len(xtrain)
xtest_len = len(xtest)

print(f"xtrain_len: {xtrain_len}")

ytrain = np.array(ytrain, dtype='int')

xtrain /= 255
xtest /= 255

#Show images
#plt.imshow(xtrain[0])
#plt.show()
#plt.imshow(xtest[1])
#plt.show()



classes = np.unique(ytrain)

weights = compute_class_weight("balanced", classes=classes, y=ytrain)
weights = dict(enumerate(weights))


#Split 20% of the training set into a validation set
xtrain, xval, ytrain, yval = train_test_split(
            xtrain, ytrain, test_size=0.2, train_size=0.8)

#Convert train labels to one-hot encoding
yval = tf.keras.utils.to_categorical(yval, 4)
ytrain = tf.keras.utils.to_categorical(ytrain, 4)





#%%


# MLP 

mlp = tf.keras.Sequential()
mlp.add(tf.keras.layers.Dense(64,input_shape=(2500,),activation='relu'))
mlp.add(tf.keras.layers.Dense(16,activation='relu'))
mlp.add(tf.keras.layers.Dense(4,activation='softmax'))
mlp.summary()

# Early Stopping

early_stop = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
#%%
mlp.compile(optimizer='Adam', loss='categorical_crossentropy')


fit = mlp.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=200, epochs=200, callbacks=[early_stop], class_weight=weights)
#fit = mlp.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=200, epochs=200)


plt.figure()
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('MLP loss as a function of the number of\nepochs (with early stopping)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

#%%

predict_labels = mlp.predict(xtest, batch_size=200)
predict_labels_index = np.argmax(predict_labels, axis=1)

np.save("y.npy", predict_labels_index)

ytrain = np.load("y.npy")



#%%


# ---------------------------------------------------
#       Part 1.3 - CNN
#----------------------------------------------------

xtrain = np.load("../../../Xtrain_Classification_Part2.npy")
xtest = np.load("../../../Xtest_Classification_Part2.npy")
ytrain = np.load("../../../Ytrain_Classification_Part2.npy")

xtrain_len = len(xtrain)
xtest_len = len(xtest)

print(f"xtrain_len: {xtrain_len}")

classes = np.unique(ytrain)

weights = compute_class_weight("balanced", classes=classes, y=ytrain)
weights = dict(enumerate(weights))



#Reshape Images
xtrain = xtrain.reshape((xtrain_len,50,50))
xtest = xtest.reshape((xtest_len,50,50))


#Add extra dimension
xtrain = np.expand_dims(xtrain, axis=3)
xtest = np.expand_dims(xtest, axis=3)

xtrain /= 255
xtest /= 255

#Show images
#plt.imshow(xtrain[0])
#plt.show()
#plt.imshow(xtest[1])
#plt.show()


# labels
#labels = np.array([0,1])


#Split 20% of the training set into a validation set
xtrain, xval, ytrain, yval = train_test_split(
            xtrain, ytrain, test_size=0.2, train_size=0.8)

#Convert train labels to one-hot encoding
yval = tf.keras.utils.to_categorical(yval, 4)
ytrain = tf.keras.utils.to_categorical(ytrain, 4)


#%%

cnn = tf.keras.Sequential()
#layer 0
cnn.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='linear', input_shape = (50,50,1)))
#layer 1
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#layer 2
cnn.add(tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu'))
#layer 3
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#layer 4
cnn.add(tf.keras.layers.Flatten())
#layer 5
cnn.add(tf.keras.layers.Dense(8,activation='linear'))
#layer 6
cnn.add(tf.keras.layers.Dense(4,activation='softmax'))

cnn.summary()


early_stop = tf.keras.callbacks.EarlyStopping(patience=10,
                                              restore_best_weights=True)

cnn.compile(optimizer='Adam', loss='categorical_crossentropy')
fit = cnn.fit(xtrain, ytrain, validation_data=(xval, 
                        yval), batch_size=200, epochs=200, 
                        callbacks=[early_stop], class_weight=weights)


plt.figure()
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('CNN loss as a function of the number of\nepochs (with early stopping)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])
plt.show()


predict_probs = cnn.predict(xtest, batch_size=200)
predict_labels = np.argmax(predict_probs, axis=1)

#%%
np.save("y.npy", predict_labels_index)

predict_labels = np.load("y.npy")




#%%



#----------------------------
#
#   Random Forests
#
#------------------------------


xtrain = np.load("../../../Xtrain_Classification_Part2.npy")
xtest = np.load("../../../Xtest_Classification_Part2.npy")
ytrain = np.load("../../../Ytrain_Classification_Part2.npy")

xtrain_len = len(xtrain)
xtest_len = len(xtest)

print(f"xtrain_len: {xtrain_len}")

classes = np.unique(ytrain)

weights = compute_class_weight("balanced", classes=classes, y=ytrain)
weights = dict(enumerate(weights))


xtrain /= 255
xtest /= 255

#Show images
#plt.imshow(xtrain[0])
#plt.show()
#plt.imshow(xtest[1])
#plt.show()


# labels
#labels = np.array([0,1])


#Split 20% of the training set into a validation set
xtrain, xval, ytrain, yval = train_test_split(
            xtrain, ytrain, test_size=0.2, train_size=0.8)

#Convert train labels to one-hot encoding
yval = tf.keras.utils.to_categorical(yval, 4)
ytrain = tf.keras.utils.to_categorical(ytrain, 4)



#%%


rf = RandomForestClassifier(n_estimators=200, criterion="gini", class_weight="balanced")
#rf = RandomForestClassifier(max_depth=10, random_state=0)
rf.fit(xtrain, ytrain)


yval_pred = rf.predict(xval)

print('Accuracy: ', accuracy_score(yval, yval_pred))

predict_probs = rf.predict(xtest)
predict_labels = np.argmax(predict_probs, axis=1)
#%%
np.save("y.npy", predict_labels)
#%%
y = np.load("y.npy")

