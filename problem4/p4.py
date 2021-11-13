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

# ---------------------------------------------------
#                   Part 1.1 - Data
#----------------------------------------------------

xtrain = np.load("xtrain.npy")
xtest = np.load("xtest.npy")
ytrain = np.load("ytrain.npy")

xtrain_len = len(xtrain)
xtest_len = len(xtest)

print(f"xtrain_len: {xtrain_len}")


#Reshape Images
#xtrain = xtrain.reshape((xtrain_len,50,50))
#xtest = xtest.reshape((xtest_len,50,50))

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
yval = tf.keras.utils.to_categorical(yval, 2)
ytrain = tf.keras.utils.to_categorical(ytrain, 2)


#Add extra dimension
#xtrain = np.expand_dims(xtrain, axis=2)
#xval = np.expand_dims(xval, axis=2)
#xtest = np.expand_dims(xtest, axis=2)

#%%


# MLP 

mlp = tf.keras.Sequential()
mlp.add(tf.keras.layers.Dense(50,input_shape=(2500,),activation='relu'))
mlp.add(tf.keras.layers.Dense(25,activation='relu'))
mlp.add(tf.keras.layers.Dense(2,activation='sigmoid'))
mlp.summary()

# Early Stopping

early_stop = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
#%%
mlp.compile(optimizer='Adam', loss='categorical_crossentropy')


fit = mlp.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=200, epochs=50, callbacks=[early_stop])
#fit = mlp.fit(xtrain, ytrain, validation_data=(xval, yval), batch_size=200, epochs=200)


plt.figure()
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('MLP loss as a function of the number of\nepochs (with early stopping)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

predict_labels = mlp.predict(xtest, batch_size=200)
predict_labels_index = np.argmax(predict_labels, axis=1)

#mlp_score = accuracy_score(predict_labels_index, test_labels)
#mlp_confusion = confusion_matrix(predict_labels_index, test_labels)

#print('==== MLP with early stopping ====')
#print('Accuracy score = ', mlp_score,'\nConfusion matrix:\n',mlp_confusion)


