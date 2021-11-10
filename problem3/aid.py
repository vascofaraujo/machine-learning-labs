# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:19:42 2021

@author: João
"""

#%%
# No Early Stopping

#New model to reset weights
mlp_no_early_stopping = tf.keras.Sequential()
mlp_no_early_stopping.add(tf.keras.layers.Flatten(input_shape = (28, 28, 1)))
mlp_no_early_stopping.add(tf.keras.layers.Dense(32,activation='relu'))
mlp_no_early_stopping.add(tf.keras.layers.Dense(64,activation='relu'))
mlp_no_early_stopping.add(tf.keras.layers.Dense(10,activation='softmax'))
#mlp_no_early_stopping.summary()

mlp_no_early_stopping.compile(optimizer='Adam', loss='categorical_crossentropy')

fit_no_early_stopping = mlp_no_early_stopping.fit(xtrain, ytrain, 
                        validation_data=(xval, 
                        yval), batch_size=200, epochs=200)

plt.figure()
plt.plot(fit_no_early_stopping.history['loss'])
plt.plot(fit_no_early_stopping.history['val_loss'])
plt.title('MLP loss as a function of the number of\nepochs (without early stopping)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])
plt.show()

predict_labels = mlp_no_early_stopping.predict(xtest, batch_size=200)
predict_labels_index = np.argmax(predict_labels, axis=1)
# mlp_score = accuracy_score(predict_labels_index, test_labels)
# mlp_confusion = confusion_matrix(predict_labels_index, test_labels)
# print('==== MLP without early stopping ====')
# print('Accuracy score = ', mlp_score,'\nConfusion matrix:\n',mlp_confusion)


# ---------------------------------------------------
#       Part 1.3 - CNN
#----------------------------------------------------

cnn = tf.keras.Sequential()
#layer 0
cnn.add(tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape = (28, 28, 1)))
#layer 1
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#layer 2
cnn.add(tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape = (13, 13, 16)))
#layer 3
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#layer 4
cnn.add(tf.keras.layers.Flatten(input_shape = (5, 5, 16)))
#layer 5
cnn.add(tf.keras.layers.Dense(32,activation='relu'))
#layer 6
cnn.add(tf.keras.layers.Dense(10,activation='softmax'))

cnn.summary()


early_stop = tf.keras.callbacks.EarlyStopping(patience=10,
                                              restore_best_weights=True)

cnn.compile(optimizer='Adam', loss='categorical_crossentropy')
fit = cnn.fit(xtrain, ytrain, validation_data=(xval, 
                        yval), batch_size=200, epochs=200, 
                        callbacks=[early_stop])


plt.figure()
plt.plot(fit.history['loss'])
plt.plot(fit.history['val_loss'])
plt.title('CNN loss as a function of the number of\nepochs (with early stopping)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training loss', 'Validation loss'])
plt.show()


predict_labels = cnn.predict(xtest, batch_size=200)
predict_labels_index = np.argmax(predict_labels, axis=1)
# cnn_score = accuracy_score(predict_labels_index, test_labels)
# cnn_confusion = confusion_matrix(predict_labels_index, test_labels)

# print('==== CNN with early stopping ====')
# print("Score: ", cnn_score)
# print("Confusion matrix:\n", cnn_confusion)