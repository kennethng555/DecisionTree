# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

def get_metrics(y, y_pred):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_pred[i] == 1:
            tp = tp + 1
        elif y[i] == 1 and y_pred[i] == 0:
            fn = fn + 1
        elif y[i] == 0 and y_pred[i] == 1:
            fp = fp + 1
        else:
            tn = tn + 1
    sen = tp / (fn + tp)
    spec = tn / (tn + fp)
    acc = (tp + tn) / len(y)
    return acc, sen, spec

# import data
df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# split train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ANN
classifier = Sequential()
# input layer
classifier.add(Dense(6, init = 'uniform', input_dim = len(X_train[0]), activation = 'relu'))
# hidden layer
classifier.add(Dense(8, init = 'uniform', activation = 'relu'))
# output layer
classifier.add(Dense(1, activation = 'sigmoid'))
# compile ann
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 5, epochs = 100)

# predict
y_predtr = classifier.predict(X_train)
y_predte = classifier.predict(X_test)
y_predtr = np.round(y_predtr)
y_predte = np.round(y_predte)

acc_train, sen_train, spec_train = get_metrics(y_train, y_predtr)
acc_test, sen_test, spec_test = get_metrics(y_test, y_predte)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predte)

print("\nTrain Accuracy: ", acc_train)
print("Train Sensitivity: ", sen_train)
print("Train Specificity: ", spec_train)

print("\nTest Accuracy: ", acc_test)
print("Test Sensitivity: ", sen_test)
print("Test Specificity: ", spec_test)