# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

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

df = pd.read_csv('Social_Network_Ads.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#classifier = SVC(kernel = 'linear', random_state = 0)
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_predtr = classifier.predict(X_train)
y_predte = classifier.predict(X_test)
acc_train, sen_train, spec_train = get_metrics(y_train, y_predtr)
acc_test, sen_test, spec_test = get_metrics(y_test, y_predte)

print("\nTrain Accuracy: ", acc_train)
print("Train Sensitivity: ", sen_train)
print("Train Specificity: ", spec_train)

print("\nTest Accuracy: ", acc_test)
print("Test Sensitivity: ", sen_test)
print("Test Specificity: ", spec_test)

plt.figure(figsize=(50,20))
plot_tree(classifier, filled=True, fontsize=10)
plt.savefig('foo.png')
plt.show()