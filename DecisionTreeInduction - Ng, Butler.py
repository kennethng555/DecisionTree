# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 17:02:28 2019

@author: Kenneth Ng
@email: kenng7183@gmail.com
@github: kennethng555
"""

import os
import psutil
import time

import pandas as pd
import numpy as np
import math
from statistics import mode
from scipy import stats
import seaborn as sns

process = psutil.Process(os.getpid())
start_time = time.time()

class Node(object):
    def __init__(self, criterion=None, attr=None, parent=None, maj_class=None):
        self.parent = parent
        self.criterion = criterion
        self.attr = attr
        self.children = []
        self.maj_class = maj_class
        
    def get_parent(self):
        return self.parent
    
    def set_attr(self, new_attr):
        self.attr = new_attr
        
    def set_criterion(self, new_criterion):
        self.criterion = new_criterion
        
    def set_parent(self, new_parent):
        self.parent = new_parent
        
    def add_child(self, node):
        self.children.append(node)

class Tree(object):
    def __init__(self, root=None):
        self.root = root
        self.head = root

def generate(df, attr_list, discrete, attr_sel_method):
    N = Node()
    if len(np.unique(df.iloc[:,-1])) == 1:
        maj = stats.mode(df.iloc[:,-1])[0][0]              # majority class labels
        N.maj_class = maj
        return N                                           #as leaf node
    if len(attr_list) == 0:
        if len(stats.mode(df.iloc[:,-1])[0]) > 0:
            maj = stats.mode(df.iloc[:,-1])[0][0]          # majority class labels
        else:
            return N
        N.maj_class = maj
        return N                                           # as leaf node labelled with majority class in dataset
    if attr_sel_method == 'gini':
        attr, pt = gini_index(df, attr_list, discrete)     # attribute_selection_method(D, attribute_list) to find the best splitting criterion
    elif attr_sel_method == 'gain_ratio':
        attr, pt = gain_ratio(df, attr_list, discrete)
    else:
        attr, pt = info_gain(df, attr_list, discrete)
    N.set_attr(attr)                                       # label N with splitting criterion
    N.set_criterion(pt)
    
    attr_list = attr_list[attr_list != attr]               #    attribute_list <- attribute_list - splitting_attribute //remove splitting_attribute
    
    if type(pt)==np.ndarray:                               # if discrete
        for criteria in pt:                                # for each outcome of j of splitting criterion //partition the tuples and grow subtrees for each partition
            Dj = df[df[attr] == criteria]                  #    let Dj be the set of data tuples in D satisfying outcome j //a partition
            if len(Dj) == 0:                               #    if Dj is empty then
                maj = stats.mode(df[:,-1])
                N.add_child(Node(maj_class= maj))          #        attach a leaf labelled with the majority class in D to node N
            else:                                          #    else attach the node returned by Generate_decision_tree(Dj, attribute_list) to node N
                N.add_child(generate(Dj, attr_list, discrete, attr_sel_method))
    else:                                                  # if continuous
        Dj = df[df[attr] > pt]
        if len(Dj) == 0:
            Dj = df[df[attr] <= pt]
            maj = stats.mode(Dj.iloc[:,-1])
            N.add_child(Node(maj_class= maj))
        else:
            N.add_child(generate(Dj, attr_list, discrete, attr_sel_method))
        
        Dj = df[df[attr] <= pt]
        if len(Dj) == 0:
            maj = stats.mode(Dj.iloc[:,-1])
            N.add_child(Node(maj_class= maj))
        else:
            N.add_child(generate(Dj, attr_list, discrete, attr_sel_method))
    return N

def info_gain(df, attr_list, discrete):
    info_g = dict()
    split = dict()
    size = len(df)
    pos = df[df.iloc[:,-1]==1]
    pos_t = len(pos)
    neg_t = len(df)-len(pos)
    info_d = -(pos_t/size * math.log(pos_t/size, 2) + neg_t/size * math.log(neg_t/size, 2))
    for a in attr_list:
        e = []
        if a in discrete:
            for u in np.unique(df[a]):
                posf = len(pos[pos[a] == u])
                negf = len(df[df[a] == u]) - posf
                tot = posf + negf
                if posf == 0 or negf == 0:
                    temp = 0
                else:
                    temp = -tot/size * (posf/tot * math.log(posf/tot, 2) + negf/tot * math.log(negf/tot, 2))
                e.append(temp)
            info_g[a] = info_d-sum(e)
            split[a] = np.unique(df[a])
        else:
            un = np.unique(df[a])
            if len(un) > 1:
                midpts = (un[:-1] + un[1:]) / 2
                for midpt in midpts:
                    reff = df[df[a] > midpt]
                    refg = df[df[a] < midpt]
                    f = pos[pos[a] > midpt]
                    g = pos[pos[a] < midpt]
                    posf = len(f)
                    negf = len(reff) - len(f)
                    totf = len(reff)
                    posg = len(g)
                    negg = len(refg) - len(g)
                    totg = len(refg)
                    if posf == 0 or negf == 0:
                        entropyf = 0
                    else:
                        entropyf = -totf/size * (posf/totf * math.log(posf/totf, 2) + negf/totf * math.log(negf/totf, 2))
                        
                    if posg == 0 or negg == 0:
                        entropyg = 0
                    else:
                        entropyg = -totg/size * (posg/totg * math.log(posg/totg, 2) + negg/totg * math.log(negg/totg, 2))
                    
                    if str(a) not in info_g.keys() or info_d - entropyf - entropyg > info_g[a]:
                        info_g[a] = info_d - entropyf - entropyg
                        split[a] = midpt
            else:
                info_g[a] = 0
                split[a] = 0
    
    max_value = max(info_g.values())
    for key in info_g.keys():
        if info_g[key] == max_value:
            return key, split[key]
    return 0

def gini_index(df, attr_list, discrete):
    gini = dict()
    split = dict()
    size = len(df)
    pos = df[df.iloc[:,-1]==1]
    neg = df[df.iloc[:,-1]==0]
    for a in attr_list:
        if a in discrete:
            un = np.unique(df[a])
            posf = len(pos[pos[a] == un[0]])
            negf = len(pos[pos[a] != un[0]])
            posg = len(neg[neg[a] == un[0]])
            negg = len(neg[pos[a] != un[0]])
            
            gini[a] = len(pos)/size * (1 - (posf/len(pos))**2 - (negf/len(pos))**2) + len(neg)/size * (1 - (posg/len(neg))**2 - (negg/len(neg))**2)
            split[a] = np.unique(df[a])
        else:
            un = np.unique(df[a])
            
            #un = un.astype(int)
            if len(un) > 1:
                midpts = (un[:-1] + un[1:]) / 2
                #midpts = (un[:-1] + un[1:]) / 2
                for midpt in midpts:
                    reff = df[df[a] > midpt]
                    refg = df[df[a] < midpt]
                    f = pos[pos[a] > midpt]
                    g = pos[pos[a] < midpt]
                    posf = len(f)
                    negf = len(reff) - len(f)
                    posg = len(g)
                    negg = len(refg) - len(g)
                    
                    if posf == 0 or negf == 0:
                        ginipart1 = 0
                    else:
                        ginipart1 = len(pos)/size * (1 - (posf/len(pos))**2 - (negf/len(pos))**2)
    
                    if posg == 0 or negg == 0:
                        ginipart2 = 0
                    else:
                        ginipart2 = len(neg)/size * (1 - (posg/len(neg))**2 - (negg/len(neg))**2)
                    
                    temp = ginipart1 + ginipart2
                    if str(a) not in gini.keys() or temp > gini[a]:
                        gini[a] = temp
                        split[a] = midpt
            else:
                gini[a] = 0
                split[a] = 0
    
    max_value = max(gini.values())        
    for key in gini.keys():
        if gini[key] == max_value:
            return key, split[key]
    return 0

def gain_ratio(df, attr_list, discrete):
    gain_r = dict()
    split = dict()
    size = len(df)
    pos = df[df.iloc[:,-1]==1]
    pos_t = len(pos)
    neg_t = len(df)-len(pos)
    info_d = -(pos_t/len(df) * math.log(pos_t/len(df)) + neg_t/len(df) * math.log(neg_t/len(df)))
    for a in attr_list:
        e = []
        counts = []
        if a in discrete:
            for u in np.unique(df[a]):
                posf = len(pos[pos[a] == u])
                negf = len(df[df[a] == u]) - posf
                tot = posf + negf
                if posf == 0:
                    temp = -tot/size * (negf/tot * math.log(negf/tot, 2))
                elif negf == 0:
                    temp = -tot/size * (posf/tot * math.log(posf/tot, 2))
                else:
                    temp = -tot/size * (posf/tot * math.log(posf/tot, 2) + negf/tot * math.log(negf/tot, 2))
                e.append(temp)
                counts.append(tot)
            info_gain = (info_d-sum(e)) / -sum(counts/len(df) * math.log(counts/len(df), 2))
            split_info = -sum(counts/size * math.log(counts/size, 2))
            gain_r[a] = info_gain / split_info
            split[a] = np.unique(df[a])
        else:
            un = np.unique(df[a])
            if len(un) > 1:
                midpts = (un[:-1] + un[1:]) / 2
                for midpt in midpts:
                    reff = df[df[a] > midpt]
                    refg = df[df[a] < midpt]
                    f = pos[pos[a] > midpt]
                    g = pos[pos[a] < midpt]
                    posf = len(f)
                    negf = len(reff) - len(f)
                    totf = len(reff)
                    posg = len(g)
                    negg = len(refg) - len(g)
                    totg = len(refg)
                    if posf == 0 or negf == 0:
                        entropyf = 0
                    else:
                        entropyf = -totf/size * (posf/totf * math.log(posf/totf, 2) + negf/totf * math.log(negf/totf, 2))
                        
                    if posg == 0 or negg == 0:
                        entropyg = 0
                    else:
                        entropyg = -totg/size * (posg/totg * math.log(posg/totg, 2) + negg/totg * math.log(negg/totg, 2))
                    
                    temp_gain = info_d - entropyf - entropyg
                    split_info = -(totf/size * math.log(totf/size, 2) + totg/size * math.log(totg/size, 2))
                    temp_ratio = temp_gain / split_info
                    if str(a) not in gain_r.keys() or temp_ratio > gain_r[a]:
                        gain_r[a] = temp_ratio
                        split[a] = midpt
            else:
                gain_r[a] = 0
                split[a] = 0
    
    max_value = max(gain_r.values())        
    for key in gain_r.keys():
        if gain_r[key] == max_value:
            return key, split[key]
    return 0

def predict(root, X, discrete):
    y_pred = []
    for i in range(len(X)):
        head = root
        while head.children != []:
            if head.attr in discrete:
                for j in range(len(head.criterion)):
                    if head.criterion[j] == X[head.attr][i]:
                        head = head.children[j]
                        break
            else:
                if X[head.attr][i] > head.criterion:
                    head = head.children[0]
                else:
                    head = head.children[1]
        y_pred.append(head.maj_class)
    return y_pred


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
            
    if fn + tp != 0:
        sen = tp / (fn + tp)
    else:
        sen = 'Inf'
    if tn + fp != 0:
        spec = tn / (tn + fp)
    else:
        spec = 'Inf'
    acc = (tp + tn) / len(y)
    return acc, sen, spec

file = "E:/ESE589/Project 3/Data/Matlab/credit approval.GINI.csv"

print('input file: ', file)
df = pd.read_csv(file)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2, random_state=0)
train = train.reset_index(drop = True)
test = test.reset_index(drop = True)
#discrete = df.columns[[0]]
discrete = []
print('Generating D-Tree...')
root = generate(train, train.columns[:-1], discrete, 'info_gain')
print('D-Tree generated. Beginninng testing...')
dectree = Tree(root)
y_trainr = predict(root, train.iloc[:,:-1], discrete)
print('.')
y_testr = predict(root, test.iloc[:,:-1], discrete)
print('Testing complete. Generating metrics...')
acc_train, sen_train, spec_train = get_metrics(train.iloc[:,-1], y_trainr)
acc_test, sen_test, spec_test = get_metrics(test.iloc[:,-1], y_testr)

print("\nTrain Accuracy: ", acc_train)
print("Train Sensitivity: ", sen_train)
print("Train Specificity: ", spec_train)

print("\nTest Accuracy: ", acc_test)
print("Test Sensitivity: ", sen_test)
print("Test Specificity: ", spec_test)

print("\n--- %s seconds ---" % (time.time() - start_time))
print("Total Bytes Used: ", end = '')
print(process.memory_info().rss)