# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:58:11 2019

@author: Soriba
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

def entropy(y):
    y=np.array(y)
    classes=np.unique(y)
    #probabilities for each class   
    p=[y[y==classes[k]].size/y.size for k in range(len(classes))]
    p=np.array(p)
    return -np.sum(p*np.log2(p))    


def gini(y):
    y=np.array(y)
    classes=np.unique(y) 
    p=[y[y==classes[k]].size/y.size for k in range(len(classes))]
    p=np.array(p)
    #gini impurity index
    return 1-np.sum(p**2) 


def ig(X,j,threshold,criterion='gini'):
    """
    Information gain. Function to maximize to get the best split
    """
    X=np.array(X)
    #we split the data according to the threshold
    Xleft=X[X[:,j]<threshold]
    Xright=X[X[:,j]>=threshold]
    #length of the origial data and length of each split
    l=X.shape[0]
    ll=Xleft.shape[0]
    lr=Xright.shape[0]
    
    if criterion=='gini':
        return gini(X)-(ll/l)*gini(Xleft)-(lr/l)*gini(Xright)
    elif criterion=='entropy':
        return entropy(X)-(ll/l)*entropy(Xleft)-(lr/l)*entropy(Xright)
    

#%% classes
        
class Node():
    
    def __init__(self, feature_idx=0, threshold=0, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right


class DecisionTreeCustom(BaseEstimator):
    def __init__(self, max_depth = None, min_samples_split = None,
                 criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
 
    
    def fit(self, X,y):
        ...
           
    def predict_proba(self, X):
        ...
        
    def predict(self, X):
        ...