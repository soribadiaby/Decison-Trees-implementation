# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 14:58:11 2019

@author: Soriba
"""
import numpy as np
from sklearn.base import BaseEstimator

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


def ig(X, y, feature_idx, threshold, criterion='gini'):
    """
    Information gain. Function to maximize to get the best split
    """
    X=np.array(X)
    #we split the data according to the threshold
    cond = X[:,feature_idx]<threshold
    y_left = y[cond]
    y_right = y[~cond]
    #length of the origial data and length of each split
    l=len(y)
    ll=len(y_left)
    lr=len(y_right)
    
    if criterion=='gini':
        return gini(y)-(ll/l)*gini(y_left)-(lr/l)*gini(y_right)
    elif criterion=='entropy':
        return entropy(y)-(ll/l)*entropy(y_left)-(lr/l)*entropy(y_right)
    
     
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