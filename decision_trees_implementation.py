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
    #probabilités pour chaque classe    
    p=[y[y==classes[k]].size/y.size for k in range(len(classes))]
    p=np.array(p)
    return -np.sum(p*np.log2(p))    

def gini(y):
    y=np.array(y)
    classes=np.unique(y) 
    p=[y[y==classes[k]].size/y.size for k in range(len(classes))]
    p=np.array(p)
    #indice d'impureté de gini
    return 1-np.sum(p**2) 

def variance(y):
    l=len(y)
    m=np.sum(y)/l
    return np.sum((y-m)**2)/l
    
def mad_median(y):
    #écart moyen par rapport à la médiane
    l=len(y)
    med=np.median(y)
    return np.sum(abs(y=med))/l

def ig(X,j,threshold,criterion='gini'): #fonction a maximiser pour obtenir la meilleure separation
    X=np.array(X)
    #on sépare le jeu de données initial en fonction du seuil et de la variable concernée
    Xleft=X[X[:,j]<threshold]
    Xright=X[X[:,j]>=threshold]
    #calcul du nombre d'entrées dans chacun des jeux de données
    l=X.shape[0]
    ll=Xleft.shape[0]
    lr=Xright.shape[0]
    
    if criterion=='gini':
        return gini(X)-(ll/l)*gini(Xleft)-(lr/l)*gini(Xright)
    elif criterion=='entropy':
        return entropy(X)-(ll/l)*entropy(Xleft)-(lr/l)*entropy(Xright)
    

#%% classes
        
class Node():
    
    def __init__(self, feature_idx=0, threshold=0, labels=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.labels = labels
        self.left = left
        self.right = right
        
        
