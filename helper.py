# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:48:01 2018

@author: bebxadvaboy
"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

def get_MNIST_data():
    
    data = pd.read_csv(os.path.join("data","train.csv"))
    X_train = (data.ix[:,1:].values).astype('float32') # all pixel values
    #X_train = X_train.reshape((X_train.shape[0], 28, 28))
    y_train = data.ix[:,0].values.astype('int32') # only labels i.e targets digits   
#    b = np.zeros((X_train.shape[0],10))
#    b[np.arange(X_train.shape[0]), y_train] = 1
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    X_train = X_train/255.0
    X_val = X_val/255.0
    Y_train = np.array([Y_train])
    Y_val = np.array([Y_val])
    train = pd.DataFrame(np.concatenate((Y_train.T, X_train), axis=1))
    test = pd.DataFrame(np.concatenate((Y_val.T, X_val), axis=1))
    test.to_csv('test.csv', index=False)    
    train.to_csv('train.csv', index=False)    

  
    

def reset_graph():
    
    if 'sess' in globals() and sess:
        sess.close()
        
    tf.reset_default_graph()