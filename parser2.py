# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 17:07:25 2018

@author: Alexandre Boyker

"""
import numpy as np


    

def parse_mnist(line):
    
    """
    Returns a parsed line for the MNIST data set
    
    """
    
    line = line.splitlines()[0].split(',')

    line = list(map(float, line))
    label = np.zeros(10)
    label[int(line[0])] = 1
    
    return (np.array(line[1:])).reshape((28, 28)), label


def parser_stsa(line):
    
    """
    Returns a parsed line for the stsa data set
    
    """
    
    line = line.split()
    label = np.zeros(2)
    label[int(line[0])] = 1
    
    return line[1:], label
    
    