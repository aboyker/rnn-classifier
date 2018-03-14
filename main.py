# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:54:50 2018

@author: Alexandre Boyker

"""

from w2v import Word2VecGenerator
from helper import get_MNIST_data, reset_graph
import matplotlib.pyplot as plt
from generator import BatchGenerator
import os
from parser2 import parse_mnist, parser_stsa
from rnn import RNN
from rnn_cell import RNN_cell
import numpy as np

w2v_path = os.path.join('data','stsa','**','*.txt')
train_path = os.path.join('data','stsa','train','*.txt')
validation_path = os.path.join('data','stsa','test','*.txt')

word2vec_epochs = 10

embedding_size = 300
padding= 49

learning_rate = 0.001

batch_size_train = 30
batch_size_valid = 300

embedding_lookup=True
hidden_layer_size = 25
n_epochs = 5


n_stacked_units = 1

attention = False
pre_trained = True

def main():
    

    reset_graph()
    
    w2v_generator = Word2VecGenerator(w2v_path, parser_stsa, embedding_size=embedding_size, batch_size=100,
                                      n_epochs=word2vec_epochs, pre_trained=pre_trained)
   
    dic, dic_ini = w2v_generator.get_mapping()
    print('len dic', len(dic))
    
    padding_term = len(dic)
    ini_embedding = np.zeros((len(dic)+1, embedding_size), dtype=np.float32)
    for key, ind in dic.items():
        
        try:
            ini_embedding[ind] = dic_ini[key]
            
        except IndexError:
            
            pass
        
    train_generator_param = {'parser':parser_stsa,'exist_labels':True, 'train_w2v':False, 
                           'data_path':train_path, 'header':True, 
                           'batch_size':batch_size_train, 'one_pass':False,
                           'dictionnary_mapping':dic, 'padding':padding, 'padding_term':padding_term}
        
    
    train_generator = BatchGenerator(**train_generator_param)
    
    validation_generator_param = {'parser':parser_stsa,'exist_labels':True, 'train_w2v':False, 
                           'data_path':validation_path, 'header':True, 
                           'batch_size':batch_size_valid, 'one_pass':True,
                           'dictionnary_mapping':dic, 'padding':padding, 'padding_term':padding_term}
        
    
    validation_generator = BatchGenerator(**validation_generator_param)
    
    rnn_param = {'RNN_cell':RNN_cell, 'learning_rate':learning_rate, 'n_epochs': n_epochs, 'target_size':2, 'input_size':embedding_size, 
                 'hidden_layer_size':hidden_layer_size, 'validation_steps':200, 
                 'vocab_size':len(dic)+1, 'attention':attention,
                 'ini_embedding':ini_embedding,'embedding_lookup':embedding_lookup, 'n_stacked_units':n_stacked_units}
    
    gru_net = RNN(**rnn_param)
    gru_net.fit_generator(train_generator, validation_generator)
    
    
if __name__ == '__main__':
    
    main()
