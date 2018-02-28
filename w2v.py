# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:57:36 2018

@author: Alexandre Boyker

"""

import gensim
from generator import BatchGenerator

class Word2VecGenerator(object):
    
    def __init__(self, data_path, parser, embedding_size=8, batch_size=10, n_epochs= 5):
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.data_path = data_path
        self.parser = parser
        self.embedding_size = embedding_size
    
    def train(self):
        
        generator_param = {'parser':self.parser,'exist_labels':False, 'train_w2v':True, 
                           'data_path':self.data_path, 'header':True, 
                           'batch_size':self.batch_size, 'one_pass':True}
        
        self.generator = BatchGenerator(**generator_param)
        
        
        model = gensim.models.Word2Vec(iter=1, size=self.embedding_size, window=5, min_count=5)  # an empty model, no training yet
        print("building vocabulary...")
        model.build_vocab(self.generator)  # can be a non-repeatable, 1-pass generator
        print("training word2vec model...")
        model.train(self.generator, total_examples=model.corpus_count, epochs=self.n_epochs)  # can be a non-repeatable, 1-pass generator
        return model
        
        
        

        