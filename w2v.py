# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:57:36 2018

@author: Alexandre Boyker

"""

import gensim
from generator import BatchGenerator

class Word2VecGenerator(object):
    
    def __init__(self, data_path, parser, embedding_size=8, batch_size=10, 
                 n_epochs= 5, pre_trained=False, pre_trained_path='GoogleNews-vectors-negative300.bin'):
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.data_path = data_path
        self.parser = parser
        self.embedding_size = embedding_size
        self.pre_trained_path = pre_trained_path
        self.pre_trained = pre_trained
        
    def train(self):
        
        generator_param = {'parser':self.parser,'exist_labels':False, 'train_w2v':True, 
                           'data_path':self.data_path, 'header':True, 
                           'batch_size':self.batch_size, 'one_pass':True}
        
        self.generator = BatchGenerator(**generator_param)
        
        model = gensim.models.Word2Vec(iter=1, size=self.embedding_size, window=5, min_count=5)  # an empty model, no training yet
        print("building vocabulary...")
        model.build_vocab(self.generator)  # can be a non-repeatable, 1-pass generator
        self.dic = {k:i.index for (k,i) in model.wv.vocab.items()}
        
        if self.pre_trained:
            print("loading word2vec model...")
            model = gensim.models.KeyedVectors.load_word2vec_format(self.pre_trained_path, binary=True)  

        else:
            
            
            print("training word2vec model...")
            model.train(self.generator, total_examples=model.corpus_count, epochs=self.n_epochs)  # can be a non-repeatable, 1-pass generator
        
        
        
        return model
        
        
    def get_mapping(self):
        
        model = self.train()
        cnt = 0
        dic_pre_trained = {}
        dic_ini = {}
        for k in model.wv.vocab.keys():
            
            if k in self.dic.keys():
                
                dic_pre_trained[k] = cnt
                cnt += 1
                
                dic_ini[k] = model.wv[k]
                
                
        return dic_pre_trained, dic_ini
        

        