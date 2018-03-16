# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:57:36 2018

@author: Alexandre Boyker

"""

import gensim
from generator import BatchGenerator

class Word2VecGenerator(object):
    
    """
    Wrapper class to generate word2vec models, dictionary mappings {token: ids} and initial word2vec embedding 
    to train other models.
    
    """
    def __init__(self, data_path, parser, embedding_size=8, batch_size=10, 
                 n_epochs= 5, pre_trained=False, pre_trained_path='GoogleNews-vectors-negative300.bin'):
        
        """
        Constructor
        
        positional arguments:
            
            -- data_path: path of the form `os.path.join('data','**','*.txt')`
            
                The generator will iterate over all files in the `data` directory and open files with the relevant extension (`.txt`)
              
            -- parser: python function that takes as input one line of each file and returns a representation 
                        
                of the line as list. The output may be fore example of the form: [1,1,2,3] 
                        
        keyword arguments:
            
            -- embedding_size: int, embedding dimension for word2vec model. Remark, if using a pre-trained model, this argument is ignore.
            
            -- batch_size: int, number of samples in each batch for word2vec model training
            
            -- n_epochs: int,  number of epochs for word2vec model training
            
            -- pre_trained: boolean, True if we use a pre-trained model. In this case, all other arguments are ignored, except `pre_trained_path`
            
            -- pre_trained_path: str, path of the pre-trained model. Ignored if pre_trained is False
        
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.data_path = data_path
        self.parser = parser
        self.embedding_size = embedding_size
        self.pre_trained_path = pre_trained_path
        self.pre_trained = pre_trained
        
    def train(self):
        
        """
        Returns a trained word2vec model
        
        """
        
        generator_param = {'parser':self.parser,'exist_labels':False, 'train_w2v':True, 
                           'data_path':self.data_path, 'header':True, 
                           'batch_size':self.batch_size, 'one_pass':True}
        
        self.generator = BatchGenerator(**generator_param)
        
        model = gensim.models.Word2Vec(iter=1, size=self.embedding_size, window=5, min_count=5)  # an empty model, no training yet
        
        print("building vocabulary...")
        
        model.build_vocab(self.generator)  # can be a non-repeatable, 1-pass generator
        
        # we only consider words appearing in the input documents, this is useful when 
        # working with large pre-trained word2vec models, for memory issues.
        self.dic = {k:i.index for (k,i) in model.wv.vocab.items()}
        
        if self.pre_trained:
            
            # override model object if using a pre-trained model
            print("loading word2vec model...")
            model = gensim.models.KeyedVectors.load_word2vec_format(self.pre_trained_path, binary=True)  

        else:
            
            print("training word2vec model...")
            model.train(self.generator, total_examples=model.corpus_count, epochs=self.n_epochs)  # can be a non-repeatable, 1-pass generator

        return model
        
        
    def get_mapping(self):
        
        """
        Returns 2 dictionareis:
            
            -- dic_mapping maps each token to a unique index from 0 to size of vocabulary
            
            -- dic_ini maps each token to a its word2vec embedding. Useful to initialze other models with a w2v embedding
        
        """
        model = self.train()
        cnt = 0
        dic_mapping = {}
        dic_ini = {}
        
        for k in model.wv.vocab.keys():
            
            if k in self.dic.keys():
                
                dic_mapping[k] = cnt
                
                cnt += 1
                
                dic_ini[k] = model.wv[k]
                
                
        return dic_mapping, dic_ini
        

        