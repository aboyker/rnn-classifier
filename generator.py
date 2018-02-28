# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:44:32 2018

@author: Alexandre Boyker
"""
import glob
import codecs
from collections import deque
import numpy as np
import os


class BatchGenerator(object):
    
    def __init__(self, parser, exist_labels=True, data_path = "*.txt", header=True, train_w2v=False, 
                 batch_size = 20, one_pass=False, dictionnary_mapping=None, 
                 padding=None, padding_term=None, inverse_trick=True):
        
        self.parser = parser
        self.data_path = data_path
        self.batch_size = batch_size
        self.header = header
        self.one_pass = one_pass
        self.exist_labels = exist_labels
        self.train_w2v = train_w2v 
        self.dictionnary_mapping = dictionnary_mapping
        self.padding = padding
        self.padding_term = padding_term
        self.inverse_trick= inverse_trick
     
    def _get_dict_item(self, item):
        
        try:
            return self.dictionnary_mapping[item]
        
        except KeyError:
            
            pass
            
    def _normalize_sequence(self, batch):
        
        if len(batch)> self.padding:
            
            return batch[:self.padding]
        
        elif len(batch) < self.padding:
            
            for i in range(self.padding -len(batch)):
                
                batch.append(self.padding_term)
            
            return batch
        
        else:
            
            return batch
        
        
    def __iter__(self):
        
        while True:
                
            for filename in glob.iglob(self.data_path, recursive=True):
                        
                f = codecs.open(filename, 'r', encoding = "utf8", errors = 'ignore')
                batch_input = list()
                batch_label = list()
                
                for i,line in enumerate(f):
                    
                    if i==0 and self.header: continue
                
                    parsed_line = self.parser(line)
                    
                    if self.dictionnary_mapping is not None:
                        
                        
                        bch = [ self._get_dict_item(item) for item in parsed_line[0] if self._get_dict_item(item) is not None ]
                        
                        if self.inverse_trick:
                            
                            bch = list(reversed(bch))
                            
                        if self.padding_term is not None:
                            bch = self._normalize_sequence(bch)

                        batch_input.append(bch)
                  
                    elif self.train_w2v:
                        
                        batch_input += parsed_line[0]
                        
                    else:
                        
                        batch_input.append(parsed_line[0])
                            
                    
                    batch_label.append(parsed_line[1])
                    
                    
                    
                    
                    
                            
                    if len(batch_label)==self.batch_size:
                        
                        if self.exist_labels:
                            
                            yield batch_input, batch_label
                            
                        else:
                            
                            yield (batch_input)

                        batch_input = list()
                        
                        batch_label = list()
                        
            if self.one_pass: break
        
        
    