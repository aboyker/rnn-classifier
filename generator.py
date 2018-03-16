# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:44:32 2018

@author: Alexandre Boyker
"""
import glob
import codecs



class BatchGenerator(object):
    
    """
    
    Mini-batches generator. This class is designed to generate batches of input data points.
    Given an input directory, the generator will iterate over each sub-directory and open each file contained in those directories.
    The most important (positional) argument is `parser`. This is a python function that takes as input one line in a flat file (.txt, .csv, ... each line is
    supposed to be separated by `\n` token). This function will parse the line and return the desired reprentation of the input. 
    The `parser` function can also returns a tuple (representation, label), if the argument `exist_labels` is set to True.

    
    """
    def __init__(self, parser, exist_labels=True, data_path = "*.txt", header=True, train_w2v=False, 
                 batch_size = 20, one_pass=False, dictionnary_mapping=None, 
                 padding=None, padding_term=None, inverse_trick=True):
        """
        
        constructor for BatchGenerator class.
        
        positional arguments:
            
            -- parser: python function that takes as input one line of each file and returns a either a representation only
                        
                or a tuple (representation, label). The output may be fore example of the form: ([1,1,2,3], [1,0]) (tuple of lists, if labels, else list)
        
        keyword arguments:
            
            -- exist_labels: boolean, True if a label is expected, else False
            
            -- data_path: string, data path of the form `os.path.join('data','**','*.txt')`
            
                Here, `data` is the main data directory, which can possibly contain many sub-directories (`**`).
                `*.txt` means that the generator will only look for .txt extension.
                        
            -- header: boolean, True if the files contain headers. If this is the case, the first line is ignored
            
            -- train_w2v: boolean. The BatchGenerator class is designed to integrate with the Gensim implementation of Word2Vec
            
                (https://radimrehurek.com/gensim/models/word2vec.html).
                            
                            
            -- batch_size: int, number of samples in each batch
            
            -- one_pass: boolean, True if the generator iterates once over the data. If False, the number of iteration grows until infinity
            
            -- dictionnary_mapping: dictionary object. If not None, each element of each batch sample is mapped to the corresponding dictionary item
            
                example: batch = [[A, B], [C, D]], dictionnary_mapping = {'A':1, 'B':2, 'C':3, 'D':4}
                                
                In case of key error, `None` is returned.
                                
            -- padding: int, if not None, all samples are normalized so that each samples have the same length. This is useful in NLP
                            
                when working with sentences of different lengths.
                        
            -- padding_term: int, str, double. If a sample is smaller than the padding size, the padding term is added accordingly ()
            
            
            -- inverse_trick: boolean. If True, all samples are inverted (before padding). This trick is useful when training Recurrent neural networks, as the last
            
                words of a sentence tend to have more importance for the embedding, while in reality the first words tend to be more significant.
        """
        
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
     
    def _get_dict_item(self, key):
        
        """
        Returns the corresponding item of dictionary
        
        positional argument:
            
            -- key: key of the dictionary
        
        """
        
        try:
            return self.dictionnary_mapping[key]
        
        except KeyError:
            
            return None
            
    def _normalize_sequence(self, sample):
        
        """
        Returns a padded sample. 
        
        positional argument:
            
            -- sample: a list of objects
        
        """
        if len(sample)> self.padding:
            
            return sample[:self.padding]
        
        elif len(sample) < self.padding:
            
            for i in range(self.padding -len(sample)):
                
                sample.append(self.padding_term)
            
            return sample
        
        else:
            
            return sample
        
        
    def __iter__(self):
        
        """
        Iterates across the data and returns batches of samples 
        
        """
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
        
        
    