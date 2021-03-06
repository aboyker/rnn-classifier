# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:57:36 2018

@author: Alexandre Boyker

"""
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import accuracy_score
import numpy as np
from rnn_cell import process_batch_input_for_RNN

class RNN(object):
    
    
    """
    Implementation of a simple Recurrent Neural Network for sentence classification.
    
               
                                                                          / label 1
    ->   ->  ->  ->   ->    ->   ->    EMBEDDING! -> Classification layer - label 2
                                                                          \  
                                                                            label q
    
    O    O       O ... O    O    O      ^
    |    |       |     |    |    |      |
    O    O       O ... O    O    O
    .    .       .     .    .    .
    .    .       .     .    .    .      ^
    .    .       .     .    .    .      |    upward direction
    O    O       O ... O    O    O
    |    |       |     |    |    |      ^
    O    O       O ... O    O    O      |      O = RNN cell
    
    ^    ^       ^     ^    ^    ^
    The  parrot  is... and very smart
    
    
    This class can readily be used as a document classifier, or can serve as a basis to build more elaborate models.
    
    
    """
    
    def __init__(self, RNN_cell,learning_rate=0.001, total_n_batches = 50, target_size=10, input_size=28, hidden_layer_size=30, 
                 n_stacked_units =1,
                 validation_steps=300, vocab_size=None, fixed_embedding_lookup=False,
                 attention= False,ini_embedding=None):
        
        """
        
        Constructor for RNN class
        
        positional arguments:
            
            -- RNN_cell: a RNN_cell object, representing the type of RNN cells (LSTM, GRU, ...)
            
        keyword arguments: 
            
            -- learning_rate: double, the learning rate
            
            -- n_epochs: 
                
            -- target_size: int, number of labels
            
            -- input_size: int, embedding dimension for the embedding lookup matrix
            
            -- hidden_layer_size: int, embedding dimension in each RNN cell
            
            -- n_stacked_units: int, number of stacked units (depth of the network)
            
            -- validation_steps: int, number of training batch required before model validation
            
            -- vocab_size: int, number of words in vocabulary, dimension of embedding lookup matrix
            
            -- fixed_embedding_lookup: boolean, if True, the embedding lookup matrix is fixed and will not be trained
            
            -- ini_embedding: numpy ndarray of size (vocab_size * input_size), initial value for embedding lookup matrix
        
        """
        self.RNN_cell = RNN_cell
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.target_size = target_size
        self.total_n_batches = total_n_batches
        self.validation_steps = validation_steps
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.ini_embedding = ini_embedding
        self.fixed_embedding_lookup = fixed_embedding_lookup
        self.n_stacked_units = n_stacked_units
        self.attention = attention
        
    
    def _build_graph(self):
        
        input_y = tf.placeholder(tf.float32, shape=[None, self.target_size], name='inputs')
    
        rnn = self.RNN_cell( self.input_size, self.hidden_layer_size, self.target_size, 
                            embedding_lookup=True, vocab_size=self.vocab_size,
                            ini_embedding=self.ini_embedding)
        input_x = rnn.input_x
        outputs = rnn.get_outputs()
        stacked_input= rnn.get_states()
        
        for layer_index in range(1,self.n_stacked_units):
            
        
            
            stacked_input = process_batch_input_for_RNN(stacked_input)
        
            rnn = self.RNN_cell( self.hidden_layer_size, self.hidden_layer_size, self.target_size, 
                            embedding_lookup=False, vocab_size=self.vocab_size,
                            ini_embedding=None, input_tensor=stacked_input)
        
            outputs = rnn.get_outputs()
            stacked_input = rnn.get_states()
        
        if self.attention:
            
            W_attention = tf.Variable(tf.truncated_normal([self.hidden_layer_size],mean=1))
            W_attention = tf.divide(tf.abs(W_attention), tf.norm(W_attention, ord=1))
            last_output = tf.multiply(stacked_input, W_attention)
            last_output = tf.map_fn(rnn.get_output, last_output)
            last_output = last_output[-1]
            
        else:
            
            W_attention = tf.Variable(tf.truncated_normal([self.hidden_layer_size],mean=1))
            W_attention = tf.divide(tf.abs(W_attention), tf.norm(W_attention, ord=1))
            last_output = outputs[-1]
        
        

        output = tf.nn.softmax(last_output)
        cross_entropy = -tf.reduce_sum(input_y * tf.log(output))
        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cross_entropy)
        #Calculation of correct prediction and accuracy
        predi = tf.argmax(output, 1)
        correct_prediction = tf.equal(tf.argmax(input_y, 1), predi)
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
        
        return input_x, input_y, train_step, cross_entropy, accuracy, predi, W_attention
    
    
    def _save_best_model(self, validation_accuracy_list):
        
        last_val = validation_accuracy_list[-1]
        
        
        if  len(validation_accuracy_list) <1:
        
            return
        
        elif len(validation_accuracy_list) ==1:
            
            print("save first model")
            return
        
            
        
        elif ((len(validation_accuracy_list) >1) and  (last_val > np.max(np.array(validation_accuracy_list[:-1])))) :
            
            print("saved best model, with accuracy {}".format(last_val))
        
    def fit_generator(self, train_generator, validation_generator=None):
        


        with tf.Session() as sess:
            
            validation_accuracy_list = []
            
            input_x, input_y, train_step, cross_entropy, accuracy, predi , last_output= self._build_graph()
            
            # init and sess.run(init) should be stuck together !!!
            init = tf.global_variables_initializer()
            sess.run(init)        

            saver = tf.train.Saver()
            
            print('aln')
            epochs_cnt = 0
            
            for i, batch_train in enumerate(train_generator):
            
                if i ==0: epochs_cnt+=1
                
                x_batch_train, y_batch_train = batch_train
                
                _, ce, acc, last_output_=sess.run([train_step, cross_entropy, accuracy, last_output],feed_dict={input_x:x_batch_train, input_y:y_batch_train})
                
                if i%10 ==0:
                    
                    print("{} iterations: {} out of {}  loss: {}  accuracy: {}".format(str(datetime.now()), 1+i, self.total_n_batches, ce, acc))
                
                if (i != 0) and (i % self.validation_steps == 0) and (validation_generator is not None):
                    prediction_val_list = []
                    ground_truth_val = []
                    
                    print('\n')
                    print('Validation')
                    
                    for j, batch_val in enumerate(validation_generator):
                        
                        x_batch_val, y_batch_val = batch_val
                       
                        predi_val, ce, acc = sess.run([predi, cross_entropy, accuracy], feed_dict={input_x:x_batch_val, input_y:y_batch_val})
                        prediction_val_list += list(predi_val)
                        ground_truth_val += list(np.argmax(y_batch_val, 1))
                        print("{} validation iteration: {} loss: {}  accuracy: {}".format(str(datetime.now()), 1+j, ce, acc))
                        
                    val_accuracy = accuracy_score(ground_truth_val, prediction_val_list)
                    print("{}  global accuracy on validation data:  {}".format(str(datetime.now()), val_accuracy))
                    print('\n')
                    validation_accuracy_list.append(val_accuracy)
                    self._save_best_model(validation_accuracy_list)
                if i >= self.total_n_batches : break

        
        