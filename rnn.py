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
    
    def __init__(self, RNN_cell,learning_rate=0.001, n_epochs = 50, target_size=10, input_size=28, hidden_layer_size=30, 
                 n_stacked_units =1,
                 validation_steps=300, vocab_size=None, embedding_lookup=False,
                 attention= False,ini_embedding=None):
        
        self.RNN_cell = RNN_cell
        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.target_size = target_size
        self.n_epochs = n_epochs
        self.validation_steps = validation_steps
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.ini_embedding = ini_embedding
        self.embedding_lookup = embedding_lookup
        self.n_stacked_units = n_stacked_units
        self.attention = attention
        
    
    def _build_graph(self):
        
        input_y = tf.placeholder(tf.float32, shape=[None, self.target_size], name='inputs')
    
        rnn = self.RNN_cell( self.input_size, self.hidden_layer_size, self.target_size, 
                            embedding_lookup=self.embedding_lookup, vocab_size=self.vocab_size,
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
        #Calculatio of correct prediction and accuracy
        predi = tf.argmax(output, 1)
        correct_prediction = tf.equal(tf.argmax(input_y, 1), predi)
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
        
        return input_x, input_y, train_step, cross_entropy, accuracy, predi, W_attention
    
    
    
    def fit_generator(self, train_generator, validation_generator=None):
        


        with tf.Session() as sess:

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
                    
                    print("{} iterations: {} out of {}  loss: {}  accuracy: {}".format(str(datetime.now()), 1+i, self.n_epochs, ce, acc))
                
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
                    
                if epochs_cnt ==self.n_epochs - 1: break

        
        