# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:54:50 2018

@author: Alexandre Boyker

"""

from w2v import Word2VecGenerator
from helper import reset_graph
from generator import BatchGenerator
import os
from parser2 import parser_stsa
from rnn import RNN
from rnn_cell import RNN_cell
from argparse import ArgumentParser



""" -------------------Parameters definition------------------- """

parser = ArgumentParser()

parser.add_argument("-bt", "--batch_size_train", dest="batch_size_train",
                    help="batch size for training, default=30", type=int, default=30)

parser.add_argument("-bv", "--batch_size_valid", dest="batch_size_valid",
                    help="batch size for validation data, default=300", type=int, default=300)

parser.add_argument("-lr", "--learning_rate", dest="learning_rate",
                    help="learning rate,  default=.001", type=float, default=.001)


parser.add_argument("-e", "--embedding_size",
                     dest="embedding_size", default=40, type=int,
                    help="the dimensions of the embedding for the input, default=40")

parser.add_argument("-es", "--hidden_layer_size",
                     dest="hidden_layer_size", default=25, type=int,
                    help="the dimensions of the embedding inside the RNN cells, default=25")

parser.add_argument("-n", "--num_classes",
                     dest="num_classes", default=2, type=int,
                    help="number of classes to predict, default=16")

parser.add_argument("-ev", "--evaluation_every",
                     dest="evaluation_every", default=200, type=int,
                    help="number of training steps required to perform new evaluation, default=100")

parser.add_argument("-ep", "--total_n_batches",
                     dest="total_n_batches", default=5000, type=int,
                    help="total number of batches to train the network, default=1000")

parser.add_argument("-nv", "--word2vec_epochs",
                     dest="word2vec_epochs", default=10, type=int,
                    help="number of epochs to train the word2vec model for initial embedding, default=10")

parser.add_argument("-v", "--voc",
                     dest="voc_path", default='voc',
                    help="path of vocabulary files, default='voc'")

parser.add_argument("-ms", "--padding",
                    dest="padding", default=50,type=int,
                    help="size of the input sentences, longer sentences are truncated, shorter are padded, default=50")

parser.add_argument("-fe", "--fixed_embedding_lookup",
                    dest="fixed_embedding_lookup", default=False,type=bool,
                    help="indicated whether or not the embedding matrix is trainable or fixed, True meaning fixed, default=False")

parser.add_argument("-su", "--n_stacked_units",
                    dest="n_stacked_units", default=2,type=int,
                    help="number of RNN units in each layer default=2")

parser.add_argument("-pt", "--pre_trained",
                    dest="pre_trained", default=False,type=bool,
                    help="whether or not to use a pre-trained word2vec model for initial embedding, default=False")

parser.add_argument("-pw", "--pre_trained_path",
                    dest="pre_trained_path", default='GoogleNews-vectors-negative300.bin',type=str,
                    help="path of the pre-trained w2v model if using one,")



parser.add_argument("-wp", "--w2v_path",
                    dest="w2v_path", default=os.path.join('data','stsa','**','*.txt'),type=str,
                    help="path to the directory containing the text documents to train the w2v model")

parser.add_argument("-tp", "--train_path",
                    dest="train_path", default=os.path.join('data','stsa','train','*.txt'),type=str,
                    help="path to the directory containing the text documents to train the RNN model")


parser.add_argument("-vp", "--validation_path",
                    dest="validation_path", default=os.path.join('data','stsa','test','*.txt'),type=str,
                    help="path to the directory containing the text documents to validate the RNN model")


parser.add_argument("-at", "--attention",
                    dest="attention", default=False,type=bool,
                    help="whether or not to use an attention mechanism, that is learning a linear combination of all output layers instead of just last layer, default=False")




args = parser.parse_args()

batch_size_train = args.batch_size_train
batch_size_valid = args.batch_size_valid
learning_rate = args.learning_rate
embedding_size = args.embedding_size
num_classes = args.num_classes
evaluation_every = args.evaluation_every
total_n_batches = args.total_n_batches
voc_path = args.voc_path
padding = args.padding
hidden_layer_size = args.hidden_layer_size
word2vec_epochs = args.word2vec_epochs
fixed_embedding_lookup = args.fixed_embedding_lookup
n_stacked_units = args.n_stacked_units
pre_trained = args.pre_trained
train_path = args.train_path
validation_path = args.validation_path
w2v_path = args.w2v_path
attention = args.attention
pre_trained_path = args.pre_trained_path


if pre_trained:
    
    embedding_size = 300

def train():
    

    reset_graph()
    
    w2v_generator = Word2VecGenerator(w2v_path, parser_stsa, embedding_size=embedding_size, batch_size=100,
                                      n_epochs=word2vec_epochs, pre_trained=pre_trained, pre_trained_path=pre_trained_path)
   
    dic, ini_embedding = w2v_generator.get_mapping()
    
    
    padding_term = len(dic)
    
        
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
    
    rnn_param = {'RNN_cell':RNN_cell, 'learning_rate':learning_rate, 'total_n_batches': total_n_batches, 'target_size':num_classes, 'input_size':embedding_size, 
                 'hidden_layer_size':hidden_layer_size, 'validation_steps':evaluation_every, 
                 'vocab_size':len(dic)+1, 'attention':attention,
                 'ini_embedding':ini_embedding,'fixed_embedding_lookup':fixed_embedding_lookup, 'n_stacked_units':n_stacked_units}
    
    gru_net = RNN(**rnn_param)
    gru_net.fit_generator(train_generator, validation_generator)
    
    
if __name__ == '__main__':
    
    train()
