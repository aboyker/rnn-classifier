# rnn-classifier

## Motivations

This is a TensorFlow implementation of a RNN classifier with GRU units. Everything is designed 'from scratches', that is we do not use the high level layers/units classes of TensorFlow so that we can keep control over every tensor operation.

This can serve as basic building blocks for more complex RNN projects. The default implementation is a standard RNN multilabel classifier, with gated recurrent units.

## Default dataset

As an example, we use the STSA dataset (Stanford movie review). This data set can be used for sentiment analysis (binary classification) 

## Requirements
TensorFlow 1.3.0

## Included features

### Word2Vec embedding

It is possible to either train a word2vec model or to use a pre-trained w2v model to initialize the embedding matrix.

### built-in generators

The network scales on large datasets thanks to the use of generators. It is possible to parse both very large file as well as a very large number of small files.

## Training on default dataset
Download the project and cd to project directory, then do

`python train.py`

on the command line.

## Training on other datasets

You have to define a parser function in the 'parser2.py' file. This parser is expected to parse each line of your documents (each line is assumed to be separated from the others with the standard '\n' token). The parser function takes as input one line of the document and returnsa list of tokens.

For instance, "There are many parrots in the park" ==> ['There', 'are', 'many', 'parrots, 'in', 'the', 'park']

The parser function can also return a tuple containing the parsed line with a label (preferably one-hot-encoded).

Ex:  "There are many parrots in the park" ==> (['There', 'are', 'many', 'parrots, 'in', 'the', 'park'], [0, 1] )


## Deployement/predictions on novel data

TODO: Fill up predict.py file

## Performance comparison with CNN for text classification

This model achieves  80.6% acccuracy on the STSA data set (on validation data), using the default parameters (performance can vary due to random initialization of weights). However, a CNN can achieve more than 95% accuracy on validation data on the same dataset (see my other repo [ cnn-documents-classification](https://github.com/aboyker/convnet-document-classification) )
