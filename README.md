# rnn-classifier

## Motivations

This is a TensorFlow implementation of a RNN classifier with GRU units. Everything is designed 'from scratches', that is we do not use the high level layers/units classes of TensorFlow so that we can keep control over every tensor operation.

## Default dataset

As an example, we use the STSA dataset (Stanford movie review). This data set can be used for sentiment analysis (binary classification) 

## Requirements
TensorFlow 1.3.0
## Training on default dataset
Download the project and cd to project directory, then do

`python train.py`

on the command line.

## Training on other datasets

You have to define a parser function in the 'parser2.py' file. This parser is expected to parse each line of your documents (each line is assumed to be separated from the others with the standard '\n' token). The parser function takes as input one line of the document and returnsa list of tokens.

For instance, "There are many parrots in the park" ==> ['There', 'are', 'many', 'parrots, 'in', 'the', 'park']

The parser function can also return a tuple containing the parsed line with a label (preferably one-hot-encoded).

Ex:  "There are many parrots in the park" ==> (['There', 'are', 'many', 'parrots, 'in', 'the', 'park'], [0, 1] )
