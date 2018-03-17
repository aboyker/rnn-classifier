import tensorflow as tf

class RNN_cell(object):

    """
    RNN cell object which takes 3 arguments for initialization.
    input_size = Input Vector size
    hidden_layer_size = Hidden layer size
    target_size = Output vector size

    """

    def __init__(self, input_size, hidden_layer_size, target_size, 
                 embedding_lookup=False, vocab_size=None, ini_embedding=None, input_tensor=None, fixed_embedding_lookup=False):

        #Initialization of given values

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size
        self.fixed_embedding_lookup = fixed_embedding_lookup
        # Weights for input and hidden tensor
        self.Wx = tf.Variable(tf.random_uniform([self.input_size, self.hidden_layer_size], -1.0, 1.0))
        
        self.Wr = tf.Variable(tf.random_uniform([self.input_size, self.hidden_layer_size], -1.0, 1.0))

        self.Wz = tf.Variable(tf.random_uniform([self.input_size, self.hidden_layer_size], -1.0, 1.0))

        self.br = tf.Variable(tf.truncated_normal([self.hidden_layer_size],mean=1))
        self.bz = tf.Variable(tf.truncated_normal([self.hidden_layer_size],mean=1))
        
        self.Wh = tf.Variable(tf.zeros([self.hidden_layer_size, self.hidden_layer_size]))

        
        #Weights for output layer
        self.Wo = tf.Variable(tf.truncated_normal([self.hidden_layer_size,self.target_size],mean=1,stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.target_size],mean=1,stddev=.01))
        
        if embedding_lookup:
            
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                
                self.input_x = tf.placeholder(tf.int32,
                                      shape=[None, None],
                                      name='inputs')
                
                if ini_embedding is not None:
                    
                    self.W_embedding = tf.Variable(
                        ini_embedding,
                        name="W_embedding", trainable=not self.fixed_embedding_lookup)
                    
                else:
                    
                    self.W_embedding = tf.Variable(
                        tf.random_uniform([vocab_size, self.input_size], -1.0, 1.0),
                        name="W_embedding", trainable= not self.fixed_embedding_lookup)
                    
                embedded_chars = tf.nn.embedding_lookup(self.W_embedding, self.input_x)
                self.processed_input = process_batch_input_for_RNN(embedded_chars)
                self.initial_hidden = embedded_chars[:, 0, :]
                self.initial_hidden = tf.matmul(
                        self.initial_hidden, tf.zeros([self.input_size, self.hidden_layer_size]))
        
        elif input_tensor is not None:
            
            self.input_x = input_tensor
            self.processed_input = process_batch_input_for_RNN(self.input_x)
          

            self.initial_hidden = self.input_x[:, 0, :]
            self.initial_hidden = tf.matmul(
                    self.initial_hidden, tf.zeros([self.input_size, self.hidden_layer_size]))
        
            
        else:
            
            # Placeholder for input vector with shape[batch, seq, embeddings]
            self.input_x = tf.placeholder(tf.float32,
                                      shape=[None, None, self.input_size],
                                      name='inputs')
            # Processing inputs to work with scan function
            self.processed_input = process_batch_input_for_RNN(self.input_x)
            '''
            Initial hidden state's shape is [1,self.hidden_layer_size]
            In First time stamp, we are doing dot product with weights to
            get the shape of [batch_size, self.hidden_layer_size].
            For this dot product tensorflow use broadcasting. But during
            Back propagation a low level error occurs.
            So to solve the problem it was needed to initialize initial
            hidden state of size [batch_size, self.hidden_layer_size].
            So here is a little hack !!!! Getting the same shaped
            initial hidden state of zeros.
            '''

            self.initial_hidden = self.input_x[:, 0, :]
            self.initial_hidden = tf.matmul(
                    self.initial_hidden, tf.zeros([self.input_size, self.hidden_layer_size]))
        
    #Function for GRU cell
    def Gru(self, previous_hidden_state, x):
        """
        GRU Equations
        
        """
        z= tf.sigmoid(tf.matmul(x,self.Wz)+ self.bz)
        r= tf.sigmoid(tf.matmul(x,self.Wr)+ self.br)
        
        h_= tf.tanh(tf.matmul(x,self.Wx) + tf.matmul(previous_hidden_state,self.Wh)*r)
                    
        
        current_hidden_state = tf.multiply((1-z),h_) + tf.multiply(previous_hidden_state,z)
        
        return current_hidden_state     
    
    # Function for getting all hidden state.
    def get_states(self):
        """
        Iterates through time/ sequence to get all hidden state
        
        """

        # Getting all hidden state throuh time
        all_hidden_states = tf.scan(self.Gru,
                                    self.processed_input,
                                    initializer=self.initial_hidden,
                                    name='states')

        return all_hidden_states

    # Function to get output from a hidden layer
    def get_output(self, hidden_state):
        
        """
        This function takes hidden state and returns output
        
        """
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)

        return output

    # Function for getting all output layers
    
    def get_outputs(self):
        """
        Iterating through hidden states to get outputs for all timestamp
        
        """
        all_hidden_states = self.get_states()

        all_outputs = tf.map_fn(self.get_output, all_hidden_states)

        return all_outputs
    
    

        
        
        

# Function to convert batch input data to use scan ops of tensorflow.
def process_batch_input_for_RNN(batch_input):
    """
    Process tensor of size [5,3,2] to [3,5,2]
    
    """
    batch_input_ = tf.transpose(batch_input, perm=[2, 0, 1])
    X = tf.transpose(batch_input_)

    return X