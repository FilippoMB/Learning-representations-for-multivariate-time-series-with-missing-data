import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, BasicRNNCell, GRUCell, LSTMStateTuple, MultiRNNCell
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core

class s2sModel():
    
    def __init__(self,config):
        self.cell_type = config['cell_type']
        self.num_layers = config['num_layers']
        self.hidden_units = config['hidden_units']
        self.input_dim = config['input_dim']
        self.bidirect = config['bidirect']
        self.max_gradient_norm = config['max_gradient_norm']
        self.learning_rate = config['learning_rate']
        self.EOS = 0
        self.PAD = 0
        
        self._make_graph()
        
        
    def _make_graph(self):    
        
        self._init_placeholders()
        self._init_cells()
        
        # ----- ENCODER -----
        if self.bidirect == False:
            self._init_simple_encoder()
        else:
            self._init_bidirectional_encoder()
            
        # ----- DECODER -----
#        self._init_decoder()
        self._init_decoder_TF()
        
        # ----- LOSS -----
        self._init_loss()
                   
    
    def _init_placeholders(self):        
        # Everything is time-major
        self.encoder_inputs = tf.placeholder(shape=(None, None, self.input_dim), dtype=tf.float32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_outputs = tf.placeholder(shape=(None, None, self.input_dim), dtype=tf.float32, name='decoder_outputs')        

        
    def _init_cells(self):
        
        if self.bidirect:
            self.decoder_units = self.hidden_units*2
        else:
            self.decoder_units = self.hidden_units
            
        if self.cell_type == 'LSTM':
            self.encoder_fw_cell= MultiRNNCell([LSTMCell(self.hidden_units) for _ in range(self.num_layers)])
            self.encoder_bw_cell= MultiRNNCell([LSTMCell(self.hidden_units) for _ in range(self.num_layers)])
            self.decoder_cell= MultiRNNCell([LSTMCell(self.decoder_units) for _ in range(self.num_layers)])
        elif self.cell_type == 'GRU': 
            self.encoder_fw_cell= MultiRNNCell([GRUCell(self.hidden_units) for _ in range(self.num_layers)])
            self.encoder_bw_cell= MultiRNNCell([GRUCell(self.hidden_units) for _ in range(self.num_layers)])
            self.decoder_cell= MultiRNNCell([GRUCell(self.decoder_units) for _ in range(self.num_layers)])
        elif self.cell_type == 'RNN': 
            self.encoder_fw_cell= MultiRNNCell([BasicRNNCell(self.hidden_units) for _ in range(self.num_layers)])
            self.encoder_bw_cell= MultiRNNCell([BasicRNNCell(self.hidden_units) for _ in range(self.num_layers)])
            self.decoder_cell= MultiRNNCell([BasicRNNCell(self.decoder_units) for _ in range(self.num_layers)])
        else:
            raise        
        
        
    def _init_simple_encoder(self):  
        with tf.variable_scope("Encoder"): 
   
            (_, encoder_states) = (tf.nn.dynamic_rnn(
                    self.encoder_fw_cell,
                    inputs= self.encoder_inputs,
                    sequence_length=self.encoder_inputs_length,
                    time_major=True,
                    dtype=tf.float32))
            
            self.encoder_state = encoder_states          
            
            
    def _init_bidirectional_encoder(self):  
        with tf.variable_scope("Encoder"): 
                        
            # not retrieving outputs (only for attention)
            ((_, _), (encoder_fw_state, encoder_bw_state)) = (tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.encoder_fw_cell,
                    cell_bw=self.encoder_bw_cell,
                    inputs=self.encoder_inputs,
                    sequence_length=self.encoder_inputs_length,
                    time_major=True,
                    dtype=tf.float32))
            
            # concatenate the states of fw and bw cells
            if isinstance(encoder_fw_state[-1], LSTMStateTuple):   
                self.encoder_state = tuple(LSTMStateTuple(c=tf.concat((encoder_fw_state[i].c, encoder_bw_state[i].c), 1), 
                                                          h=tf.concat((encoder_fw_state[i].h, encoder_bw_state[i].h), 1))
                                                          for i in range(self.num_layers))                    
            elif isinstance(encoder_fw_state[-1], tf.Tensor):
                self.encoder_state = tuple(tf.concat((encoder_fw_state[i], encoder_bw_state[i]), 1) for i in range(self.num_layers))
    
    
    # Native implementation with training on inferenced outputs            
    def _init_decoder(self):      
        with tf.variable_scope("Decoder"):    
            decoder_lengths = self.encoder_inputs_length +  1 # extra length for the leading <EOS> token in decoder inputs
                    
            # outputs weights. need to be specified explicitly (no dense layer) as they are used multiple times 
            # TODO: try also glorot_normal
            W = tf.get_variable("W_dec", shape=[self.decoder_units, self.input_dim], initializer=tf.contrib.keras.initializers.glorot_uniform(), dtype=tf.float32)
            b = tf.get_variable("b_dec", shape=[self.input_dim], initializer=tf.zeros_initializer(), dtype=tf.float32)
            
            # initial decoder input and pad slice (not used)
            _, d_batch_size, _ = tf.unstack(tf.shape(self.encoder_inputs))
            self.EOS_slice = tf.ones([1,d_batch_size,self.input_dim], dtype=tf.float32, name='EOS_slice')*self.EOS
            PAD_time_slice = tf.ones([d_batch_size, self.input_dim], dtype=tf.float32, name='PAD_slice')*self.PAD
                        
            # loop function (initial step)
            def loop_fn_initial():
                initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
                initial_input = self.EOS_slice[0,:,:]
                initial_cell_state = self.encoder_state
                initial_cell_output = None
                initial_loop_state = None  # we don't need to pass any additional information -> loop_state ALWAYS None
                return (initial_elements_finished,
                        initial_input,
                        initial_cell_state,
                        initial_cell_output,
                        initial_loop_state)
            
            # loop function (transition step)
            def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
            
                def get_next_input():
                    next_input = tf.add(tf.matmul(previous_output, W), b) # previous output of the DECODER
                    return next_input
                
                elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size,]
                                                              # defining if corresponding sequence has ended
            
                finished = tf.reduce_all(elements_finished) # -> boolean scalar (logical AND between the elements of the input tensor)
                
                # the next decoder input is 0 (PAD) if the maximum number of decoder outputs have been produced
                _next_input = tf.cond(finished, lambda: PAD_time_slice, get_next_input) 
                
                state = previous_state # state is just passed through
                output = previous_output
                loop_state = None # loop state is just passed through
            
                return (elements_finished, 
                        _next_input,
                        state,
                        output,
                        loop_state)
            
            # combine the 2 loop function    
            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:    # time == 0
                    assert previous_output is None and previous_state is None # why this assert??
                    return loop_fn_initial()
                else:
                    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
            
            decoder_outputs_ta, _, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn)
            decoder_outputs = decoder_outputs_ta.stack() # outputs are a list of 2D tensors. Stack them in a single 3D tensor to compute the output projections
            
            # flatten the outputs since tf.matmul() can multiply only 2D tensors
            decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
            decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
            decoder_pred_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b) # here is where we use W and b again
            
            # restore the shape [time_steps, batch_size, input_dim]
            self.tr_outputs = tf.reshape(decoder_pred_flat, (decoder_max_steps, decoder_batch_size, self.input_dim))
            
    
    # implementation using TF seq2seq library          
    def _init_decoder_TF(self):
        with tf.variable_scope("Decoder") as decoder_scope: 
            
            _, d_batch_size, _ = tf.unstack(tf.shape(self.encoder_inputs)) 
            self.EOS_slice = tf.ones([1,d_batch_size,self.input_dim], dtype=tf.float32) * self.EOS
            decoder_train_inputs = tf.concat([self.EOS_slice, self.decoder_outputs], axis=0)
            decoder_train_lengths = self.encoder_inputs_length + 1
            
            # projection of decoder output
            self.output_layer = layers_core.Dense(self.input_dim, use_bias=False, name="output_proj")
                     
            # ------------------ TRAINING ------------------
            
            # Training Helper
            tr_helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_train_inputs, 
                    decoder_train_lengths, 
                    time_major=True)
            
            # Decoder
            tr_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.decoder_cell, 
                    tr_helper, 
                    self.encoder_state, 
                    output_layer=None)
            
            # Dynamic decoding
            tr_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    tr_decoder,
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope)

            self.tr_outputs = self.output_layer(tr_outputs.rnn_output) # projection applied all togheter at the end
                    
            # ------------------ INFERENCE ------------------ 

            # callable that returns (finished, next_inputs) for the first iteration.
            def initialize_fn():
                next_inputs = tf.ones([d_batch_size,self.input_dim], dtype=tf.float32) * self.EOS
                finished = tf.tile([False], [d_batch_size])
                return (finished, next_inputs)
            
            # callable that takes (time, outputs, state) and emits tensor sample_ids.
            def sample_fn(time, outputs, state): 
                del time, outputs, state # not using them: we always return the same output
                return tf.zeros([d_batch_size], dtype=tf.int32)
            
            # callable that takes (time, outputs, state, sample_ids) and emits (finished, next_inputs, next_state)
            def next_inputs_fn(time, outputs, state, sample_ids): 
                del sample_ids
#                finished = tf.tile([False], [d_batch_size])
#                next_inputs = outputs
#                max_time = tf.ones([d_batch_size], dtype=tf.int32) * 10
#                finished = (time >= max_time)
                finished = (time >= decoder_train_lengths)
                next_inputs = tf.where(finished, self.EOS_slice[0,:,:], outputs)
                return (finished, next_inputs, state)
            
            # Inference Helper 
            inf_helper = tf.contrib.seq2seq.CustomHelper(
                    initialize_fn, 
                    sample_fn, 
                    next_inputs_fn)

            # Decoder
            inf_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.decoder_cell,
                    inf_helper,
                    self.encoder_state,
                    #output_layer=None,
                    output_layer=self.output_layer)  # projection applied per timestep
                    
            # Dynamic decoding
            inf_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                    inf_decoder,
                    maximum_iterations=tf.reduce_max(decoder_train_lengths),
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope)
            
            #self.inf_outputs = self.output_layer(inf_outputs.rnn_output)
            self.inf_outputs = inf_outputs.rnn_output
            
            
    def _init_loss(self):
        with tf.variable_scope("Loss"): 
            
            # TODO: L2 norm and lr decay
            # L2 regularization (to avoid overfitting and to have a better generalization capacity)
            reg_loss = 0
            for tf_var in tf.trainable_variables():
                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
#                if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
#                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
            
            # compute training loss
            decoder_train_outputs = tf.concat([self.decoder_outputs, self.EOS_slice], axis=0)
#            self.train_loss = tf.losses.mean_squared_error(labels=decoder_train_outputs, predictions=self.tr_outputs)
            self.train_loss = tf.losses.mean_squared_error(labels=decoder_train_outputs, predictions=self.inf_outputs) # when training with TF inference
            
            # Calculate and clip gradients
            parameters = tf.trainable_variables()
            gradients = tf.gradients(self.train_loss, parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            
            # Optimization
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))
            
            # tensorboard
            tf.summary.scalar('train_loss', self.train_loss)
            tvars = tf.trainable_variables()
            for tvar in tvars:
                tf.summary.histogram(tvar.name.replace(':','_'), tvar)
            self.merged_summary = tf.summary.merge_all()