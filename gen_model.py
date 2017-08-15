import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, BasicRNNCell, GRUCell, LSTMStateTuple, MultiRNNCell
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
import sys

class s2s_ts_Model():
    
    def __init__(self,config):
        self.cell_type = config['cell_type']
        self.num_layers = config['num_layers']
        self.hidden_units = config['hidden_units']
        self.input_dim = config['input_dim']
        self.bidirect = config['bidirect']
        self.max_gradient_norm = config['max_gradient_norm']
        self.learning_rate = config['learning_rate']
        self.decoder_init = config['decoder_init']
        self.sched_prob = config['sched_prob']
        self.w_align = config['w_align']
        
        self.EOS = 0
        
        self._make_graph()
        
        
    def _make_graph(self):    
        
        self._init_placeholders()
        self._init_cells()
        
        # ----- ENCODER -----
        if self.bidirect == False:
            self._init_simple_encoder()
        else:
            self._init_bidirectional_encoder()
            
        # ----- DEC INIT STATE -----
        self._init_decoder_state()
            
        # ----- CONTEXT -----
        self._get_context()
            
        # ----- DECODER -----
        self._init_decoder()
        
        # ----- LOSS -----
        self._init_loss()
                   
    
    def _init_placeholders(self):        
        # Everything is time-major
        self.encoder_inputs = tf.placeholder(shape=(None, None, self.input_dim), dtype=tf.float32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_outputs = tf.placeholder(shape=(None, None, self.input_dim), dtype=tf.float32, name='decoder_outputs')     
        self.prior_K = tf.placeholder(shape=(None, None), dtype=tf.float32, name='prior_K')

        
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
   
            (_, self.encoder_states) = (tf.nn.dynamic_rnn(
                    self.encoder_fw_cell,
                    inputs= self.encoder_inputs,
                    sequence_length=self.encoder_inputs_length,
                    time_major=True,
                    dtype=tf.float32))
                        
                  
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
                self.encoder_states = tuple(LSTMStateTuple(c=tf.concat((encoder_fw_state[i].c, encoder_bw_state[i].c), 1), 
                                                      h=tf.concat((encoder_fw_state[i].h, encoder_bw_state[i].h), 1))
                                       for i in range(self.num_layers))                    
            elif isinstance(encoder_fw_state[-1], tf.Tensor):
                self.encoder_states = tuple(tf.concat((encoder_fw_state[i], encoder_bw_state[i]), 1) for i in range(self.num_layers))
                
                
    def _init_decoder_state(self):

        # all decoder layers -> encoder last layer
        if self.decoder_init == 'last': 
            self.decoder_init_state = tuple(self.encoder_states[-1] for _ in range(self.num_layers)) 
        
        # first decoder layer -> encoder last layer / other decoder layers -> zeros
        elif self.decoder_init == 'zero':
            if isinstance(self.encoder_states[-1], LSTMStateTuple):
                self.decoder_init_state = tuple([self.encoder_states[-1]] + 
                                                [LSTMStateTuple(c=tf.zeros_like(self.encoder_states[-1].c), h=tf.zeros_like(self.encoder_states[-1].h)) for _ in range(self.num_layers-1)])
            else:
                self.decoder_init_state = tuple([self.encoder_states[-1]] + [tf.zeros_like(self.encoder_states[-1]) for _ in range(self.num_layers-1)])
        
        # last states from all encoder layers (in reverse order)
        elif self.decoder_init == 'all': 
            self.decoder_init_state = self.encoder_states[::-1]     
        
        else:
            sys.exit('Invalid decoder initialization')            
    
    
    def _get_context(self):
        to_be_concat = []
        for state in self.decoder_init_state:
            if isinstance(state, LSTMStateTuple):
                to_be_concat.append(tf.concat((state.h, state.c),1))
            else:
                to_be_concat.append(state)

        if self.decoder_init == 'all':
            self.context_vector = tf.concat(to_be_concat,1)
        else:
            self.context_vector = to_be_concat[0]
    
        # inner products of the context vectors
        self.code_K = tf.tensordot(self.context_vector, tf.transpose(self.context_vector), axes=1)
        
    def _init_decoder(self):
        with tf.variable_scope("Decoder") as decoder_scope: 
            
            _, d_batch_size, _ = tf.unstack(tf.shape(self.encoder_inputs)) 
            self.EOS_slice = tf.ones([1,d_batch_size,self.input_dim], dtype=tf.float32) * self.EOS
            decoder_train_inputs = tf.concat([self.EOS_slice, self.decoder_outputs], axis=0)
            decoder_train_lengths = self.encoder_inputs_length + 1
            
            # projection of decoder output
            self.output_layer = layers_core.Dense(self.input_dim, use_bias=False, name="output_proj")
                     
            # ------------------ TEACHER FORCING ------------------
            
            # Training Helper
            teach_helper = seq2seq.TrainingHelper(
                    decoder_train_inputs, 
                    decoder_train_lengths, 
                    time_major=True)
            
            # Decoder
            teach_decoder = seq2seq.BasicDecoder(
                    self.decoder_cell, 
                    teach_helper, 
                    self.decoder_init_state, 
                    output_layer=None)
            
            # Dynamic decoding
            teach_outputs, final_context_state, _ = seq2seq.dynamic_decode(
                    teach_decoder,
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope)

            self.teach_outputs = self.output_layer(teach_outputs.rnn_output) # projection applied all togheter at the end
                    
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
                finished = (time >= decoder_train_lengths)
                next_inputs = tf.where(finished, self.EOS_slice[0,:,:], outputs)
                return (finished, next_inputs, state)
            
            # Inference Helper 
            inf_helper = seq2seq.CustomHelper(
                    initialize_fn, 
                    sample_fn, 
                    next_inputs_fn)

            # Decoder
            inf_decoder = seq2seq.BasicDecoder(
                    self.decoder_cell,
                    inf_helper,
                    self.decoder_init_state,
                    output_layer=self.output_layer)  # projection applied per timestep
                    
            # Dynamic decoding
            inf_outputs, _, _ = seq2seq.dynamic_decode(
                    inf_decoder,
                    maximum_iterations=tf.reduce_max(decoder_train_lengths),
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope)
            
            self.inf_outputs = inf_outputs.rnn_output
            
            
            # ------------------ SCHEDULED SAMPLING ------------------
            
            # Scheduled Sampling Helper
            sched_helper = seq2seq.ScheduledOutputTrainingHelper(decoder_train_inputs,
                                                                 decoder_train_lengths,
                                                                 self.sched_prob,
                                                                 time_major=True, 
                                                                 seed=None, 
                                                                 next_input_layer=None)
            
            # Decoder
            sched_decoder = seq2seq.BasicDecoder(
                    self.decoder_cell, 
                    sched_helper, 
                    self.decoder_init_state, 
                    output_layer=self.output_layer)          
            
            # Dynamic decoding
            sched_outputs, _, _ = seq2seq.dynamic_decode(
                    sched_decoder,
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope)
            
            self.sched_outputs = sched_outputs.rnn_output
            
            
    def _init_loss(self):
        with tf.variable_scope("Loss"): 
            
            decoder_train_outputs = tf.concat([self.decoder_outputs, self.EOS_slice], axis=0)
            parameters = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            # kernel alignment loss with normalized Frobenius norm
            code_K_norm = self.code_K/tf.norm(self.code_K, ord='fro', axis=[-2,-1])
            prior_K_norm = self.prior_K/tf.norm(self.prior_K, ord='fro', axis=[-2,-1])
            self.k_loss = tf.norm(code_K_norm - prior_K_norm, ord='fro', axis=[-2,-1])
#            self.k_loss = tf.norm(self.code_K - self.prior_K, ord='fro', axis=[-2,-1])
            
            # TODO: think about dropout, L2 norm and lr decay
#            reg_loss = 0
#            for tf_var in tf.trainable_variables():
#                reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
#                if not ("Bias" in tf_var.name or "Output_" in tf_var.name):
#                    reg_loss += tf.reduce_mean(tf.nn.l2_loss(tf_var))
            
            
            # teacher loss
            self.teach_loss = tf.losses.mean_squared_error(labels=decoder_train_outputs, predictions=self.teach_outputs)
                                 
            # inference loss
            self.inf_loss = tf.losses.mean_squared_error(labels=decoder_train_outputs, predictions=self.inf_outputs)
                       
            #  scheduled sampling loss
            self.sched_loss = tf.losses.mean_squared_error(labels=decoder_train_outputs, predictions=self.sched_outputs)
            
            # ============= TOT LOSS =============
            self.tot_loss = self.sched_loss + self.w_align*self.k_loss
                        
            # Calculate and clip gradients
            gradients = tf.gradients(self.tot_loss, parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))
            
#            # --------- TEACHER + INFERENCE LOSS --------- 
#            TODO: see if this stuff can work
#            self.teach_inf_loss = 0.1*self.teach_loss + self.inf_loss + self.w_align*self.k_loss
#            
#            # Calculate and clip gradients
#            teach_inf_gradients = tf.gradients(self.teach_inf_loss, parameters)
#            teach_inf_clipped_gradients, _ = tf.clip_by_global_norm(teach_inf_gradients, self.max_gradient_norm)
#                       
#            self.teach_inf_update_step = optimizer.apply_gradients(zip(teach_inf_clipped_gradients, parameters))
            
            # ============= TENSORBOARD =============             
            mean_grads = tf.reduce_mean([tf.reduce_mean(grad) for grad in gradients])
            tf.summary.scalar('mean_grads', mean_grads)
            tf.summary.scalar('teach_loss', self.teach_loss)
            tf.summary.scalar('inf_loss', self.inf_loss)
            tf.summary.scalar('sched_loss', self.sched_loss)
            tf.summary.scalar('k_loss', self.k_loss)
            tvars = tf.trainable_variables()
            for tvar in tvars:
                tf.summary.histogram(tvar.name.replace(':','_'), tvar)
            self.merged_summary = tf.summary.merge_all()