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
        self.max_gradient_norm = config['max_gradient_norm']
        self.learning_rate = config['learning_rate']
        self.decoder_init = config['decoder_init']
        self.sched_prob = config['sched_prob']
        self.w_align = config['w_align']
        self.w_l2 = config['w_l2']
        
        self.EOS = 0
        
        self._make_graph()
        
        
    def _make_graph(self):    
        
        self._init_placeholders()
        self._init_cells()
        
        # ----- ENCODER -----
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
        self.missing_mask = tf.placeholder(shape=(None, None, self.input_dim), dtype=tf.float32, name='missing_mask')
        self.prior_K = tf.placeholder(shape=(None, None), dtype=tf.float32, name='prior_K')

        
    def _init_cells(self):
                    
        if self.cell_type == 'LSTM':
            self.encoder_fw_cell= MultiRNNCell([LSTMCell(self.hidden_units) for _ in range(self.num_layers)])
            self.encoder_bw_cell= MultiRNNCell([LSTMCell(self.hidden_units) for _ in range(self.num_layers)])
            self.decoder_cell= MultiRNNCell([LSTMCell(self.hidden_units) for _ in range(self.num_layers)])
        elif self.cell_type == 'GRU': 
            self.encoder_fw_cell= MultiRNNCell([GRUCell(self.hidden_units) for _ in range(self.num_layers)])
            self.encoder_bw_cell= MultiRNNCell([GRUCell(self.hidden_units) for _ in range(self.num_layers)])
            self.decoder_cell= MultiRNNCell([GRUCell(self.hidden_units) for _ in range(self.num_layers)])
        elif self.cell_type == 'RNN': 
            self.encoder_fw_cell= MultiRNNCell([BasicRNNCell(self.hidden_units) for _ in range(self.num_layers)])
            self.encoder_bw_cell= MultiRNNCell([BasicRNNCell(self.hidden_units) for _ in range(self.num_layers)])
            self.decoder_cell= MultiRNNCell([BasicRNNCell(self.hidden_units) for _ in range(self.num_layers)])
        else:
            raise        
            
                
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
            conc_state = []
            if isinstance(encoder_fw_state[-1], LSTMStateTuple):   
                for i in range(self.num_layers):
#                    h_conc = tf.contrib.layers.fully_connected(tf.concat((encoder_fw_state[i].h, encoder_bw_state[i].h), 1),
#                                                              num_outputs=self.hidden_units,
#                                                              activation_fn=tf.nn.tanh)
#                    c_conc = tf.contrib.layers.fully_connected(tf.concat((encoder_fw_state[i].c, encoder_bw_state[i].c), 1),
#                                                              num_outputs=self.hidden_units,
#                                                              activation_fn=tf.nn.tanh)
#                    conc_state.append(LSTMStateTuple(c=c_conc, h=h_conc))
                    lstm_conc = tf.contrib.layers.fully_connected(tf.concat((encoder_fw_state[i].h, 
                                                                             encoder_bw_state[i].h, 
                                                                             encoder_fw_state[i].c, 
                                                                             encoder_bw_state[i].c), 1),
                                                                  num_outputs=self.hidden_units*2,
                                                                  activation_fn=tf.nn.tanh)
                    conc_state.append(LSTMStateTuple(h=lstm_conc[:,:self.hidden_units], c=lstm_conc[:,self.hidden_units:]))
                                                                          
            elif isinstance(encoder_fw_state[-1], tf.Tensor):       
                for i in range(self.num_layers):
                    conc_state.append(tf.contrib.layers.fully_connected(tf.concat((encoder_fw_state[i], encoder_bw_state[i]), 1),
                                                                      num_outputs=self.hidden_units,
                                                                      activation_fn=tf.nn.tanh))
                
            self.encoder_states = tuple(conc_state)
                
                                
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

        if self.decoder_init == 'all': # context vector is the first state of each decoder layer
            self.context_vector = tf.concat(to_be_concat,1)
        else: # context vector is first state of the first decoder layer
            self.context_vector = to_be_concat[0]
    
        # inner products of the context vectors
        self.code_K = tf.tensordot(self.context_vector, tf.transpose(self.context_vector), axes=1)
        
        
    def _init_decoder(self):
        with tf.variable_scope("Decoder") as decoder_scope: 
            
            _, d_batch_size, _ = tf.unstack(tf.shape(self.encoder_inputs)) 
            self.EOS_slice = tf.ones([1,d_batch_size,self.input_dim], dtype=tf.float32) * self.EOS
            decoder_train_inputs = tf.concat([self.EOS_slice, self.decoder_outputs], axis=0)
            self.decoder_train_lengths = self.encoder_inputs_length + 1 # 1 extra length for the EOS value
            
            # projection of decoder output
            self.output_layer = layers_core.Dense(self.input_dim, use_bias=False, name="output_proj")
                     
            # ------------------ TEACHER FORCING ------------------
            
            # Training Helper
            teach_helper = seq2seq.TrainingHelper(
                    decoder_train_inputs, 
                    self.decoder_train_lengths, 
                    time_major=True)
            
            # Decoder
            teach_decoder = seq2seq.BasicDecoder(
                    self.decoder_cell, 
                    teach_helper, 
                    self.decoder_init_state, 
                    output_layer=None)
            
            # Dynamic decoding
            teach_outputs, _, _ = seq2seq.dynamic_decode(
                    teach_decoder,
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope)

            self.teach_outputs = self.output_layer(teach_outputs.rnn_output) # projections applied all togheter at the end
                    
            # ------------------ INFERENCE ------------------ 

            # callable that returns (finished, next_inputs) for the first iteration.
            def initialize_fn():
                next_inputs = tf.ones([d_batch_size,self.input_dim], dtype=tf.float32) * self.EOS
                finished = tf.tile([False], [d_batch_size])
                return (finished, next_inputs)
            
            # callable that takes (time, outputs, state) and emits tensor sample_ids.
            def sample_fn(time, outputs, state): 
                del time, outputs, state # not using them: we always return the same ids
                return tf.zeros([d_batch_size], dtype=tf.int32)
            
            # callable that takes (time, outputs, state, sample_ids) and emits (finished, next_inputs, next_state)
            def next_inputs_fn(time, outputs, state, sample_ids): 
                del sample_ids
                finished = (time >= self.decoder_train_lengths)
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
                    maximum_iterations=tf.reduce_max(self.decoder_train_lengths),
                    output_time_major=True,
                    swap_memory=True,
                    scope=decoder_scope)
            
            self.inf_outputs = inf_outputs.rnn_output
            
            
            # ------------------ SCHEDULED SAMPLING ------------------
            
            # Scheduled Sampling Helper
            sched_helper = seq2seq.ScheduledOutputTrainingHelper(decoder_train_inputs,
                                                                 self.decoder_train_lengths,
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
            
            # reshape target outputs to match the size of the predicted outputs
            max_time_step = tf.reduce_max(self.decoder_train_lengths)
            decoder_train_outputs = tf.slice(tf.concat([self.decoder_outputs, self.EOS_slice], axis=0), [0,0,0], [max_time_step, -1, -1])
            train_miss_mask = tf.slice(tf.concat([self.missing_mask, tf.zeros_like(self.EOS_slice)], axis=0), [0,0,0], [max_time_step, -1, -1])
            
            # mask padding elements beyond the target sequence length with values 0 
            decoder_mask = tf.transpose(tf.sequence_mask(self.decoder_train_lengths, max_time_step, dtype=tf.float32))
           
            # discard the first output (produced when EOS is fed in) and apply the mask
            self.teach_outputs = tf.expand_dims(decoder_mask,-1)*self.teach_outputs
            self.inf_outputs = tf.expand_dims(decoder_mask,-1)*self.inf_outputs
            self.sched_outputs = tf.expand_dims(decoder_mask,-1)*self.sched_outputs 
            
            parameters = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            
            # kernel alignment loss with normalized Frobenius norm
            code_K_norm = self.code_K/tf.norm(self.code_K, ord='fro', axis=[-2,-1])
            prior_K_norm = self.prior_K/tf.norm(self.prior_K, ord='fro', axis=[-2,-1])
            self.k_loss = tf.norm(code_K_norm - prior_K_norm, ord='fro', axis=[-2,-1])
            
            # L2 norm loss
            self.reg_loss = 0
            for tf_var in tf.trainable_variables():
                if not ("Bias" in tf_var.name or "noreg" in tf_var.name):
                    self.reg_loss += tf.nn.l2_loss(tf_var)
                     
            # teacher loss
            self.teach_loss = tf.reduce_sum(
                    (decoder_train_outputs*train_miss_mask - self.teach_outputs*train_miss_mask)**2
                    )/tf.reduce_sum(train_miss_mask)
                                 
            # inference loss
            self.inf_loss = tf.reduce_sum(
                    (decoder_train_outputs*train_miss_mask - self.inf_outputs*train_miss_mask)**2
                    )/tf.reduce_sum(train_miss_mask)
                       
            #  scheduled sampling loss
            self.sched_loss = tf.reduce_sum(
                    (decoder_train_outputs*train_miss_mask - self.sched_outputs*train_miss_mask)**2
                    )/tf.reduce_sum(train_miss_mask) 
            
            # Huber loss
#            self.sched_loss = tf.losses.huber_loss(labels=decoder_train_outputs, predictions=self.sched_outputs, delta=0.5)
            
#            # correntropy loss
#            sig = 2
#            cnum = 1 - tf.reduce_mean( tf.exp(- tf.pow(decoder_train_outputs - self.sched_outputs,2))/(2*sig**2) )
#            cden = 1 - tf.exp(-1/(2*sig**2))
#            self.corr_sched_loss = cnum/cden
            
            
            # ============= TOT LOSS =============
            self.tot_loss = self.sched_loss + self.w_align*self.k_loss + self.w_l2*self.reg_loss
                        
            # Calculate and clip gradients
            gradients = tf.gradients(self.tot_loss, parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))
                        
#            # ============= TENSORBOARD =============             
#            mean_grads = tf.reduce_mean([tf.reduce_mean(grad) for grad in gradients])
#            tf.summary.scalar('mean_grads', mean_grads)
#            tf.summary.scalar('teach_loss', self.teach_loss)
#            tf.summary.scalar('inf_loss', self.inf_loss)
#            tf.summary.scalar('sched_loss', self.sched_loss)
#            tf.summary.scalar('k_loss', self.k_loss)
#            tvars = tf.trainable_variables()
#            for tvar in tvars:
#                tf.summary.histogram(tvar.name.replace(':','_'), tvar)
#            self.merged_summary = tf.summary.merge_all()