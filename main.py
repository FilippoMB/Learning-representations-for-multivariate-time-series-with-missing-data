from gen_model import s2s_ts_Model
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
import time
import tensorflow as tf
from TS_datasets import getSynthData, getECGData, getJapData
import argparse, sys

plot_on = 0

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='JAP', help="ID of the dataset (SYNTH, ECG, JAP)", type=str)
parser.add_argument("--cell_type", default='GRU', help="type of cell for encoder/decoder (RNN, LSTM, GRU)", type=str)
parser.add_argument("--num_layers", default=1, help="number of stacked layers in ecoder/decoder", type=int)
parser.add_argument("--hidden_units", default=10, help="number of hidden units in the encoder/decoder. If encoder is bidirectional, decoders units are doubled", type=int)
parser.add_argument("--num_epochs", default=5000, help="number of epochs in training", type=int)
parser.add_argument("--batch_size", default=50, help="number of samples in each batch", type=int)
parser.add_argument("--bidirect", dest='bidirect', action='store_true', help="use an encoder which is bidirectional")
parser.add_argument("--max_gradient_norm", default=1.0, help="max gradient norm for gradient clipping", type=float)
parser.add_argument("--learning_rate", default=0.01, help="Adam initial learning rate", type=float)
parser.add_argument("--decoder_init", default='all', help="init decoder with last state of only last layer (last, zero, all)", type=str)
parser.add_argument("--reverse_input", dest='reverse_input', action='store_true', help="fed input reversed for training")
parser.add_argument("--sched_prob", default=0.9, help="probability of sampling from teacher signal in scheduled sampling", type=float)
parser.add_argument("--w_align", default=0, help="kernel alignment weight", type=float)
parser.set_defaults(bidirect=True)
parser.set_defaults(reverse_input=False)
args = parser.parse_args()

config = dict(cell_type = args.cell_type,
              num_layers = args.num_layers,
              hidden_units = args.hidden_units,
              bidirect = args.bidirect,
              max_gradient_norm = args.max_gradient_norm, 
              learning_rate = args.learning_rate,
              decoder_init = args.decoder_init,
              reverse_input = args.reverse_input,
              num_epochs = args.num_epochs,
              batch_size = args.batch_size,
              sched_prob = args.sched_prob,
              w_align = args.w_align)
print(config)

# ================= DATASET =================
if args.dataset_id == 'SYNTH':
    (train_data, _, train_len, train_targets, K_tr,
        valid_data, _, valid_len, valid_targets, K_vs,
        test_data, _, test_len, test_targets, _) = getSynthData(name='Lorentz', 
                                                                tr_data_samples=2000, 
                                                                vs_data_samples=2000, 
                                                                ts_data_samples=2000)
elif args.dataset_id == 'ECG':
    (train_data, _, train_len, train_targets, K_tr,
        valid_data, _, valid_len, valid_targets, K_vs,
        test_data, _, test_len, test_targets, _) = getECGData(tr_ratio = 0.4)
       
elif args.dataset_id == 'JAP':        
    (train_data, _, train_len, train_targets, K_tr,
        valid_data, _, valid_len, valid_targets, K_vs,
        test_data, _, test_len, test_targets, _) = getJapData(kernel='TCK',inp=None)
    
else:
    sys.exit('Invalid dataset_id')

config['input_dim'] = train_data.shape[2]
print('\n**** Processing {}: Tr{}, Vs{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, valid_data.shape, test_data.shape))

# revert time
if config['reverse_input']:
    train_data = train_data[::-1,:,:]
    valid_data = valid_data[::-1,:,:]
    test_data = test_data[::-1,:,:]

# ================= GRAPH =================
tf.reset_default_graph() # needed when working with iPython
sess = tf.Session()
G = s2s_ts_Model(config)
sess.run(tf.global_variables_initializer())

# ================= DEBUG =================
#fd = {G.encoder_inputs: train_data, G.encoder_inputs_length: train_len, G.decoder_outputs: train_targets, G.prior_K: K_tr}
#teach_out, inf_out, sched_out,e_states = sess.run([G.teach_outputs, G.inf_outputs, G.sched_outputs,G.encoder_states], fd )  
#
#raise
# ================= TRAINING =================

# initialize training stuff
batch_size = config['batch_size']
time_tr_start = time.time()
max_batches = train_data.shape[1]//batch_size
teach_loss_track = []
inf_loss_track = []
min_vs_loss = np.infty
model_name = "/tmp/tkae_models/m_"+str(time.strftime("%Y%m%d-%H%M%S"))+".ckpt"
train_writer = tf.summary.FileWriter('/tmp/tensorboard', graph=sess.graph)
saver = tf.train.Saver()

try:
    for ep in range(config['num_epochs']):
        
        # shuffle training data
        idx = np.random.permutation(train_data.shape[1])
        train_data = train_data[:,idx,:] 
        train_targets = train_targets[:,idx,:] 
        K_tr = K_tr[idx,:][:,idx]
        
        for batch in range(max_batches):
            
            fdtr = {G.encoder_inputs: train_data[:,(batch)*batch_size:(batch+1)*batch_size,:],
                    G.encoder_inputs_length: train_len[(batch)*batch_size:(batch+1)*batch_size],
                    G.decoder_outputs: train_targets[:,(batch)*batch_size:(batch+1)*batch_size,:],
                    G.prior_K: K_tr[(batch)*batch_size:(batch+1)*batch_size, (batch)*batch_size:(batch+1)*batch_size]}  
            
            _, inf_loss, teach_loss = sess.run([G.update_step, G.inf_loss, G.teach_loss], fdtr)    
                        
            inf_loss_track.append(inf_loss)
            teach_loss_track.append(teach_loss)
            
        # check training progress on the validations set    
        if ep % 100 == 0:            
            print('Ep: {}'.format(ep))
            
            fdvs = {G.encoder_inputs: valid_data,
                    G.encoder_inputs_length: valid_len,
                    G.decoder_outputs: valid_targets,
                    G.prior_K: K_vs
                    }
            inf_outvs, inf_lossvs, teach_outvs, teach_lossvs, vs_code_K, summary = sess.run([G.inf_outputs, G.inf_loss, G.teach_outputs, G.teach_loss, G.code_K, G.merged_summary], fdvs)
            train_writer.add_summary(summary, ep)
            print('VS: inf_loss=%.3f, teach_loss=%.3f -- TR: min_loss=.%3f'%(inf_lossvs, teach_lossvs, np.min(inf_loss_track)))     
            
            # Save model yielding best results on validation
            if inf_lossvs < min_vs_loss:
                tf.add_to_collection("encoder_inputs",G.encoder_inputs)
                tf.add_to_collection("encoder_inputs_length",G.encoder_inputs_length)
                tf.add_to_collection("decoder_outputs",G.decoder_outputs)
                tf.add_to_collection("code_K",G.code_K)
                tf.add_to_collection("inf_outputs",G.inf_outputs)
                tf.add_to_collection("inf_loss",G.inf_loss)
                tf.add_to_collection("context_vector",G.context_vector)
                save_path = saver.save(sess, model_name)        
                                           
            # plot a random ts from the validation set
            if plot_on:
                plt.matshow(vs_code_K)
                plt.show(block=False)
                plot_idx1 = np.random.randint(low=0,high=valid_targets.shape[1])
                plot_idx2 = np.random.randint(low=0,high=valid_targets.shape[2])
                target = valid_targets[:,plot_idx1,plot_idx2]
                inf_pred = inf_outvs[:-1,plot_idx1,plot_idx2]
                teach_pred = teach_outvs[:-1,plot_idx1,plot_idx2]
                plt.plot(target, label='target')
                plt.plot(inf_pred, label='inf')
                plt.plot(teach_pred, label='teach')
                plt.legend(loc='upper right')
                plt.show(block=False)  
                                                    
except KeyboardInterrupt:
    print('training interrupted')

if plot_on:
    plt.plot(teach_loss_track, label='teach_loss_track')
    plt.legend(loc='upper right')
    plt.show(block=False)
    plt.plot(inf_loss_track, label='inf_loss_track')
    plt.legend(loc='upper right')
    plt.show(block=False)
    
time_tr_end = time.time()
print('Tot training time: {}'.format((time_tr_end-time_tr_start)//60) )

# ================= TEST =================
print('************ TEST ************ \n>>restoring from:'+model_name+'<<')

sess.run(tf.global_variables_initializer()) # be sure that correct weights are loaded
saver.restore(sess, model_name)

fdts = {G.encoder_inputs: test_data,
        G.encoder_inputs_length: test_len,
        G.decoder_outputs: test_targets
        }
ts_loss = sess.run(G.inf_loss, fdts)
print('Test MSE: %.3f' % (ts_loss))

train_writer.close()
sess.close()

with open('results','a') as f:
    f.write('cell: '+args.cell_type+', n_layers: '+str(args.num_layers)+', h_units: '+str(args.hidden_units)+', bidir: '+str(args.bidirect)+', max_grad: '+str(args.max_gradient_norm)+ 
            ', lr: '+str(args.learning_rate)+', decoder_init: '+args.decoder_init+', reverse_inp: '+str(args.reverse_input)+', sched_prob: '+str(args.sched_prob)+ 
            ', w_align: '+str(args.w_align)+', time: '+str((time_tr_end-time_tr_start)//60)+', MSE: '+str(ts_loss)+'\n')