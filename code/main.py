from gen_model import s2s_ts_Model
import numpy as np
np.set_printoptions(precision=2)
import time
import tensorflow as tf
from TS_datasets import *
import argparse, sys
from utils import classify_with_knn, mse_and_corr, anomaly_detect

plot_on = 0
_seed = None
np.random.seed(_seed)

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='AF', help="ID of the dataset", type=str)
parser.add_argument("--cell_type", default='GRU', help="type of cell for encoder/decoder (RNN, LSTM, GRU)", type=str)
parser.add_argument("--num_layers", default=1, help="number of stacked layers in ecoder/decoder", type=int)
parser.add_argument("--hidden_units", default=10, help="number of hidden units in the encoder/decoder", type=int)
parser.add_argument("--decoder_init", default='all', help="init decoder with last state of only last layer (last, zero, all)", type=str)
parser.add_argument("--sched_prob", default=1.0, help="probability of sampling from teacher signal in scheduled sampling", type=float)
parser.add_argument("--learning_rate", default=0.001, help="Adam initial learning rate", type=float)
parser.add_argument("--batch_size", default=25, help="number of samples in each batch", type=int)
parser.add_argument("--w_align", default=0.2, help="kernel alignment weight", type=float)
parser.add_argument("--w_l2", default=0.0, help="l2 norm regularization weight", type=float)
parser.add_argument("--num_epochs", default=5000, help="number of epochs in training", type=int)
parser.add_argument("--max_gradient_norm", default=1.0, help="max gradient norm for gradient clipping", type=float)
args = parser.parse_args()

config = dict(cell_type = args.cell_type,
              num_layers = args.num_layers,
              hidden_units = args.hidden_units,
              max_gradient_norm = args.max_gradient_norm, 
              learning_rate = args.learning_rate,
              decoder_init = args.decoder_init,
              num_epochs = args.num_epochs,
              batch_size = args.batch_size,
              sched_prob = args.sched_prob,
              w_align = args.w_align,
              w_l2 = args.w_l2)
print(config)

# ================= DATASET =================
if args.dataset_id == 'SYNTH':
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getSynthData(name='Lorentz', 
                                                                tr_data_samples=100, 
                                                                vs_data_samples=100, 
                                                                ts_data_samples=500)
elif args.dataset_id == 'ECG':
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getECGData()

elif args.dataset_id == 'ECG2':
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getECGDataFull()
       
elif args.dataset_id == 'JAP':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getJapDataFull()

elif args.dataset_id == 'JAPm':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getJapDataMiss(kernel='TCK', inp='zero', miss=0.9)

elif args.dataset_id == 'ARAB':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getArab()
   
elif args.dataset_id == 'LIB':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getLibras()
    
elif args.dataset_id == 'CHAR':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getCharDataFull()

elif args.dataset_id == 'WAF':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getWafer()
    
elif args.dataset_id == 'SIN':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getSins() 

elif args.dataset_id == 'MSO':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getMSO()  
    
elif args.dataset_id == 'ODE':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getODE()  
    
elif args.dataset_id == 'ODE2':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getODE_mc()      
    
elif args.dataset_id == 'AUS':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getAuslan()
    
elif args.dataset_id == 'BLOOD':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getBlood()  

elif args.dataset_id == 'AF':        
    (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, _) = getAF()  
    
else:
    sys.exit('Invalid dataset_id')


config['input_dim'] = train_data.shape[2]
print('\n**** Processing {}: Tr{}, Vs{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, valid_data.shape, test_data.shape))

# sort validation data (for visualize the learned K)
sort_idx = np.argsort(valid_labels,axis=0)[:,0]
valid_data = valid_data[:,sort_idx,:]
valid_targets = valid_targets[:,sort_idx,:]
K_vs = K_vs[sort_idx,:][:,sort_idx]

# ================= GRAPH =================
tf.reset_default_graph() # needed when working with iPython
sess = tf.Session()
tf.set_random_seed(_seed)
G = s2s_ts_Model(config)
sess.run(tf.global_variables_initializer())

# trainable parameters count
total_parameters = 0
for variable in tf.trainable_variables():
    vshape = variable.get_shape()
    variable_parametes = 1
    for dim in vshape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print('Total parameters: {}'.format(total_parameters))

# ================= DEBUG =================
#fd = {G.encoder_inputs: train_data[:,11:20,:], G.encoder_inputs_length: train_len[11:20], G.decoder_outputs: train_targets[:,11:20,:], G.prior_K: K_tr[:,11:20][11:20,:]}
#fd = {G.encoder_inputs: train_data, G.encoder_inputs_length: train_len, G.decoder_outputs: train_targets, G.prior_K: K_tr}
#e_states = sess.run(G.lstm_conc, fd )  
#
#raise

# ================= TRAINING =================

# initialize training variables
batch_size = config['batch_size']
time_tr_start = time.time()
max_batches = train_data.shape[1]//batch_size
teach_loss_track = []
inf_loss_track = []
min_vs_loss = np.infty
model_name = "../models/tkae_"+str(time.strftime("%Y%m%d-%H%M%S"))+".ckpt" 
#train_writer = tf.summary.FileWriter('/logs', graph=sess.graph)
saver = tf.train.Saver()

try:
    for ep in range(config['num_epochs']):
        
        # shuffle training data
        idx = np.random.permutation(train_data.shape[1])
        train_data_s = train_data[:,idx,:] 
        train_targets_s = train_targets[:,idx,:] 
        K_tr_s = K_tr[idx,:][:,idx]
        
        for batch in range(max_batches):
            
            fdtr = {G.encoder_inputs: train_data_s[:,(batch)*batch_size:(batch+1)*batch_size,:],
                    G.encoder_inputs_length: train_len[(batch)*batch_size:(batch+1)*batch_size],
                    G.decoder_outputs: train_targets_s[:,(batch)*batch_size:(batch+1)*batch_size,:],
                    G.prior_K: K_tr_s[(batch)*batch_size:(batch+1)*batch_size, (batch)*batch_size:(batch+1)*batch_size]}  
            
            _, inf_loss, teach_loss = sess.run([G.update_step, G.inf_loss, G.teach_loss], fdtr)    
                        
            inf_loss_track.append(inf_loss)
            teach_loss_track.append(teach_loss)
            
        # check training progress on the validations set    
        if ep % 100 == 0:            
            print('Ep: {}'.format(ep))
            
            fdvs = {G.encoder_inputs: valid_data,
                    G.encoder_inputs_length: valid_len,
                    G.decoder_outputs: valid_targets,
                    G.prior_K: K_vs}
            (inf_outvs, 
             inf_lossvs, 
             teach_outvs, 
             teach_lossvs, 
             tot_lossvs,
             reg_lossvs, 
             k_lossvs,
             vs_code_K, 
#             summary
             ) = (sess.run([G.inf_outputs, 
                           G.inf_loss, 
                           G.teach_outputs, 
                           G.teach_loss,
                           G.tot_loss,
                           G.reg_loss, 
                           G.k_loss,
                           G.code_K, 
                           ], fdvs)) #G.merged_summary
#            train_writer.add_summary(summary, ep)
            
            fdts = {G.encoder_inputs: test_data,
                    G.encoder_inputs_length: test_len}
            inf_outs = sess.run(G.inf_outputs, fdts)            
            test_mse, _ = mse_and_corr(test_targets, inf_outs, test_len)
            
            print('TS: MSE=%.3f -- VS: tot_loss=%.3f inf_loss=%.3f, k_loss=%.3f, reg_loss=%.3f -- TR: mean_loss=%.3f'
                  %(test_mse, tot_lossvs, inf_lossvs, k_lossvs*args.w_align, reg_lossvs*args.w_l2, np.mean(inf_loss_track[-10:])))     
            
            # Save model yielding best loss on validation
            if tot_lossvs < min_vs_loss:
                min_vs_loss = tot_lossvs
                tf.add_to_collection("encoder_inputs",G.encoder_inputs)
                tf.add_to_collection("encoder_inputs_length",G.encoder_inputs_length)
                tf.add_to_collection("decoder_outputs",G.decoder_outputs)
                tf.add_to_collection("code_K",G.code_K)
                tf.add_to_collection("inf_outputs",G.inf_outputs)
                tf.add_to_collection("inf_loss",G.inf_loss)
                tf.add_to_collection("context_vector",G.context_vector)
                save_path = saver.save(sess, model_name)        
                                           
            # plot a random ts from the validation set and code inner products
            if plot_on:
                import matplotlib.pyplot as plt
                plt.matshow(vs_code_K)
                plt.show()
                
                plot_idx1 = np.random.randint(low=0,high=valid_targets.shape[1])
                target = valid_targets[:,plot_idx1,:]
                inf_pred = inf_outvs[:,plot_idx1,:]
                teach_pred = teach_outvs[:,plot_idx1,:]
                plt.plot(target.flatten(), linewidth=1.5, label='target')
                plt.plot(teach_pred.flatten(), '--', label='teach')
                plt.plot(inf_pred.flatten(), linewidth=1.5, label='inf')
                
                plt.legend(loc='best')
                plt.show()  
                                                    
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

tf.reset_default_graph() # be sure that correct weights are loaded
saver.restore(sess, model_name)

fdts = {G.encoder_inputs: test_data,
        G.encoder_inputs_length: test_len}
inf_outs, ts_context = sess.run([G.inf_outputs, G.context_vector], fdts)

# MSE and corr
test_mse, test_corr = mse_and_corr(test_targets, inf_outs, test_len)
print('Test MSE=%.3f, Corr=%.3f'%(test_mse, test_corr))

# anomaly detect
anomaly_detect(test_targets, inf_outs, test_len, test_labels, 0.3, plot_on)

# kNN classification on the codes
fdtr = {G.encoder_inputs: train_data,
        G.encoder_inputs_length: train_len}
tr_context = sess.run(G.context_vector, fdtr)
acc, f1 = classify_with_knn(tr_context, train_labels[:, 0], ts_context, test_labels[:, 0], k=3)
print('kNN -- acc: %.3f, F1: %.3f'%(acc, f1))

#train_writer.close()
sess.close()

#with open('results', 'a') as f:
#    f.write('cell: '+args.cell_type+', n_layers: '+str(args.num_layers)+', h_units: '+str(args.hidden_units)+', bidir: '+str(args.bidirect)+', max_grad: '+str(args.max_gradient_norm)+ 
#            ', lr: '+str(args.learning_rate)+', decoder_init: '+args.decoder_init+', reverse_inp: '+str(args.reverse_input)+', sched_prob: '+str(args.sched_prob)+ 
#            ', w_align: '+str(args.w_align)+', time: '+str((time_tr_end-time_tr_start)//60)+', MSE: '+str(ts_loss)+', ACC: '+str(accuracy)+'\n')