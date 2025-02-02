import tensorflow as tf
import argparse, sys
from TS_datasets import getImpTestData
import numpy as np
from utils import classify_with_knn, interp_data, mse_and_corr, dim_reduction_plot
import math, time

dim_red = 0
plot_on = 0

# parse input data
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id", default='Libras/LIB_miss05', help="ID of the dataset (SYNTH, ECG, JAP, etc..)", type=str)
parser.add_argument("--code_size", default=10, help="size of the code", type=int)
parser.add_argument("--w_reg", default=0.001, help="weight of the regularization in the loss function", type=float)
parser.add_argument("--a_reg", default=0.0, help="weight of the kernel alignment", type=float)
parser.add_argument("--num_epochs", default=5000, help="number of epochs in training", type=int)
parser.add_argument("--batch_size", default=25, help="number of samples in each batch", type=int)
parser.add_argument("--max_gradient_norm", default=1.0, help="max gradient norm for gradient clipping", type=float)
parser.add_argument("--learning_rate", default=0.001, help="Adam initial learning rate", type=float)
parser.add_argument("--hidden_size", default=30, help="size of the code", type=int)
parser.add_argument("--tied_weights", dest='tied_weights', action='store_true', help="use tied weights in the decoder")
parser.add_argument("--lin_dec", dest='lin_dec', action='store_true', help="use decoder with linear activations")
parser.add_argument("--interp_on", dest='interp_on', action='store_true', help="interpolate time series to match the length of the longest one")
parser.set_defaults(tied_weights=False)
parser.set_defaults(lin_dec=False)
parser.set_defaults(interp_on=False)
args = parser.parse_args()
print(args)

# ================= DATASET =================
(train_data_shaped, train_labels, train_len, _, K_tr,
        valid_data, _, valid_len, _, K_vs,
        test_data_shaped, test_labels, test_len, _, K_ts,
        M_train, M_valid, M_test,
        train_data_orig, valid_data_orig, test_data_orig) = getImpTestData(data_name=args.dataset_id,inp='last')
       
# interpolation
if np.min(train_len) < np.max(train_len) and args.interp_on:
    print('-- Data Interpolation --')
    train_data = interp_data(train_data_shaped, train_len)
    valid_data = interp_data(valid_data, valid_len)
    test_data = interp_data(test_data_shaped, test_len)
else:
    test_data = test_data_shaped
    train_data = train_data_shaped

# transpose and reshape [T, N, V] --> [N, T, V] --> [N, T*V]
train_data = np.transpose(train_data,axes=[1,0,2])
train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1]*train_data.shape[2]))
valid_data = np.transpose(valid_data,axes=[1,0,2])
valid_data = np.reshape(valid_data, (valid_data.shape[0], valid_data.shape[1]*valid_data.shape[2]))
test_data = np.transpose(test_data,axes=[1,0,2])
test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1]*test_data.shape[2]))    

M_train = np.transpose(M_train,axes=[1,0,2])
M_train = np.reshape(M_train, (M_train.shape[0], M_train.shape[1]*M_train.shape[2]))
M_valid = np.transpose(M_valid,axes=[1,0,2])
M_valid = np.reshape(M_valid, (M_valid.shape[0], M_valid.shape[1]*M_valid.shape[2]))
M_test = np.transpose(M_test,axes=[1,0,2])
M_test = np.reshape(M_test, (M_test.shape[0], M_test.shape[1]*M_test.shape[2]))

print('\n**** Processing {}: Tr{}, Vs{}, Ts{} ****\n'.format(args.dataset_id, train_data.shape, valid_data.shape, test_data.shape))

input_length = train_data.shape[1] # same for all inputs
# ================= GRAPH =================

# init session
tf.reset_default_graph() # needed when working with iPython
sess = tf.Session()

# placeholders
encoder_inputs = tf.placeholder(shape=(None,input_length), dtype=tf.float32, name='encoder_inputs')
prior_K = tf.placeholder(shape=(None, None), dtype=tf.float32, name='prior_K')
missing_mask = tf.placeholder(shape=(None,input_length), dtype=tf.float32, name='missing_mask')

# encoder
We1 = tf.Variable(tf.random_uniform((input_length, args.hidden_size), -1.0 / math.sqrt(input_length), 1.0 / math.sqrt(input_length)))
We2 = tf.Variable(tf.random_uniform((args.hidden_size, args.code_size), -1.0 / math.sqrt(args.hidden_size), 1.0 / math.sqrt(args.hidden_size)))

be1 = tf.Variable(tf.zeros([args.hidden_size]))
be2 = tf.Variable(tf.zeros([args.code_size]))

hidden_1 = tf.nn.tanh(tf.matmul(encoder_inputs, We1) + be1)
code = tf.nn.tanh(tf.matmul(hidden_1, We2) + be2)

# kernel on codes
code_K = tf.tensordot(code, tf.transpose(code), axes=1)

# decoder
if args.tied_weights:
    Wd1 = tf.transpose(We2)
    Wd2 = tf.transpose(We1)
else:
    Wd1 = tf.Variable(tf.random_uniform((args.code_size, args.hidden_size), -1.0 / math.sqrt(args.code_size), 1.0 / math.sqrt(args.code_size)))
    Wd2 = tf.Variable(tf.random_uniform((args.hidden_size, input_length), -1.0 / math.sqrt(args.hidden_size), 1.0 / math.sqrt(args.hidden_size)))
    
bd1 = tf.Variable(tf.zeros([args.hidden_size]))  
bd2 = tf.Variable(tf.zeros([input_length])) 

if args.lin_dec:
    hidden_2 = tf.matmul(code, Wd1) + bd1
else:
    hidden_2 = tf.nn.tanh(tf.matmul(code, Wd1) + bd1)

dec_out = tf.matmul(hidden_2, Wd2) + bd2

# ----- LOSS --------

# kernel alignment loss with normalized Frobenius norm
code_K_norm = code_K/tf.norm(code_K, ord='fro', axis=[-2,-1]) #tf.reduce_max(code_K)
prior_K_norm = prior_K/tf.norm(prior_K, ord='fro', axis=[-2,-1]) #tf.reduce_max(prior_K)
k_loss = tf.norm(code_K_norm - prior_K_norm, ord='fro', axis=[-2,-1])

# reconstruction loss    
parameters = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(args.learning_rate)
#reconstruct_loss = tf.losses.mean_squared_error( labels=tf.multiply(dec_out,missing_mask), predictions=tf.multiply(encoder_inputs,missing_mask) )
reconstruct_loss = tf.reduce_sum((dec_out*missing_mask - encoder_inputs*missing_mask)**2)/tf.reduce_sum(missing_mask) 

# L2 loss
reg_loss = 0
for tf_var in tf.trainable_variables():
    if not ("Bias" in tf_var.name or "noreg" in tf_var.name):
        reg_loss += tf.nn.l2_loss(tf_var)
        
tot_loss = reconstruct_loss + args.w_reg*reg_loss + args.a_reg*k_loss

# Calculate and clip gradients
gradients = tf.gradients(tot_loss, parameters)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, args.max_gradient_norm)
update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))

sess.run(tf.global_variables_initializer())

# trainable parameters count
total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print('Total parameters: {}'.format(total_parameters))

# ============= TENSORBOARD =============             
mean_grads = tf.reduce_mean([tf.reduce_mean(grad) for grad in gradients])
tf.summary.scalar('mean_grads', mean_grads)
tf.summary.scalar('reconstruct_loss', reconstruct_loss)
tvars = tf.trainable_variables()
for tvar in tvars:
    tf.summary.histogram(tvar.name.replace(':','_'), tvar)
merged_summary = tf.summary.merge_all()

# ================= TRAINING =================

# initialize training variables
time_tr_start = time.time()
batch_size = args.batch_size
max_batches = train_data.shape[0]//batch_size
loss_track = []
kloss_track = []
min_vs_loss = np.infty
model_name = "../models/kae_"+str(time.strftime("%Y%m%d-%H%M%S"))+".ckpt"
train_writer = tf.summary.FileWriter('../logs', graph=sess.graph)
saver = tf.train.Saver()

try:
    for ep in range(args.num_epochs):
        
        # shuffle training data
        idx = np.random.permutation(train_data.shape[0])
        train_data_s = train_data[idx,:] 
        K_tr_s = K_tr[idx,:][:,idx]
        M_train_s = M_train[idx,:]
        
        for batch in range(max_batches):
            
            fdtr = {encoder_inputs: train_data_s[(batch)*batch_size:(batch+1)*batch_size,:],
                    prior_K: K_tr_s[(batch)*batch_size:(batch+1)*batch_size, (batch)*batch_size:(batch+1)*batch_size],
                    missing_mask: M_train_s[(batch)*batch_size:(batch+1)*batch_size,:]
                    }           
            _,train_loss, train_kloss = sess.run([update_step, reconstruct_loss, k_loss], fdtr)    
            loss_track.append(train_loss)
            kloss_track.append(train_kloss)
            
        # check training progress on the validations set (in blood data valid=train) 
        if ep % 100 == 0:            
            print('Ep: {}'.format(ep))
            
            fdvs = {encoder_inputs: valid_data,
                    prior_K: K_vs,
                    missing_mask: M_valid}
            tot_loss_vs, reconstruct_loss_vs, k_loss_vs, code_K_vs, summary = sess.run([tot_loss, 
                                                                                        reconstruct_loss,
                                                                                        k_loss,
                                                                                        code_K,
                                                                                        merged_summary], fdvs)
            train_writer.add_summary(summary, ep)
            print('VS: tot_l=%.3f, rec_l=%.3f, k_l=%.3f -- TR: avg_rec_l=%.3f'%
                  (tot_loss_vs, reconstruct_loss_vs, k_loss_vs, np.mean(loss_track[-100:])))     
            
            # Save model yielding best results on validation
            if tot_loss_vs < min_vs_loss:
                min_vs_loss = tot_loss_vs
                tf.add_to_collection("encoder_inputs",encoder_inputs)
                tf.add_to_collection("dec_out",dec_out)
                tf.add_to_collection("reconstruct_loss",reconstruct_loss)
                save_path = saver.save(sess, model_name)
                                                    
except KeyboardInterrupt:
    print('training interrupted')

   
time_tr_end = time.time()
print('Tot training time: {}'.format((time_tr_end-time_tr_start)//60) )

# ================= TEST =================
print('************ TEST ************ \n>>restoring from: '+model_name+'<<')

# restore and evaluate graph
tf.reset_default_graph() # be sure that correct weights are loaded
saver.restore(sess, model_name)
tr_pred, tr_code = sess.run([dec_out, code], {encoder_inputs: train_data})
ts_pred, ts_code, ts_code_K = sess.run([dec_out, code, code_K], {encoder_inputs: test_data})
print('Test rec loss: %.3f'%(np.mean((ts_pred-test_data)**2)))

# reverse shape transformations
ts_pred = np.reshape(ts_pred, (test_data_shaped.shape[1], test_data_shaped.shape[0], test_data_shaped.shape[2]))
ts_pred = np.transpose(ts_pred,axes=[1,0,2])
test_data = test_data_shaped
tr_pred = np.reshape(tr_pred, (train_data_shaped.shape[1], train_data_shaped.shape[0], train_data_shaped.shape[2]))
tr_pred = np.transpose(tr_pred,axes=[1,0,2])
train_data = train_data_shaped

if np.min(train_len) < np.max(train_len) and args.interp_on:
    print('-- Reverse Interpolation --')
    ts_pred = interp_data(ts_pred, test_len, restore=True)

if plot_on:
    
    import matplotlib.pyplot as plt
    
    # plot the reconstruction of a random time series
    plot_idx1 = np.random.randint(low=0,high=test_data.shape[1])
    target_imp = test_data[:,plot_idx1,:]
    target_true = test_data_orig[:,plot_idx1,:]
    ts_out = ts_pred[:,plot_idx1,:]
    mask = 1-M_test[plot_idx1,:]
    plt.plot(target_imp.flatten()*mask, label='inp')
    plt.plot(target_true.flatten()*mask, label='true')
    plt.plot(ts_out.flatten()*mask, label='AE_pred')
    plt.legend(loc='best')
    plt.title('Prediction of a random MTS variable')
    plt.show()  
    np.savetxt('../logs/AE_pred',ts_out)
    
    import pylab    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(K_ts,cmap=pylab.cm.YlGnBu)  #'binary_r'
    fig.colorbar(cax)
    plt.title('Prior TCK kernel')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(ts_code_K,cmap=pylab.cm.YlGnBu)  #'binary_r'
    fig.colorbar(cax)
    plt.title('Codes inner products')
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()
    
    
    
# MSE and corr
test_mse, test_corr = mse_and_corr(ts_pred, test_data_orig, test_len)
print('TS: AE_pred vs orig: MSE=%.3f, Corr=%.3f'%(test_mse, test_corr))
test_mse2, test_corr2 = mse_and_corr(test_data, test_data_orig, test_len)
print('TS: inp vs orig: MSE=%.3f, Corr=%.3f'%(test_mse2, test_corr2))

train_mse, train_corr = mse_and_corr(tr_pred, train_data_orig, train_len)
print('TR: AE_pred vs orig: MSE=%.3f, Corr=%.3f'%(train_mse, train_corr))
train_mse2, train_corr2 = mse_and_corr(train_data, train_data_orig, train_len)
print('TR: inp vs orig: MSE=%.3f, Corr=%.3f'%(train_mse2, train_corr2))

# kNN classification on the codes
acc, f1 = classify_with_knn(tr_code, train_labels[:, 0], ts_code, test_labels[:, 0], k=3)
print('kNN -- acc: %.3f, F1: %.3f'%(acc, f1))


# dim reduction plots
if dim_red:
    dim_reduction_plot(ts_code, test_labels)

#train_writer.close()
sess.close()