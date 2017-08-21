import numpy as np
from scipy.integrate import odeint
import scipy.io
from sklearn import preprocessing
import sys
from utils import ideal_kernel
import pandas as pd

"""
Data manager for different time series datasets.
"""

# ========== SINUSOID ==========
def getSinusoids():
      
    while True:
        
        t = np.arange(0.0, 10.0, 0.1)
        f = np.random.rand()
        f = 1
        n = np.random.randn(t.shape[0])*0.2
        sinusoid = np.sin(2*np.pi*t*f) + n
        
        yield sinusoid
 
       
# ========== LORENTZ ==========
def getLorentz():

    # init cond
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    t = np.arange(0.0, 5.0, 0.05)
    
    def f(state, t):
      x, y, z = state  # unpack the state vector
      return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives
       
    while True:
#        state0 = np.random.randint(5,15,3)
        state0 = np.random.uniform(size=3,low=5.0,high=15.0)
        states = odeint(f, state0, t)
        
        yield states[:,0]
       

# ========== LOGISTIC MAP ==========
def getLM():
    
    n_iter = 201       # Number of iterations per point
    r = 3.5
    
    def logisticmap(x, r):
    
        return x * r * (1 - x)
       
    # Return nth iteration of logisticmap(x. r)
    def iterate(n, x, r):
    
        X = []
        for i in range(1,n):
            x = logisticmap(x, r)
            X.append(x)
        
        return np.asarray(X)
    
    while True:
        lm = iterate(n_iter,np.random.uniform(),r)
        
        yield lm


# ========== SYNTH TS DATA ==========
def getSynthData(tr_data_samples, vs_data_samples, ts_data_samples, name='Lorentz'):   

    if name == 'Lorentz':
        TS_gen = getLorentz()
    elif name == 'Sinusoids':
        TS_gen = getSinusoids()
    elif name == 'LM':
        TS_gen == getLM()
    else:
        sys.exit('Invalid time series generator name')   
                
    # training data
    train_data = np.asarray([next(TS_gen) for _ in range(tr_data_samples)])
    train_data = preprocessing.scale(train_data,axis=1) # standardize the data
    train_data = np.expand_dims(train_data,-1)
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    train_targets = train_data
    train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
    K_tr = np.ones([tr_data_samples,tr_data_samples]) # just for compatibility
    train_labels = np.ones([tr_data_samples,1]) # just for compatibility
    
    # validation
    valid_data = np.asarray([next(TS_gen) for _ in range(vs_data_samples)])
    valid_data = preprocessing.scale(valid_data,axis=1) # standardize the data
    valid_data = np.expand_dims(valid_data,-1)
    valid_data = np.transpose(valid_data,axes=[1,0,2]) # time_major=True
    valid_targets = valid_data
    valid_len = [valid_data.shape[0] for _ in range(valid_data.shape[1])]
    K_vs = np.ones([vs_data_samples,vs_data_samples]) # just for compatibility
    valid_labels = np.ones([vs_data_samples,1]) # just for compatibility
    
    # test data
    test_data = np.asarray([next(TS_gen) for _ in range(ts_data_samples)])
    test_data = preprocessing.scale(test_data,axis=1) # standardize the data
    test_data = np.expand_dims(test_data,-1)
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    test_targets = test_data
    test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
    K_ts = np.ones([ts_data_samples,ts_data_samples]) # just for compatibility
    test_labels = np.ones([ts_data_samples,1]) # just for compatibility
        
    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts)


# ========== ECG TS DATA ==========
def getECGData(tr_ratio = 0, rnd_order = False):
    datadir = 'ECG5000/ECG5000'
    train_data = np.loadtxt(datadir+'_TRAIN',delimiter=',')
    test_data = np.loadtxt(datadir+'_TEST',delimiter=',')
    
    # standardize the data
    train_data[:,1:] = preprocessing.scale(train_data[:,1:], axis=1) 
    test_data[:,1:] = preprocessing.scale(test_data[:,1:], axis=1) 

    if tr_ratio == 0: 
        train_data, test_data = np.expand_dims(test_data,-1), np.expand_dims(train_data,-1) # switch training and test
        train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
        test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
        train_data, train_labels = train_data[1:,:,:], train_data[0,:,:]
        test_data, test_labels = test_data[1:,:,:], test_data[0,:,:]
        
    else:
        data = np.concatenate((train_data,test_data),axis=0)
        data = np.expand_dims(data,-1)
        data = np.transpose(data,axes=[1,0,2]) # time_major=True
        
        # split
        num_ts = data.shape[1]
        ind_cut = int(tr_ratio*num_ts)
        if rnd_order:
            rnd_ind = np.random.permutation(num_ts)
        else:
            rnd_ind = np.arange(num_ts)
        train_data = data[1:,rnd_ind[:ind_cut],:]
        train_labels = data[0,rnd_ind[:ind_cut],:]
        test_data = data[1:,rnd_ind[ind_cut:],:]
        test_labels = data[0,rnd_ind[ind_cut:],:]
    
    train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
    test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
           
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data
    
    # kernel matrices
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
        
    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts)


# ========== JAP VOWELS DATA ==========
def getJapData(kernel='TCK', inp='zero'):
    jap_data = scipy.io.loadmat('JapaneseVowels/TCK_data.mat')
    
    # ------ train -------
    train_data = jap_data['X']
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
    
    if inp=='zero': # substitute NaN with 0
        train_data[np.isnan(train_data)] = 0 
    
    elif inp == 'last': # replace NaN with the last seen value
       train_data0 = train_data[0,:,:]
       train_data0[np.isnan(train_data0)] = 0
       train_data[0,:,:] = train_data0
       for i in range(train_data.shape[1]):
           train_data_i = pd.DataFrame(train_data[:,i,:])
           train_data_i.fillna(method='ffill',inplace=True)  
           train_data[:,i,:] = train_data_i.values
           
    elif inp=='mean':
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        for i in range(train_data.shape[2]):
            print(train_data[:,:,i])
            train_data[:,:,i] = imp.fit_transform(train_data[:,:,i])
            print(train_data[:,:,i])
           
    train_labels = np.asarray(jap_data['Y'])
    
    # ----- test -------
    test_data = jap_data['Xte'] 
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
    
    if inp == 'zero': # substitute NaN with 0
        test_data[np.isnan(test_data)] = 0 
    
    elif inp == 'last': # replace NaN with the last seen value
       test_data0 = test_data[0,:,:]
       test_data0[np.isnan(test_data0)] = 0
       test_data[0,:,:] = test_data0
       for i in range(test_data.shape[1]):
           test_data_i = pd.DataFrame(test_data[:,i,:])
           test_data_i.fillna(method='ffill',inplace=True)
           test_data[:,i,:] = test_data_i.values
    
    elif inp=='mean':
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        for i in range(test_data.shape[2]):
            test_data[:,:,i] = imp.fit_transform(test_data[:,:,i])
        
    test_labels = np.asarray(jap_data['Yte'])
                
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data    
    
    if kernel=='TCK':
        K_tr = jap_data['Ktrtr']
        K_vs = K_tr
        K_ts = jap_data['Ktete']
    else:
        K_tr = ideal_kernel(train_labels)
        K_vs = ideal_kernel(valid_labels)
        K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, K_ts)

# ========== JAP VOWELS FULL (vriable lengths) ==========
def getJapDataFull():
    jap_data = scipy.io.loadmat('JapaneseVowels/JAP_full.mat')
    train_data = jap_data['X']
    train_labels = jap_data['Y']
    train_len = jap_data['X_len']
    test_data = jap_data['Xte']
    test_labels = jap_data['Yte']
    test_len = jap_data['Xte_len']
    
#    train_data[train_data==0] = np.nan
#    test_data[test_data==0] = np.nan
    
    # time_major=True
    train_data = np.transpose(train_data,axes=[1,0,2])
    test_data = np.transpose(test_data,axes=[1,0,2]) 
    
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len  
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data 
    
    # TODO: add TCK/LPS kernel
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)
  
# ========== CHAR FULL (vriable lengths) ==========
def getCharDataFull():
    char_data = scipy.io.loadmat('CharacterTrajectories/CHAR_full.mat')
    train_data = char_data['X']
    train_labels = char_data['Y']
    train_len = char_data['X_len']
    test_data = char_data['Xte']
    test_labels = char_data['Yte']
    test_len = char_data['Xte_len']
    
    # time_major=True
    train_data = np.transpose(train_data,axes=[1,0,2])
    test_data = np.transpose(test_data,axes=[1,0,2]) 
    
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len  
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data 
    
    # TODO: add TCK/LPS kernel
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)
    
# ========== LIBRAS ==========
def getLibras():
    lib_data = scipy.io.loadmat('Libras/Libras_prep.mat')
    
    # ------ train -------
    train_data = lib_data['X']
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    train_len = [train_data.shape[0] for _ in range(train_data.shape[1])]
    train_labels = np.asarray(lib_data['Y'])
    
    # ----- test -------
    test_data = lib_data['Xte'] 
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    test_len = [test_data.shape[0] for _ in range(test_data.shape[1])]
    test_labels = np.asarray(lib_data['Yte'])
    
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data 
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
 
    return (train_data, train_labels, train_len, train_targets, K_tr,
        valid_data, valid_labels, valid_len, valid_targets, K_vs,
        test_data, test_labels, test_len, test_targets, K_ts)    
    
# ========== SYNTH VAR DATA ==========    
def getVarData():
    X = scipy.io.loadmat(file_name='../../data/VAR_data.mat')['x']
    X = preprocessing.scale(X,axis=1) # standardize the data