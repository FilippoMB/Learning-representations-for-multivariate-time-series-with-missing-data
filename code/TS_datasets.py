import numpy as np
from scipy.integrate import odeint
import scipy.io
from sklearn import preprocessing
import sys
from utils import ideal_kernel
from scipy import sparse
from scipy.sparse import linalg as slinalg

"""
Data manager for different time series datasets.
"""

# ========== SINUSOID ==========
def getSinusoids():
      
    while True:
        
        t = np.arange(0.0, 100.0, 0.5)
        a = np.random.rand()
        b = np.random.rand()
        sinusoid = np.sin(t*a+b)
        
        yield sinusoid
 
       
# ========== LORENTZ ==========
def getLorentz():

    # init cond
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0
    t = np.arange(0.0, 10.0, 0.05)
    
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
    r = 3.6
    
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
        TS_gen = getLM()
    else:
        sys.exit('Invalid time series generator name')   
                
    # training data
    train_data = np.asarray([next(TS_gen) for _ in range(tr_data_samples)])
    train_data = preprocessing.scale(train_data,axis=1) # standardize the data
    train_data = np.expand_dims(train_data,-1)
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    train_targets = train_data
    train_len = np.asarray([train_data.shape[0] for _ in range(train_data.shape[1])])
    K_tr = np.ones([tr_data_samples,tr_data_samples]) # just for compatibility
    train_labels = np.ones([tr_data_samples,1]) # just for compatibility
    
    # validation
    valid_data = np.asarray([next(TS_gen) for _ in range(vs_data_samples)])
    valid_data = preprocessing.scale(valid_data,axis=1) # standardize the data
    valid_data = np.expand_dims(valid_data,-1)
    valid_data = np.transpose(valid_data,axes=[1,0,2]) # time_major=True
    valid_targets = valid_data
    valid_len = np.asarray([valid_data.shape[0] for _ in range(valid_data.shape[1])])
    K_vs = np.ones([vs_data_samples,vs_data_samples]) # just for compatibility
    valid_labels = np.ones([vs_data_samples,1]) # just for compatibility
    
    # test data
    test_data = np.asarray([next(TS_gen) for _ in range(ts_data_samples)])
    test_data = preprocessing.scale(test_data,axis=1) # standardize the data
    test_data = np.expand_dims(test_data,-1)
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    test_targets = test_data
    test_len = np.asarray([test_data.shape[0] for _ in range(test_data.shape[1])])
    K_ts = np.ones([ts_data_samples,ts_data_samples]) # just for compatibility
    test_labels = np.ones([ts_data_samples,1]) # just for compatibility
        
    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts)


# ========== SINUSOIDS WITH RANDOM FREQ ==========
def getSins(min_len=100, max_len=101, n_var=1):
    np.random.seed(0)
    num_train_data = 200
    num_test_data = 1000
    train_data = np.zeros([max_len, num_train_data, n_var])
    test_data = np.zeros([max_len, num_test_data, n_var])
    
    train_len = np.zeros([num_train_data,],dtype=int)
    test_len = np.zeros([num_test_data,],dtype=int)
    
    for i in range(train_data.shape[1]):
        m = np.random.rand()
        b = np.random.rand()
        n = np.random.randint(min_len,high=max_len)
        for j in range(n_var):
            x_ij = np.sin(np.arange(0,n)*m+b)       
            train_data[:n,i,j] = x_ij
        train_len[i] = n
        
    for i in range(test_data.shape[1]):
        m = np.random.rand()
        b = np.random.rand()
        n = np.random.randint(min_len,high=max_len)
        for j in range(n_var):
            x_ij = np.sin(np.arange(0,n)*m+b)      
            test_data[:n,i,j] = x_ij
        test_len[i] = n
           
    valid_data = train_data
    valid_len = train_len
    
    train_labels = np.ones([train_data.shape[1],1])    
    valid_labels = train_labels
    test_labels = np.ones([test_data.shape[1],1])
        
    train_targets = train_data
    valid_targets = train_targets
    test_targets = test_data
    
    K_tr = np.ones([train_data.shape[1],train_data.shape[1]])
    K_vs = K_tr
    K_ts = np.ones([test_data.shape[1],test_data.shape[1]])
    
    np.random.seed(None)
    
    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts)    

# ========== MSO ==========    
def getMSO(min_len=20, max_len=100, n_var=1):
    num_train_data = 100
    num_test_data = 2000    
    
    tot_len = 50000
    t = np.arange(0,tot_len)
    X = np.sin(0.2*t) + np.sin(0.311*t) + np.sin(0.42*t)  + np.sin(0.51*t) + np.sin(0.63*t) #+ np.sin(0.74*t) + np.sin(0.81*t);
    X = (X - np.mean(X))/np.std(X)
       
    train_data = np.zeros([max_len, num_train_data, n_var])
    train_len = np.zeros([num_train_data,],dtype=int)   
    for i in range(train_data.shape[1]):
        start_idx = np.random.randint(0, high=tot_len-max_len)
        ts_len = np.random.randint(min_len, high=max_len+1)
        ts_i = X[start_idx:start_idx+ts_len]
        for j in range(n_var):
            ts_ij = np.concatenate((ts_i[j*10:],ts_i[:j*10]), axis=0)
            train_data[:ts_len,i,j] = ts_ij
         
        train_len[i] = ts_len
    
    test_data = np.zeros([max_len, num_test_data, n_var])    
    test_len = np.zeros([num_test_data,],dtype=int)
    for i in range(test_data.shape[1]):
        start_idx = np.random.randint(0, high=tot_len-max_len)
        ts_len = np.random.randint(min_len, high=max_len+1)
        ts_i = X[start_idx:start_idx+ts_len]
        for j in range(n_var):
            ts_ij = np.concatenate((ts_i[j*10:],ts_i[:j*10]), axis=0)
            test_data[:ts_len,i,j] = ts_ij
        test_len[i] = ts_len
       
    valid_data = train_data
    valid_len = train_len
    
    train_labels = np.ones([train_data.shape[1],1])    
    valid_labels = train_labels
    test_labels = np.ones([test_data.shape[1],1])
        
    train_targets = train_data
    valid_targets = train_targets
    test_targets = test_data
    
    K_tr = np.ones([train_data.shape[1],train_data.shape[1]])
    K_vs = K_tr
    K_ts = np.ones([test_data.shape[1],test_data.shape[1]])
    
    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts)  
    
# ========== ECG TS DATA ==========
def getECGData():
    datadir = '../dataset/ECG5000/ECG5000'
    train_data = np.loadtxt(datadir+'_TRAIN',delimiter=',')
    test_data = np.loadtxt(datadir+'_TEST',delimiter=',')
    
    # standardize the data
    train_data[:,1:] = preprocessing.scale(train_data[:,1:], axis=1) 
    test_data[:,1:] = preprocessing.scale(test_data[:,1:], axis=1) 

    train_data, test_data = np.expand_dims(train_data,-1), np.expand_dims(test_data,-1) 
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    train_data, train_labels = train_data[1:,:,:], train_data[0,:,:]
    test_data, test_labels = test_data[1:,:,:], test_data[0,:,:]
           
    train_len = np.asarray([train_data.shape[0] for _ in range(train_data.shape[1])])
    test_len = np.asarray([test_data.shape[0] for _ in range(test_data.shape[1])])
           
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

def getECGDataFull():
    ecg2_data = scipy.io.loadmat('../dataset/ECG2/ECG_full.mat')
    train_data = ecg2_data['X']
    train_labels = ecg2_data['Y']
    train_len = ecg2_data['X_len']
    test_data = ecg2_data['Xte']
    test_labels = ecg2_data['Yte']
    test_len = ecg2_data['Xte_len']
    
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
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)


# ========== JAP VOWELS ==========
def getJapDataFull():
    jap_data = scipy.io.loadmat('../dataset/JapaneseVowels/JAP_full.mat')
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
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)

def getJapDataMiss(kernel='TCK', inp='zero', miss=0):
    jap_data = scipy.io.loadmat('../dataset/JapaneseVowels/TCK_data_'+str(miss)+'.mat')
    
    # ------ train -------
    train_data = jap_data['X']
    train_data = np.transpose(train_data,axes=[1,0,2]) # time_major=True
    train_len = jap_data['X_len']
    
    if inp=='zero': # substitute NaN with 0
        train_data[np.isnan(train_data)] = 0 
    
    elif inp == 'last': # replace NaN with the last seen value
       import pandas as pd
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
    test_len = jap_data['Xte_len']
    
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
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)
    
# ========== Blood data (OCC) ==========
def getBlood(inp='last'):
    blood_data = scipy.io.loadmat('../dataset/Blood/Blood_LPS.mat')
    train_data = blood_data['X']
    train_labels = blood_data['Y']
    train_len = blood_data['X_len']
    test_data = blood_data['Xte']
    test_labels = blood_data['Yte']
    test_len = blood_data['Xte_len']
    K_tr = blood_data['Ktrtr']
    K_ts = blood_data['Ktete']
    K_tr = (K_tr-np.amin(K_tr))/(np.amax(K_tr)-np.amin(K_tr))
    K_ts = (K_ts-np.amin(K_ts))/(np.amax(K_ts)-np.amin(K_ts))
    
    # time_major=True
    train_data = np.transpose(train_data,axes=[1,0,2])
    test_data = np.transpose(test_data,axes=[1,0,2]) 
    
    if inp == 'zero':
        # zero imputation
        train_data[np.isnan(train_data)] = 0 
        test_data[np.isnan(test_data)] = 0 
    
    elif inp == 'last': # replace NaN with the last seen value
       import pandas as pd
       train_data0 = train_data[0,:,:]
       train_data0[np.isnan(train_data0)] = 0
       train_data[0,:,:] = train_data0
       for i in range(train_data.shape[1]):
           train_data_i = pd.DataFrame(train_data[:,i,:])
           train_data_i.fillna(method='ffill',inplace=True)  
           train_data[:,i,:] = train_data_i.values    
          
       test_data0 = test_data[0,:,:]
       test_data0[np.isnan(test_data0)] = 0
       test_data[0,:,:] = test_data0
       for i in range(test_data.shape[1]):
           test_data_i = pd.DataFrame(test_data[:,i,:])
           test_data_i.fillna(method='ffill',inplace=True)
           test_data[:,i,:] = test_data_i.values
    
    elif inp=='mean':
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        for i in range(train_data.shape[2]):
            train_data[:,:,i] = imp.fit_transform(train_data[:,:,i])
            test_data[:,:,i] = imp.transform(test_data[:,:,i])
    
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len  
    K_vs = K_tr
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data 
        
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)
    
    
# ========== AF data ==========
def getAF(inp='zero'):
    af_data = scipy.io.loadmat('../dataset/AF/TCK_AF_30_2v_50m.mat')
    train_data = af_data['X']
    train_labels = af_data['Y']
    train_len = af_data['X_len']
    test_data = af_data['Xte']
    test_labels = af_data['Yte']
    test_len = af_data['Xte_len']
    K_tr = af_data['Ktrtr']
    K_ts = af_data['Ktete']
    K_tr = (K_tr-np.amin(K_tr))/(np.amax(K_tr)-np.amin(K_tr))
    K_ts = (K_ts-np.amin(K_ts))/(np.amax(K_ts)-np.amin(K_ts))
    
    # add trailing singleton dim
    if len(train_data.shape) < 3:
        train_data = np.expand_dims(train_data, axis=-1)
        test_data = np.expand_dims(test_data, axis=-1)
    
    # time_major=True
    train_data = np.transpose(train_data,axes=[1,0,2])
    test_data = np.transpose(test_data,axes=[1,0,2]) 
    
    if inp == 'zero':
        # zero imputation
        train_data[np.isnan(train_data)] = 0 
        test_data[np.isnan(test_data)] = 0 
    
    elif inp == 'last': # replace NaN with the last seen value
       import pandas as pd
       train_data0 = train_data[0,:,:]
       train_data0[np.isnan(train_data0)] = 0
       train_data[0,:,:] = train_data0
       for i in range(train_data.shape[1]):
           train_data_i = pd.DataFrame(train_data[:,i,:])
           train_data_i.fillna(method='ffill',inplace=True)  
           train_data[:,i,:] = train_data_i.values    
          
       test_data0 = test_data[0,:,:]
       test_data0[np.isnan(test_data0)] = 0
       test_data[0,:,:] = test_data0
       for i in range(test_data.shape[1]):
           test_data_i = pd.DataFrame(test_data[:,i,:])
           test_data_i.fillna(method='ffill',inplace=True)
           test_data[:,i,:] = test_data_i.values
           
    elif inp=='mean':
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        for i in range(train_data.shape[2]):
            train_data[:,:,i] = imp.fit_transform(train_data[:,:,i])
            test_data[:,:,i] = imp.transform(test_data[:,:,i])
            
    # valid == train   
    valid_data = train_data
    valid_labels = train_labels
    valid_len = train_len  
    K_vs = K_tr
    
    # target outputs
    train_targets = train_data
    valid_targets = valid_data
    test_targets = test_data 
        
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)

    
# ========== CHAR FULL (vriable lengths) ==========
def getCharDataFull():
    char_data = scipy.io.loadmat('../dataset/CharacterTrajectories/CHAR_full.mat')
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
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)
    
# ========== WAFER FULL (vriable lengths) ==========
def getWafer():
    waf_data = scipy.io.loadmat('../dataset/Wafer/WAF_full.mat')
    train_data = waf_data['X']
    train_labels = waf_data['Y']
    train_len = waf_data['X_len']
    test_data = waf_data['Xte']
    test_labels = waf_data['Yte']
    test_len = waf_data['Xte_len']
    
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
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)    
    
# ========== LIBRAS ==========
def getLibras():
    lib_data = scipy.io.loadmat('../dataset/Libras/LIB_full.mat')
    
    train_data = lib_data['X']
    train_labels = lib_data['Y']
    train_len = lib_data['X_len']
    test_data = lib_data['Xte']
    test_labels = lib_data['Yte']
    test_len = lib_data['Xte_len']
    
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
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)    


# ========== ARABIC DIGITS ==========
def getArab():
    arab_data = scipy.io.loadmat('../dataset/Arabic/ARAB_full.mat')
    
    train_data = arab_data['X']
    train_labels = arab_data['Y']
    train_len = arab_data['X_len']
    test_data = arab_data['Xte']
    test_labels = arab_data['Yte']
    test_len = arab_data['Xte_len']
    
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
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)     
    

# ========== AUSLAN DIGITS ==========
def getAuslan():
    aus_data = scipy.io.loadmat('../dataset/AUSLAN/AUS_full.mat')
    
    train_data = aus_data['X']
    train_labels = aus_data['Y']
    train_len = aus_data['X_len']
    test_data = aus_data['Xte']
    test_labels = aus_data['Yte']
    test_len = aus_data['Xte_len']
    
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
    
    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
    
    return (train_data, train_labels, train_len[:,0], train_targets, K_tr,
        valid_data, valid_labels, valid_len[:,0], valid_targets, K_vs,
        test_data, test_labels, test_len[:,0], test_targets, K_ts)  
    

# ========== RANDOM ODE ==========    
    
def getODE(n_var=5):
    num_train_data = 100
    num_test_data = 250    
    np.random.seed(0)
    
    A = np.asarray(sparse.rand(n_var, n_var, density=0.5).todense())
    A[np.where(A > 0)] -= 0.5
    w,_ = slinalg.eigs(A, k=1, which='LM')
    A /= np.abs(w)/0.8
    t = np.linspace(0, 7, 100) 
           
    train_data = np.zeros([100, num_train_data, n_var])
    train_len = np.ones([num_train_data,],dtype=int)*100   
    for i in range(train_data.shape[1]):
        y0 = np.random.rand(n_var)
        train_data[:,i,:] = odeint(_state_fun, y0, t, args=(A,A))
    
    test_data = np.zeros([100, num_test_data, n_var])    
    test_len = np.ones([num_test_data,],dtype=int)*100  
    for i in range(test_data.shape[1]):
        y0 = np.random.rand(n_var)
        test_data[:,i,:] = odeint(_state_fun, y0, t, args=(A,A))
        
    for i in range(n_var):
        train_data[:,:,i] = preprocessing.scale(train_data[:,:,i],axis=0) # standardize the data
        test_data[:,:,i] = preprocessing.scale(test_data[:,:,i],axis=0) # standardize the data
       
    valid_data = train_data
    valid_len = train_len
    
    train_labels = np.ones([train_data.shape[1],1])    
    valid_labels = train_labels
    test_labels = np.ones([test_data.shape[1],1])
        
    train_targets = train_data
    valid_targets = train_targets
    test_targets = test_data
    
    K_tr = np.ones([train_data.shape[1],train_data.shape[1]])
    K_vs = K_tr
    K_ts = np.ones([test_data.shape[1],test_data.shape[1]])
    
    np.random.seed(None)
    
    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts) 
    
def getODE_mc():
    np.random.seed(0)
    num_train_data = 100
    num_test_data = 500 
    n_var = 5
    num_class = 1
    min_time_steps = 200
    max_time_steps = 201
    
    train_data = np.zeros([min_time_steps*num_class, num_train_data*num_class, n_var]) 
    test_data = np.zeros([min_time_steps*num_class, num_test_data*num_class, n_var]) 
    train_len = []
    test_len = []
    train_labels = []
    test_labels = []
    for c in range(num_class):
        
        convergence = False
        while (not convergence):
            A = np.asarray(sparse.rand(n_var, n_var, density=0.5).todense())
            A[np.where(A > 0)] -= 0.5
            try:
                w,_ = slinalg.eigs(A, k=1, which='LM')
                convergence = True                
            except:
                continue    
            A /= np.abs(w)/0.8
#        time_steps = min_time_steps*(c+1)
#        t = np.linspace(0, 7, time_steps)
        
        for i in range(num_train_data):
            time_steps = np.random.randint(min_time_steps,high=max_time_steps)
            t = np.linspace(0, time_steps/2, time_steps)
            train_len.append(time_steps)
            y0 = np.random.rand(n_var)
            train_data[:time_steps,c*num_train_data+i,:] = odeint(_state_fun, y0, t, args=(A,A))
            train_labels.append([c])

        for i in range(num_test_data):
            time_steps = np.random.randint(min_time_steps,high=max_time_steps)
            t = np.linspace(0, time_steps/2, time_steps)
            test_len.append(time_steps)
            y0 = np.random.rand(n_var)
            test_data[:time_steps,c*num_test_data+i,:] = odeint(_state_fun, y0, t, args=(A,A))
            test_labels.append([c])


    train_len = np.asarray(train_len)
    test_len = np.asarray(test_len)
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    
    train_targets = train_data
    test_targets = test_data

    valid_data = train_data
    valid_len = train_len
    valid_labels = train_labels
    valid_targets = train_targets

    K_tr = ideal_kernel(train_labels)
    K_vs = ideal_kernel(valid_labels)
    K_ts = ideal_kernel(test_labels)
#    K_tr = np.ones([train_data.shape[1],train_data.shape[1]])
#    K_vs = K_tr
#    K_ts = np.ones([test_data.shape[1],test_data.shape[1]])
    
    np.random.seed(None)
   
    return (train_data, train_labels, train_len, train_targets, K_tr,
            valid_data, valid_labels, valid_len, valid_targets, K_vs,
            test_data, test_labels, test_len, test_targets, K_ts) 
            
def _state_fun(y, t, A, A1):
#    y_d = A.dot(np.sin(y))
    y_d = A.dot(np.tanh(y))
#    y_d = A.dot(y)
    return y_d
