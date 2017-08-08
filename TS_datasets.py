import numpy as np
from scipy.integrate import odeint
from sklearn import preprocessing
import sys
import scipy

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
     
    #training data
    training_data = np.asarray([next(TS_gen) for _ in range(tr_data_samples)])
    training_data = preprocessing.scale(training_data,axis=1) # standardize the data
    training_data = np.expand_dims(training_data,-1)
    training_data = np.transpose(training_data,axes=[1,0,2]) # time_major=True
    training_targets = training_data
    
    # validation
    valid_data = np.asarray([next(TS_gen) for _ in range(vs_data_samples)])
    valid_data = preprocessing.scale(valid_data,axis=1) # standardize the data
    valid_data = np.expand_dims(valid_data,-1)
    valid_data = np.transpose(valid_data,axes=[1,0,2]) # time_major=True
    valid_targets = valid_data
    
    # test data
    test_data = np.asarray([next(TS_gen) for _ in range(ts_data_samples)])
    test_data = preprocessing.scale(test_data,axis=1) # standardize the data
    test_data = np.expand_dims(test_data,-1)
    test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
    test_targets = test_data
    
    return training_data, training_targets, valid_data, valid_targets, test_data, test_targets

# ========== ECG TS DATA ==========
def getECGData(tr_ratio = 0, rnd_order = False):
    datadir = 'ECG5000/ECG5000'
    training_data = np.loadtxt(datadir+'_TRAIN',delimiter=',')
    test_data = np.loadtxt(datadir+'_TEST',delimiter=',')

    if tr_ratio == 0: 
        training_data, test_data = np.expand_dims(test_data,-1), np.expand_dims(training_data,-1) # switch training and test
        training_data = np.transpose(training_data,axes=[1,0,2]) # time_major=True
        test_data = np.transpose(test_data,axes=[1,0,2]) # time_major=True
        training_data, training_labels = training_data[1:,:,:], training_data[0,:,:]
        test_data, test_labels = test_data[1:,:,:], test_data[0,:,:]
        
    else:
        data = np.concatenate((training_data,test_data),axis=0)
        data[:,1:] = preprocessing.scale(data[:,1:],axis=1)
        data = np.expand_dims(data,-1)
        data = np.transpose(data,axes=[1,0,2]) # time_major=True
        
        # split
        num_ts = data.shape[1]
        ind_cut = int(tr_ratio*num_ts)
        if rnd_order:
            rnd_ind = np.random.permutation(num_ts)
        else:
            rnd_ind = np.arange(num_ts)
        training_data = data[1:,rnd_ind[:ind_cut],:]
        training_labels = data[0,rnd_ind[:ind_cut],:]
        test_data = data[1:,rnd_ind[ind_cut:],:]
        test_labels = data[0,rnd_ind[ind_cut:],:]
    
    # valid == train   
    valid_data = training_data
    valid_labels = training_labels
        
    return training_data, training_labels, valid_data, valid_labels, test_data, test_labels
    
def getVarData():
    X = scipy.io.loadmat(file_name='../../data/VAR_data.mat')['x']
    X = preprocessing.scale(X,axis=1) # standardize the data