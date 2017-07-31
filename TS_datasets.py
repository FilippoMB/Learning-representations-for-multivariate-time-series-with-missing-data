import numpy as np
from scipy.integrate import odeint

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