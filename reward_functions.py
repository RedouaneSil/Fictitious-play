import numpy as np
from dynamic import q

"""
Describe the reward functions of the problem.
"""

beta=0.2 #parameter of the reward function f1
gamma=0.3 #parameter of the reward function f1 to explain a price impact
rho=1 #parameter of the reward function f2
def f1_factory(m, y, Time):
    """
    Argument:
    m -- 2-array mean field term of size Time-State
    y -- Array. Auxiliary variable to compute the integrals with respect to m
    Time -- Array. Time variable
    Returns:
    f -- instantaneous reward associated to m
    """
    def f1(t, state): 
        """
        Argument:
        t -- Float. Time variable.
        x -- Float. State variable.
        
        Returns:
        f[m](t, x) -- Float.
        """
        x=state[0]
        j=state[1]
        i= np.where(Time[:, 0] == t)[0][0]
        y0=y[0:y.shape[0]//2,0]
        y1=y[(y.shape[0]//2):,0]
        #print(m.shape)
        m_t = m[i, :,:]
        m_t_conc=np.concatenate((m_t[:,0],m_t[:,1]),axis=0) #concatenate the two states in m_t to align with the state
        mean = np.sum((np.concatenate((y0[1:-1],y1[1:-1])))*m_t_conc)
        mean_0 = np.sum((x-y0[1:-1])*m_t[:,0])
        mean_1 = np.sum((x-y1[1:-1])*m_t[:,1]) 
        #return(mean_1)+x+j
        return j*q(t,x,j,y)*(1-beta)*(P(t)-gamma*mean_1) + F(t)*q(t,x,j,y)
    return np.vectorize(f1,signature='(),(n)->()')

def f2_factory(Lambda, y, Time):
    """
    Argument:
    Lambda -- 2-array mean field term of size Time-State
    y -- Array. Auxiliary variable to compute the integrals with respect to m
    Time -- Array. Time variable
    Returns:
    f -- instantaneous reward associated to m
    """
    def f2(t, state): 
        """
        Argument:
        t -- Float. Time variable.
        x -- Float. State variable.
        
        Returns:
        f[m](t, x) -- Float.
        """
        x=state[0]
        j=state[1]
        i= np.where(Time[:, 0] == t)[0][0]
        Lambda_t = Lambda[i, :,:]
        Lambda_conc=np.concatenate((Lambda_t[:,0],Lambda_t[:,1]),axis=0) #concatenate the two states in lambda to align with the state
        mean = np.sum((y[:,0])*Lambda_conc)
        
        return rho*x*j*mean
    return np.vectorize(f2,signature='(),(n)->()')
def g_factory(m, y):
    """
    Argument:
    lam-- 2-array mean field term of size Time-State
    y -- Array. Auxiliary variable to compute the integrals with respect to mT
    Time -- Array. Time variable
    Returns:
    g -- final reward associated to lam
    """
    def g(state):
        """
        Argument:
        t -- Float. Time variable.
        x -- Float. State variable.
        
        Returns:
        g[m](T, x) -- Float.
        """
        x=state[0]
        j=state[1]
        #mean_t = np.sum((t-s)*m)
        return x
    return np.vectorize(g,signature='(n)->()')

def P(t):
    """
    Fish price
    Argument:
    t -- Float. Time variable.
    
    Returns:
    P(t) -- Float.
    """
    return 1 + 0.1*np.sin(2*np.pi*t/100)
def F(t):
    """
    Fuel price
    Argument:
    t -- Float. Time variable.
    
    Returns:
    P(t) -- Float.
    """
    return 1