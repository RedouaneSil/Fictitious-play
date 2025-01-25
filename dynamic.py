import numpy as np


"""
Define the drift, volatility and potential singular term of the SDE describing the representating agent dynamic.
"""
alpha=2 #parameter of the drift

def b_factory(Time,y):
    """
    Argument:
    m -- 2-array mean field term of size Time-State
    y -- Array. Auxiliary variable to compute the integrals with respect to m
    
    Returns:
    b -- drift associated to m
    """
    def b(t, state):
        """
        Argument:
        t -- Float. Time variable.
        x -- 2-dimensionnal array : State variable.
        
        Returns:
        b(t, x) -- Float.
        """
        
        x=state[0]
        j=state[1]
        
        i= np.where(Time[:, 0] == t)[0][0]
        return (q(t,x,j,y))
        # if x>y[-3,0] and j==0:
        #     return 0
        # elif x<y[2,0] and j==1:
        #     return 0
        # elif x<=y[-3,0] and j==0:
        #     return (1+x)**alpha
        # else:
        #     return -x
        # if x>y[-3,0]:
        #     return -1
        # else:
        #     return j*x
    return np.vectorize(b,signature='(),(n)->()')

def s_factory(Time,y):
    """
    Argument:
    m -- 2-array mean field term of size Time-State
    y -- Array. Auxiliary variable to compute the integrals with respect to m
    
    Returns:
    s -- volatility associated to m
    """
    def s(t, state):
        """
        Argument:
        t -- Float. Time variable.
        x -- Float. State variable.
        
        Returns:
        s(t, x) -- Float.
        """
        x=state[0]
        j=state[1]
        i= np.where(Time[:, 0] == t)[0][0]
        if x>y[-3,0]:
            return 0
        elif x==y[1,0]:
            return 0
        return 1
    return np.vectorize(s,signature='(),(n)->()')

def D_factory(Time):
    """
    Argument:
    m -- 2-array mean field term of size Time-State
    y -- Array. Auxiliary variable to compute the integrals with respect to m
    
    Returns:
    b -- drift associated to m
    """
    def D(t, state):
        """
        Argument:
        t -- Float. Time variable.
        x -- 2-dimensionnal array : State variable.
        
        Returns:
        D(t, x) -- Float.
        """
        x=state[0]
        j=state[1]
        i= np.where(Time[:, 0] == t)[0][0]
        if j==0:
            return 0
        return 0
    return np.vectorize(D,signature='(),(n)->()')

def q(t,x,j,y):
    """
    Argument:
    t -- Float. Time variable.
    x -- Float. State variable.
    j -- Integer. State variable.
    y -- State grid.
    
    Returns:
    q(t, x) -- Float.
    """
    if x>y[-3,0] and j==0:
        return 0
    elif x<y[2,0] and j==1:
        return 0
    elif x<=y[-3,0] and j==0:
        return (1+x)**alpha
    else:
        return -2*x
