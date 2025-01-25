import numpy as np
from scipy.sparse import csr_matrix,lil_matrix,hstack
from utils.auxiliary_functions import*
from sympy import symbols
"""
A class used to build constraints for a given state and time mesh.
Attributes
----------
State_mesh_m : np.ndarray
    A grid for the state space for m.
Time_mesh_m : np.ndarray
    A grid for the time space for m.
Time_mesh_lam : np.ndarray
    A grid for the time space for Lambda.
State_mesh_lam : np.ndarray
    A grid for the state space for Lambda.
State : np.ndarray
    The state.
Time : np.ndarray
    The time.
Delta : float
    The time step.
delta : float
    The state step.
U : np.ndarray
    The basis of functions
m0 : np.ndarray
    Initial measure.
b : function
    A function representing the drift term.
sigma : function
    A function representing the diffusion term.
A_eq : np.ndarray
    A matrix to be constructed.
b_eq : np.ndarray
    A matrix to be constructed.
Methods
-------
build_A_t()
    Constructs the A_t matrix.
build_A_xu()
    Constructs the A_xu matrix.
build_A_xd()
    Constructs the A_xd matrix.
build_A_xx()
    Constructs the A_xx matrix.
build_B()
    Constructs the B matrix.
build_A_eq_m()
    Constructs the A_eq_m matrix.
build_A_eq_lam()
    Constructs the A_eq_lam matrix.
build_A_eq_mterm()
    Constructs the A_eq_mterm matrix.
build_b_eq()
    Constructs the b_eq vector.
"""

class Constraints:
    def __init__(self, State_mesh_m, Time_mesh_m,State_mesh_lam ,Time_mesh_lam,State,Time ,Delta, delta,U , m0, b, sigma):
        self.State_mesh_m = State_mesh_m
        self.Time_mesh_m = Time_mesh_m
        self.Time_mesh_lam=Time_mesh_lam
        self.State_mesh_lam = State_mesh_lam
        self.State=State
        self.Time=Time
        self.Delta=Delta
        self.delta=delta
        self.U=U
        self.m0 = m0
        self.b = b
        self.sigma = sigma
        # Compute dimensions based on input shapes
        self.d_t = Time.shape[0]  # Number of time steps
        self.d_s = State.shape[0] // 2  # Half the number of states
        self.d_j = State.shape[1]  # Dimensionality of the state space
        self.d_f = U.shape[0]  # Number of functions in the basis

        # Initialize attributes for A_eq, b_eq, and C1, C2, C3
        self.C1,self.C2=self.compute_b()
        self.C3=self.compute_C3()
        self.A_eq_m = self.build_A_eq_m()
        self.A_eq_lam = self.build_A_eq_lam()
        self.A_eq_mterm =self.build_A_eq_mterm() 
        self.A_eq = hstack((self.A_eq_m,self.A_eq_mterm,self.A_eq_lam))
        self.b_eq = self.build_b_eq()
        
        
    def compute_b(self):
        """
        Computes C1,C3 using the function b, State_mesh_m, and Time_mesh_m.

        Returns:
        - B: np.ndarray, calculated based on the b function and mesh grids
        """
        if not callable(self.b):
            raise ValueError("b must be a callable function")
        B=self.b(self.Time_mesh_m,self.State_mesh_m)
        B=B[:-1,:] 
        #print(B) 
        C1 = 1/self.delta * np.maximum(B, 0)
        C2 = 1/self.delta * np.minimum(B, 0)
        return C1,C2   
        
    def compute_C3(self):
        """
        Computes C3 using the function sigma, State_mesh_m, and Time_mesh_m.

        Returns:
        - C3: np.ndarray, calculated based on the sigma function and mesh grids
        """
        if not callable(self.sigma):
            raise ValueError("sigma must be a callable function")
        S=self.sigma(self.Time_mesh_m,self.State_mesh_m)
        S=S[:-1,:]
        #print(S)
        return (1/(self.delta**2)) *S**2/2
        
    def build_A_t(self,V_0,V_1):
        """
        Constructs the A_t matrix.
        - V_0 and V_1: arrays representing the value of functions on the grid.
        Returns:
        - A_t: np.ndarray, the constructed A_t matrix
        """
        A_t = 1/self.Delta *np.concatenate((V_0[1:,1:-1] - V_0[:-1,1:-1],V_1[1:,1:-1] - V_1[:-1,1:-1]),axis=1)
        return A_t

    def build_A_xu(self,V_0,V_1):
        """
        Constructs the A_xu matrix.
        - V_0 and V_1: arrays representing the value of functions on the grid.

        Returns:
        - A_xu: np.ndarray, the constructed A_xu matrix
        """
        A_xu = self.C1 * (np.concatenate((V_0[1:, 2:] - V_0[1:, 1:-1],V_1[1:, 2:] - V_1[1:, 1:-1]),axis=1))
        return A_xu
    

    def build_A_xd(self,V_0,V_1):
        """
        Constructs the A_xd matrix.
        - V_0 and V_1: arrays representing the value of functions on the grid.
        Returns:
        - A_xd: np.ndarray, the constructed A_xu matrix
        """
        A_xd = self.C2 * (np.concatenate((V_0[1:, 1:-1] - V_0[1:, :-2],V_1[1:, 1:-1] - V_1[1:, :-2]),axis=1))
        return A_xd

    def build_A_xx(self,V_0,V_1):
        """
        Constructs the A_xx matrix.
        - V_0 and V_1: arrays representing the value of functions on the grid.
        Returns:
        - A_xx: np.ndarray, the constructed A_xx matrix
        """
        A_xx= self.C3 * (np.concatenate((V_0[1:, 2:],V_1[1:, 2:]),axis=1)+np.concatenate((V_0[1:, :-2],V_1[1:, :-2]),axis=1)-2*np.concatenate((V_0[1:, 1:-1],V_1[1:, 1:-1]),axis=1))
        return A_xx

    def build_B(self,V_0,V_1):
        """
        Constructs the B matrix.
        - V_0 and V_1: arrays representing the value of functions on the grid.
        Returns:
        - B: np.ndarray, the constructed B matrix
        """
        B=np.concatenate((V_1-V_0,V_0-V_1),axis=1)
        return B

    def build_A_eq_m(self):
        """
        Constructs the A_eq matrix.
        Returns:    
        - A_eq_m: sparse matrix, the constructed A_eq_m matrix
        """
        A_eq_m = lil_matrix((self.d_f, (self.d_t-1) * (self.d_s-2)*self.d_j))
        for counter in range(self.d_f):
            V=self.U[counter, :].reshape((self.d_t,self.d_j*self.d_s))
            V_0, V_1 = self.build_V0V1(V)
            A_t=self.build_A_t(V_0,V_1)
            A_xu=self.build_A_xu(V_0,V_1)
            #print('counter',counter)
            
            A_xd=self.build_A_xd(V_0,V_1)
            A_xx=self.build_A_xx(V_0,V_1)
           # print('A_xx',A_xx)
            A=A_t + A_xu + A_xd + A_xx
            # print('A_t',A_t)
            # print('A_xu',A_xu)
            # print('A_xd',A_xd)
            A_eq_m[counter, :] = - self.Delta*A.reshape(1,-1)
        return A_eq_m.tocsr()
    
    def build_A_eq_lam(self):
        """
        Constructs the A_eq_lam matrix.
        Returns:    
        - A_eq_lam: sparse matrix, the constructed A_eq_lam matrix
        """
        A_eq_lam = lil_matrix((self.d_f, (self.d_t) * (self.d_s)*(self.d_j)))
        for counter in range(self.d_f):
            V=self.U[counter, :].reshape((self.d_t,self.d_j*self.d_s))
            V_0, V_1 = self.build_V0V1(V)
            B=self.build_B(V_0,V_1)
            A_eq_lam[counter,:]=-B.reshape(1,-1)
        return A_eq_lam.tocsr()
    def build_A_eq_mterm(self):
        """
        Constructs the A_eq_mterm matrix.
        Returns:    
        - A_eq_mterm: sparse matrix, the constructed mterm matrix
        """
        
        A_eq_mterm=np.concatenate((self.U[:,(-self.d_j*(self.d_s)+1):(-(self.d_s+1))],self.U[:,(-(self.d_s-1)):-1]),axis=1)
        return csr_matrix(A_eq_mterm)
    

    

    def build_b_eq(self):
        """
        Constructs the b_eq vector.
        Returns:
        - b_eq: np.ndarray, the constructed b_eq vector
        """
        b_eq = np.zeros((self.d_f, 1))
        for counter in range(self.d_f):
            V=self.U[counter, :].reshape((self.d_t,self.d_j*self.d_s))
            V_0, V_1 = self.build_V0V1(V)
            b_eq[counter,0]=np.dot(V_0[0, 1:-1].T, self.m0[1:-1,0])+np.dot(V_1[0, 1:-1].T, self.m0[1:-1,1])
        return b_eq
    def build_V0V1(self, V):
        """
        Splits the matrix V into its components V_0 and V_1.
        
        Parameters:
        - V: np.ndarray.

        Returns:
        - V_0, V_1: np.ndarray, two halves of the matrix V.
        """
        v_s = V.shape
        V_0 = V[:, :v_s[1] // 2]
        V_1 = V[:, v_s[1] // 2:]
        return V_0, V_1
