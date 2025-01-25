import numpy as np
from scipy.stats import norm
from utils.auxiliary_functions import construct_vector,generate_invertible_matrix
"""
This module contains the global variables for initializing the problem.
"""
class Config:
    """
    Initialize the Config class with the parameters of the problem.
    """ 
    ## Define the time horizon and the domain on which the state is defined.
    T = 1
    xmin=0
    xmax=10
    ## Number of points in the st_ate space and in the time space
    n_t =40
    n_s = 40
    #Tolerance and max number of iterations
    tol = 10**(-8)             # the iterations stop when error < tol
    N_iter = 200                # the iterations stop when they reach N_iter



def initialize_Time_and_State_space(Config):
    """
    Initialize the Time and State space.
    """
    T = Config.T
    n_t = Config.n_t
    n_s = Config.n_s
    ## Time space
    Time = np.linspace(0, T, num=n_t + 1)
    Time = np.reshape(Time, (Time.shape[0], 1))

    Delta = Time[1, 0] - Time[0, 0]
    ## State space
    State = np.linspace(Config.xmin, Config.xmax, num=n_s + 1)
    State = np.reshape(State, (State.shape[0], 1))

    delta = State[1, 0] - State[0, 0]
    # Create combined state with 0 and 1
    State_with_0 = np.hstack((State, np.zeros_like(State)))
    State_with_1 = np.hstack((State, np.ones_like(State)))
    combined_state = np.vstack((State_with_0, State_with_1))

    return Time,delta,Delta,State,combined_state




def intialize_m0(combined_state0,combined_state1,n_s,State):
    """
    Initialize the initial measure m0.
    """
    ## Define m0
    norm_0 = norm.pdf((State[1:-1]), np.mean(State[1:-1]),1)
    norm_0=np.column_stack((norm_0, norm_0))
    #norm_0 = norm.pdf((combined_state0[1:-1]), np.mean(combined_state0[1:-1]),1)
    #modified_n0=np.zeros((norm_0.shape[0]+2,   norm_0.shape[1]))
    modified_n0=np.zeros((norm_0.shape[0]+2,   norm_0.shape[1]))
    modified_n0[1:-1]=norm_0
    #modified_n0=norm_0

    modified_n0=modified_n0/np.sum(modified_n0)

    m0 = modified_n0
    return m0

def guess_lambda_tilde0_m_tilde0(combined_state,Time,combined_state0,combined_state1):
    """
    Initialize the initial guess for the measures lambda_tilde and m_tilde.
    """
    # m_tilde0
    normal_3 = norm.pdf(combined_state0[1:-1], np.mean(combined_state0[1:-1]), 4**(1/2))
    normal_3 = normal_3/np.sum(normal_3)


    Time_m_0, State_m_0 = np.meshgrid(Time[:,0], normal_3, indexing='ij') 
    State_m_0 = State_m_0.reshape(len(Time[:, 0]), -1, 2)
    Time_m_0=Time_m_0[:,:State_m_0.shape[1]//2]

    m_tilde_0 = State_m_0 
    m_tilde_0 /= m_tilde_0.sum(axis=(1, 2), keepdims=True)

    # Lambda_tilde_0
    normal_1 = norm.pdf(0.5*combined_state[:,0], np.mean(combined_state), 4**(1/2))

    normal_1 = normal_1#/np.sum(normal_1)

    Time_lam_0, State_lam_0 = np.meshgrid(Time[:,0], normal_1, indexing='ij') 

    State_lam_0 = State_lam_0.reshape(len(Time[:, 0]), -1, 2)
    Time_lam_0=Time_lam_0[:,:State_lam_0.shape[1]]

    lam_tilde_0 = State_lam_0 
    return lam_tilde_0,m_tilde_0

def initialize_basis_functions(d_t,d_s,d_j,f):
    """
    Initialize the basis functions.
    """
    U = np.identity((d_t*d_s*d_j))
    #diagonal=construct_vector(d_t,d_s,f)
    #U = np.diag(diagonal)
    #U=generate_invertible_matrix(d_j,d_t,d_s,1)
    return U

def f(t,x,j):
    """
    Function f(t,x,j) for the basis functions.
    """
    
    return np.ones_like(t)
def intialize_grids(Time,combined_state,combined_state0,combined_state1):
    """
    Initialize the mesh grids for lambda and m.
    """
    ## Mesh grids: we will use the following meshgrids for lambda and m
    Time_mesh_m, State_mesh_m = np.meshgrid(Time[:, 0], np.concatenate((combined_state0[1:-1],combined_state1[1:-1])), indexing='ij')
    State_mesh_m = State_mesh_m.reshape(len(Time[:, 0]), -1, 2)
    Time_mesh_m=Time_mesh_m[:,:State_mesh_m.shape[1]]
    Time_mesh_lam, State_mesh_lam = np.meshgrid(Time[:, 0],  np.concatenate((combined_state0,combined_state1)), indexing='ij')
    State_mesh_lam = State_mesh_lam.reshape(len(Time[:, 0]), -1, 2)
    Time_mesh_lam=Time_mesh_lam[:,:State_mesh_lam.shape[1]]

    return Time_mesh_m,State_mesh_m,Time_mesh_lam,State_mesh_lam

def initialize_config():
    """
    Initialize all the global variables.
    """
    config=Config()
    Time,delta,Delta,State,combined_state=initialize_Time_and_State_space(config)
    d_t = Time.shape[0]
    d_s = combined_state.shape[0]//2
    d_j= combined_state.shape[1]
    combined_state0=combined_state[:config.n_s+1]
    combined_state1=combined_state[config.n_s+1:]
    m0=intialize_m0(combined_state0,combined_state1,config.n_s,State)
    lam_tilde_0,m_tilde_0=guess_lambda_tilde0_m_tilde0(combined_state,Time,combined_state0,combined_state1)
    U=initialize_basis_functions(d_t,d_s,d_j,f)
    Time_mesh_m,State_mesh_m,Time_mesh_lam,State_mesh_lam=intialize_grids(Time,combined_state,combined_state0,combined_state1)
    return {
        "Time": Time, "Delta": Delta,
        "State": combined_state, "delta": delta,
        "combined_state0": combined_state0, "combined_state1": combined_state1,
        "Time_mesh_m": Time_mesh_m, "State_mesh_m": State_mesh_m,
        "Time_mesh_lam": Time_mesh_lam, "State_mesh_lam": State_mesh_lam,
        "m0": m0, "m_tilde_0": m_tilde_0, "lam_tilde_0": lam_tilde_0,
        "U": U, "tol": config.tol, "N_iter": config.N_iter, "d_t": d_t, "d_s": d_s, "d_j": d_j
    }


    