import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csr_matrix,lil_matrix,hstack
from utils.matrix_verif import*
from classes.Linprog import Linprog
from classes.reward import Reward
from classes.constraints import Constraints
from reward_functions import *
"""
Class representing an optimization problem.
Attributes:
    - State_mesh_m (array-like): The state mesh grid for m.
    - Time_mesh_m (array-like): The time mesh grid for m.
    - State_mesh_lam (array-like): The state mesh grid for lambda.
    - Time_mesh_lam (array-like): The time mesh grid for lambda.
    - m_tilde_0 (array-like): Initial guess for the measure m_tilde.
    - lam_tilde_0 (array-like): Initial guess for the measure lambda_tilde.
    - constraints (Constraints): An object of the Constraints class.
    - reward (Reward): An object of the Reward class.
    - tol (float): The tolerance for the error.
    - N_iter (int): The maximum number of iterations.
    - f1_factory (function): A function that returns the vectorised f1.
    - f2_factory (function): A function that returns vectorised f2.
    - g_factory (function): A function that returns vectorised g.
Methods:
    compute_optimal_solution():
        Calculates the optimal solution and returns a vector z_opt as a numpy array using the class Linprog.
    evaluate_error(m_bar, lam_bar, m_hat, lam_hat):
        Evaluates the error between the current and previous measures.
"""

class Problem:
    def __init__(self, State_mesh_m,Time_mesh_m,State_mesh_lam,Time_mesh_lam,m_tilde_0, lam_tilde_0,State,Time ,Delta, delta, constraints, reward,tol,N_iter,f1_factory,f2_factory,g_factory, error='exploitability'):
        self.State_mesh_m = State_mesh_m
        self.Time_mesh_m = Time_mesh_m
        self.State_mesh_lam = State_mesh_lam
        self.Time_mesh_lam = Time_mesh_lam
        self.m_tilde_0 = m_tilde_0
        self.lam_tilde_0 = lam_tilde_0
        self.constraints = constraints
        self.reward = reward
        self.A_eq = constraints.A_eq
        self.b_eq = constraints.b_eq
        self.State = State
        self.Time = Time
        self.Delta = Delta
        self.delta = delta
        self.tol = tol
        self.N_iter = N_iter
        self.f1_factory = f1_factory
        self.f2_factory = f2_factory
        self.g_factory = g_factory
        self.error = error
        
        # Compute dimensions based on input shapes
        self.d_t = Time.shape[0]  # Number of time steps
        self.d_s = State.shape[0] // 2  # Half the number of states
        self.d_j = State.shape[1]  # Dimensionality of the state space
        self.State0 = State[:self.d_s]
        self.State1 = State[self.d_s:]
        # Initialize attributes for A_eq, b_eq, and C1, C2, C3
        self.value, self.lam_bar, self.m_bar, self.lam_hat, self.m_hat, self.eps_array=None,None,None,None,None,None

    def compute_optimal_solution(self):
        """
        Compute the optimal solution by iterating over the best response.
        """
        m_bar = self.m_tilde_0
        lam_bar = self.lam_tilde_0
        #print('self.reward.c',self.reward.c)
        linprog=Linprog(self.reward.c, self.A_eq, self.b_eq, self.d_s, self.d_t, self.d_j)
        value, lam_hat, m_hat = linprog.opt_value, linprog.lam_sol, linprog.m_sol
        
        m_bar = m_hat
        lam_bar = lam_hat

        n = 1
        
        eps = self.tol + 1 
        
        eps_list = []
        
        message = "Iteration: {:4d}; Error:  {:.6f}"
        
        while (np.abs(eps) > self.tol) and (n <= self.N_iter):
                
            f1_n = self.f1_factory(m_bar, self.State, self.Time)
            f2_n = self.f2_factory(lam_bar, self.State, self.Time)
            g_n = self.g_factory(m_bar, self.State)
            #print('self.reward.c',self.reward.c)
            #print(m_bar)
            self.reward.update_c(f1_n, f2_n, g_n)
            #print('self.reward.c',self.reward.c)
            linprog.update_optimal_solution(self.reward.c)

            value, lam_hat, m_hat = linprog.opt_value, linprog.lam_sol, linprog.m_sol
            #print('self.reward.c',self.reward.c)
            eps = self.evaluate_error(m_bar, lam_bar, m_hat, lam_hat)
            
            
            eps_list.append(eps)
            
            lam_bar = n/(n+1)*lam_bar + 1/(n+1)*lam_hat
            m_bar = n/(n+1)*m_bar + 1/(n+1)*m_hat
            
            print(message.format(n, np.abs(eps)))
            
            n += 1

        eps_array = np.array(eps_list)
        self.value, self.lam_bar, self.m_bar, self.lam_hat, self.m_hat, self.eps_array = value, lam_bar, m_bar, lam_hat, m_hat, eps_array
        return value, lam_bar, m_bar, lam_hat, m_hat, eps_array
        
    def evaluate_error(self, m_bar, lam_bar, m_hat, lam_hat):
            """
            Evaluate the error between the current and previous measures.
            Parameters:
            - m_bar (np.ndarray): The previous measure.
            - lam_bar (np.ndarray): The previous measure.
            - m_hat (np.ndarray): The current measure.
            - lam_hat (np.ndarray): The current measure.
            Returns:
            - eps (float): The error value.
            """
            if self.error == 'exploitability':
                m_bminushat=(m_hat-m_bar).reshape(m_hat.shape[0],m_hat.shape[1]*m_hat.shape[2])
                lam_bminushat=(lam_hat-lam_bar).reshape(lam_hat.shape[0],lam_hat.shape[1]*lam_hat.shape[2])
                
                eps = self.Delta*np.sum(self.reward.f1(self.Time_mesh_m, self.State_mesh_m)[:-1,:] * m_bminushat[:-1,:]) + \
                    np.sum(self.reward.g(np.concatenate((self.State0[1:-1],self.State1[1:-1]))) * m_bminushat[-1,:])+ \
                    self.Delta*np.sum(self.reward.f2(self.Time_mesh_lam, self.State_mesh_lam)*lam_bminushat) 
            else:
                
                eps = np.maximum(self.Delta*np.sum(np.abs(m_hat - m_bar)), self.Delta*np.sum(np.abs(lam_hat - lam_bar)))
            return eps