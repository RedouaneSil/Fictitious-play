import numpy as np
from classes.reward import Reward
from classes.constraints import Constraints
from classes.problem import Problem
from reward_functions import *
from dynamic import *
from scipy.stats import norm
from config import initialize_config
from utils.matrix_verif import *
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lars

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import lasso_path
"""
Main script for solving a Mean Field Game (MFG) with singular controls in the case of optimal switching through a LPFP algorithm.
This script imports necessary classes and functions to calculate the optimal solution.
It uses the following components:
- Reward: Defines the reward structure for the game.
- Constraints: Specifies the constraints for the control problem.
- Problem: Represents the MFG problem setup.
- reward_functions: Contains reward function definitions.
- dynamic: Contains dynamic of the SDE describing the representating agent dynamic.
The script orchestrates the interaction between these components to solve the MFG problem.

All the initial parameters are defined in the config.py file.
"""
config = initialize_config()

Time = config["Time"]
Delta = config["Delta"]
State = config["State"]
delta = config["delta"]
State_mesh_m = config["State_mesh_m"]
Time_mesh_m = config["Time_mesh_m"]
Time_mesh_lam = config["Time_mesh_lam"]
State_mesh_lam = config["State_mesh_lam"]
combined_state0 = config["combined_state0"]
combined_state1 = config["combined_state1"]
m0 = config["m0"]
U = config["U"]
d_t = config["d_t"]
d_s = config["d_s"]
d_j = config["d_j"]
tol = config["tol"]
N_iter = config["N_iter"]
m_tilde_0 = config["m_tilde_0"]
lam_tilde_0 = config["lam_tilde_0"]
print('delta',delta)
print('Delta',Delta)
def initialize_reward_and_constraints():
    """
    Initialize the Reward and Constraints objects.
    """
    # Define the reward functions
    f1 = f1_factory(m_tilde_0,State,Time)
    f2 = f2_factory(lam_tilde_0,State,Time)
    g = g_factory(m_tilde_0,State)

    # Create a Reward object
    reward = Reward(State_mesh_m, Time_mesh_m, State_mesh_lam, Time_mesh_lam, State, Delta, delta, f1, f2, g)

    # Define the drift term
    b=b_factory(Time)
    # Define the diffusion term
    sigma=s_factory(Time)
    # Create a Constraints object
    constraints = Constraints(State_mesh_m, Time_mesh_m,State_mesh_lam, Time_mesh_lam,State,Time ,Delta, delta,U , m0, b, sigma)

    return reward, constraints


def tests():
    # Initialize the Reward and Constraints objects
    reward, constraints = initialize_reward_and_constraints()
    A_eq = constraints.A_eq
    b_eq = constraints.b_eq
    eigenvalues=np.linalg.eigvals(np.dot(A_eq,A_eq.T).todense())
    print(eigenvalues)
    c=reward.c
    b_eq = b_eq.reshape(-1)
    print('rank A_eq',np.linalg.matrix_rank(A_eq.todense()))
    print('rank_A_eq,b_eq',np.linalg.matrix_rank(np.concatenate((A_eq.todense(),b_eq.reshape(-1,1)),axis=1)))
    #test_positive_solutions(A_eq,b_eq)
    #scaler = StandardScaler()
    #A_eq_scaled = scaler.fit_transform(A_eq.todense())

    # Appliquer LARS classique
    
    # lars = LassoLars(fit_intercept=False,fit_path=True)  # fit_intercept=False car on impose A_eq z = b_eq
    # lars.fit(np.asarray(A_eq.todense()), b_eq)

    # Obtenir les coefficients
    # z = lars.coef_

    # print('coef_path',lars.coef_path_)
    # print('df',pd.DataFrame(lars.coef_path_))
    # Résultat
    lasso_Path=lasso_path(np.asarray(A_eq.todense()),b_eq,eps=1e-32,positive=True)
    lass_path=lasso_Path[1]
    #print('lasso_path',lass_path)
    print('lasso_path',pd.DataFrame(lass_path))
    #print("Solution z :", z)
    
    # Vérification : Produit matriciel pour voir si la contrainte est respectée
    # print("A_eq z :", A_eq @ z)
    # print("b_eq :", b_eq)
    return 0
tests()
