import numpy as np
from scipy import linalg
from scipy.linalg import null_space
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import lsq_linear

def detect_inconsistent_constraints(A_eq, b_eq):
    """
    Identifies inconsistent linear combinations of constraints in A_eq x = b_eq.

    Parameters:
        A_eq (numpy.ndarray): The equality constraint matrix.
        b_eq (numpy.ndarray): The equality constraint vector.

    Returns:
        inconsistencies (list): List of inconsistent linear combinations (vectors in the null space).
    """
    # Compute the null space of A_eq.T (linear dependence of rows)
    if(A_eq.shape[1]<A_eq.shape[0]):
        null_A = null_space(A_eq)
    else:
        null_A = null_space(A_eq.T)

    # If the null space is empty, there are no redundant constraints
    if null_A.size == 0:
        print("No linear dependencies in A_eq. System is consistent.")
        return []

    # Check which linear combinations of b_eq are inconsistent
    inconsistencies = []
    for z in null_A.T:  # Each column of null_A represents a linear combination
        inconsistency_value = z @ b_eq  # Check if this combination results in zero
        if not np.isclose(inconsistency_value, 0):
            inconsistencies.append((z, inconsistency_value))
    if inconsistencies:
        print("Inconsistent linear combinations detected:")
        for z, value in inconsistencies:
            print(f"Combination: {z}, Inconsistency value: {value}")
    else:
        print("No inconsistencies detected.")
    return inconsistencies
#

def solve_matrix_equation(A, b):
    """

    Finds a particular solution for the equation A x = b
    where A is a matrix and b is a vector with pseudo_inv.

    Parameters:
    - A : np.ndarray, matrix of size (m, n)
    - b : np.ndarray, vector of size (m, )

    Returns:
    - x : np.ndarray, particular solution
    """

    # Calcul de la pseudo-inverse de A
    A_pseudo_inv = np.linalg.pinv(A)
    
    # Solution particulière
    x = np.dot(A_pseudo_inv, b)
    b=b.reshape(-1,1)
    x=x.reshape(-1,1)
    
    return x


def test_gurobipy_model(model,variables):
    if model.Status == GRB.OPTIMAL:
        print("Optimal solution found!")
        solution = [var.X for var in variables]  
        print("Solution :", solution)
    else:
        raise ValueError(f"Optimisation error, status : {model.Status}")
    

def test_positive_solutions(A_eq,b_eq):
    b_copy=b_eq.reshape(-1)
    result = lsq_linear(A_eq.todense(), b_copy, bounds=(0, np.inf))
    print('rank A_eq',np.linalg.matrix_rank(A_eq.todense()))
    print('rank_A_eq,b_eq',np.linalg.matrix_rank(np.concatenate((A_eq.todense(),b_copy.reshape(-1,1)),axis=1)))
    
    if result.success:
        print("Atleast one solution found :")
        print("Solution :", result.x)
        print("Residuals :", A_eq @ result.x - b_copy)
        print("Residuals :", np.linalg.norm(A_eq @ result.x - b_copy))
    else:
        print("No solution found")
        raise ValueError("We must have atleast one positive solution")