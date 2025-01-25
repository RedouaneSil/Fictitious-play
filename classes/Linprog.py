import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.optimize import linprog
from utils.matrix_verif import *

"""
    Initialize the Linprog class with the given parameters.

    Attributes:
    - c (array-like): Coefficients for the linear objective function.
    - A_eq (array-like) : Coefficient matrix for the equality constraints.
    - b_eq (array-like): Right-hand side vector for the equality constraints.
    - d_s (int) 
    - d_t (int) 
    - d_j (int) 
"""
class Linprog:
    def __init__(self, c, A_eq, b_eq, d_s, d_t, d_j):
        
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.d_s = d_s
        self.d_t = d_t
        self.d_j = d_j
        self.opt_value,self.lam_sol, self.m_sol = self.best_response()
        
    def best_response(self):
        """
        Solve the best response problem using the gurobipy package, free academic license available.
        """
        model = gp.Model()
        model.Params.LogToConsole = 0
        rows, cols = len(self.b_eq), len(self.c)
        variables = []
        for j in range(cols):
            variables.append(model.addVar(lb=0, ub=1, obj=self.c[j, 0]))
        model.update()
        ##test for positive solutions
        #test_positive_solutions(A_eq=self.A_eq, b_eq=self.b_eq)    
        # iterate over the rows of A_eq adding each row into the model
        for i in range(rows):
            start = self.A_eq.indptr[i]
            end = self.A_eq.indptr[i+1]
            variables_row = [variables[j] for j in self.A_eq.indices[start:end]]
            coeff = self.A_eq.data[start:end]
            expr = gp.LinExpr(coeff, variables_row)
            model.addLConstr(lhs=expr, sense=GRB.EQUAL, rhs=self.b_eq[i, 0])
        
        model.update()
        model.ModelSense = -1
        model.optimize()
        #test_gurobipy_model(model,variables)
        
    
        z = [variables[j].X for j in range(cols)]
        
        z_m = np.array(z[:((self.d_t)*(self.d_s-2)*self.d_j)])
        z_lam = np.array(z[((self.d_t)*(self.d_s-2)*self.d_j):])
        
        lam_sol = z_lam.reshape((self.d_t, self.d_s,self.d_j))
        m_sol = z_m.reshape((self.d_t, (self.d_s-2),self.d_j))
                            
        return model.ObjVal, lam_sol, m_sol
    # def best_response_bis(self):
    #     def callback(optimize_result):
    #         print('x',optimize_result.x, optimize_result.fun,optimize_result.con)

    #     res = linprog(self.c, A_eq = self.A_eq, b_eq = self.b_eq, method ='interior-point',callback=callback,options={'disp': True})   
    #     #res = linprog(c, A_eq = A_eq, b_eq = b_eq, method ='highs',options={'disp': True}) 
    #     z = res.x
        
    #     print(res)
    #     print(res.success)
    #     print(res.message)
    #     print(res.status)
    #     print(np.linalg.norm(self.A_eq.dot(z)-self.b_eq.reshape(-1)))
    #     print('z',z.shape)
    #     print('z=',z)
    #     # z_m1=z[:d_t*d_s*d_j]
    #     # z_m2 = z[d_t*d_s*d_j:(d_t+1)*d_s*d_j]
    #     z_m=z[:(self.d_t)*(self.d_s-2)*self.d_j]
    #     z_lam = z[(self.d_t)*(self.d_s-2)*self.d_j:]
    #     print('z_m',z_m.shape)
    #     print('z_lam',z_lam.shape)
    #     lam_sol = z_lam.reshape((self.d_t, self.d_s,self.d_j))
    #     m_sol = z_m.reshape((self.d_t, (self.d_s-2),self.d_j))
        
    #     value = - res.fun 
    #     #print(m_sol)
    #     return value, lam_sol, m_sol
    def update_optimal_solution(self,c):
        """
        Update the procedure with the new c and compute the new optimal solution.
        """
        self.c=c
        self.opt_value,self.lam_sol, self.m_sol=self.best_response()
        return 0
