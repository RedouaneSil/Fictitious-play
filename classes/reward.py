import numpy as np
"""
A class to represent the reward structure in a fictitious play scenario with singular control.
Attributes
----------
State_mesh_m : array-like
    The state mesh grid for m.
Time_mesh_m : array-like
    The time mesh grid for m.
State_mesh_lam : array-like
    The state mesh grid for lambda.
Time_mesh_lam : array-like
    The time mesh grid for lambda.
State : array-like
    The state 
Delta : float
    The time step for m.
delta : float  
    The state step.
f1 : function
    A function that takes state and time as inputs and returns a value.
f2 : function
    A function that takes state and time as inputs and returns a value.
g : function
    A function that takes state and time as inputs and returns a value.

Methods
-------
build_F1():
    builds a numpy array F1 based on the function f1.
build_F2():
    builds a numpy array F2 based on the function g1.
build_G():
    builds a numpy array G based on the function g.
build_c():
    builds a numpy array c based on the sum of functions f1, g1, and g.
"""

class Reward:
    def __init__(self, State_mesh_m, Time_mesh_m, State_mesh_lam, Time_mesh_lam,State,Delta, delta , f1, f2, g):
        self.State_mesh_m = State_mesh_m
        self.Time_mesh_m = Time_mesh_m
        self.State_mesh_lam = State_mesh_lam
        self.Time_mesh_lam = Time_mesh_lam
        self.State=State
        self.f1 = f1
        self.f2 = f2
        self.g = g
        self.Delta=Delta
        self.delta=delta
        

        # Initialize c
        self.c=self.build_c()

    def build_F1(self):
        return self.f1(self.Time_mesh_m,self.State_mesh_m)

    def build_F2(self):
        return self.f2(self.Time_mesh_lam,self.State_mesh_lam)

    def build_G(self):
        d_s = self.State.shape[0]//2
        State0=self.State[0:d_s]
        State1=self.State[d_s:]
        return self.g(np.concatenate((State0[1:-1],State1[1:-1]),axis=0))

    def build_c(self):
        F1 = self.build_F1()
        F2 = self.build_F2()
        G = self.build_G()
        r_term = G.reshape(1,-1)
        r_m = self.Delta * F1[:-1,].reshape(1,-1)
        r_lam=self.Delta *F2.reshape(1,-1)
        c=-np.concatenate((r_m.T, r_term.T,r_lam.T), axis=0)
        #c=np.concatenate((r_m.T, r_term.T), axis=0)
        return c
    
    def update_c(self,f1,f2,g):
        self.f1=f1
        self.f2=f2
        self.g=g
        self.c=self.build_c()

        