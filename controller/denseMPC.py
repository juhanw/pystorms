import numpy as np
import scipy.linalg as la
# import qpsolvers
import cvxopt

class denseMPC:
    """
    Dynamics:
    z^+ = A*z + B*u
    x   = C*z

    Cost:
    J = (x_N - xr_N)'*Q_N*(x_N - xr_N) + sum_{i=0:N-1} [ (x_i - xr_i)'*Q*(x_i - xr_i) + u_i'*R*u_i + rlin_i'u + qlin'*x ]

    Constraints:
    Aieq*z <= bieq

    Input:
    A, B, C                  -- model
    Q ,R, Qh, ulin, qlin     -- cost
    Ulb, Uub, Xlb, Xub       -- constraints
    nh, solver               -- MPC controller

    Output:
    u                        -- Koopman-based control

    """
    

    def __init__(self,A, B, C, Q = None, R = None, Qh = None, ulin = None, qlin = None, 
                Ulb = None, Uub = None, Zub = None, Zlb = None, nh = 10, solver = "quadprog" ):

        self.nh = nh
        self.nk = np.size(A,0)
        self.n = np.size(C,0)
        self.m = np.size(B,1)

        # Dense Form MPC
        # set MPC cost matrices (consider x0)
        if Q == None:
            Q = np.eye(self.n)
        
        if Qh == None:
            Qh = Q
        self.Q = la.block_diag(*([Q]*(nh-1)), Qh)

        if R == None:
            R = 0.01*np.eye(self.m)
        self.R = la.block_diag(*([R]*nh ))
            
        if qlin == None:
            qlin = np.zeros((self.n,1))
        if np.size(qlin) == self.n:
            self.qlin = np.tile(qlin.reshape(self.n,1),(self.nh,1))
        elif np.size(qlin) == self.n*(self.nh):
            self.qlin = qlin.reshape(self.n*(self.nh),1) 

        if ulin == None:
            ulin = 0.01*np.zeros((self.m,1))
        if np.size(ulin) == self.m:
            self.ulin = np.tile(ulin.reshape(self.m,1),(self.nh,1))
        elif np.size(ulin) == self.m * self.nh:
            self.ulin = ulin.reshape(self.m*self.nh,1)

        if np.size(Ulb) == self.m:
            self.Au = -np.eye(self.m)
            self.bu = -1*Ulb
        else:
            self.Au = np.zeros((self.m,self.m))
            self.bu = np.zeros((self.m,1))
            
        if np.size(Uub) == self.m:
            self.Au = np.vstack((self.Au,np.eye(self.m)))
            self.bu = np.vstack((self.bu,Uub))
        else:
            self.Au = np.vstack((self.Au,np.zeros((self.m,self.m)) ))
            self.bu = np.vstack((self.bu,np.zeros((self.m,1))))

        if np.size(Zlb) == self.nk:
            self.Az = -np.eye(self.nk)
            self.bz = -1*Zlb
        else:
            self.Az = np.zeros((self.nk,self.nk))
            self.bz = np.zeros((self.nk,1))
            
        if np.size(Zub) == self.nk:
            self.Az = np.vstack((self.Az,np.eye(self.nk)))
            self.bz = np.vstack((self.bz,Zub))
        else:
            self.Az = np.vstack((self.Az,np.zeros((self.nk,self.nk)) ))
            self.bz = np.vstack((self.bz,np.zeros((self.nk,1))))

    def getMPC(self, z0,A,B,C,xr = None):

        # set model matrices
        self.Sz = np.eye(self.nk*(self.nh + 1), self.nk)
        self.Su = np.zeros((self.nk*(self.nh + 1), self.m*self.nh))
        for i in range(self.nh):
            self.Sz[(i + 1)*self.nk:(i + 2)*self.nk, :] = np.matmul(self.Sz[i*self.nk:(i + 1)*self.nk, :],A)
            self.Su[(i + 1)*self.nk:(i + 2)*self.nk, :] = np.matmul(A,self.Su[i*self.nk:(i + 1)*self.nk, :])
            self.Su[(i + 1)*self.nk:(i + 2)*self.nk, i*self.m:(i + 1)*self.m] = B
        self.Sz = self.Sz[self.nk:,:]
        self.Su = self.Su[self.nk:,:]
        self.C = la.block_diag(*([C]*(self.nh)))
        if xr == None:
            xr = np.zeros((self.n*self.nh,1))

        # set MPC matrices

        # with substitutions ===================================================================================================
        Aieq_u = la.block_diag(*([self.Au]*self.nh))
        w0_u = np.tile(self.bu.reshape(2*self.m,1),(self.nh,1))
        E0_u = np.zeros((self.nh*2*self.m,self.nk))
        Aieq_z = la.block_diag(*([self.Az]*self.nh)) @ self.Su
        w0_z = np.tile(self.bz.reshape(2*self.nk,1),(self.nh,1))
        E0_z = la.block_diag(*([self.Az]*self.nh)) @ -self.Sz
        self.Aieq = np.vstack((Aieq_u,Aieq_z))
        self.w0 = np.vstack((w0_u,w0_z))
        self.E0 = np.vstack((E0_u,E0_z))
        self.bieq = self.w0 + self.E0 @ z0

        self.H = (self.C@self.Su).T @ self.Q @ (self.C@self.Su) + self.R
        self.H = (self.H + self.H.T)/2
        self.f = 2*(z0.T@self.Sz.T@self.C.T@self.Q@self.C@self.Su).T #+ (-2*self.Q@self.C@self.Su).T@xr + self.ulin

        # without substitutions ===================================================================================================
        Aieq_u = la.block_diag(*([self.Au]*self.nh))
        w0_u = np.tile(self.bu.reshape(2*self.m,1),(self.nh,1))
        E0_u = np.zeros((self.nh*2*self.m,self.nk))
        Aieq_z = la.block_diag(*([self.Az]*self.nh)) @ self.Su
        w0_z = np.tile(self.bz.reshape(2*self.nk,1),(self.nh,1))
        E0_z = la.block_diag(*([self.Az]*self.nh)) @ -self.Sz
        self.Aieq = np.vstack((Aieq_u,Aieq_z))
        self.w0 = np.vstack((w0_u,w0_z))
        self.E0 = np.vstack((E0_u,E0_z))
        self.bieq = self.w0 + self.E0 @ z0

        self.H = (self.C@self.Su).T @ self.Q @ (self.C@self.Su) + self.R
        self.H = (self.H + self.H.T)/2
        self.f = 2*(z0.T@self.Sz.T@self.C.T@self.Q@self.C@self.Su).T #+ (-2*self.Q@self.C@self.Su).T@xr + self.ulin

        # QP solver ===================================================================================================
        # u = qpsolvers.solve_qp(self.H, self.f, self.Aieq, self.bieq, lb = self.Ulb, ub = self.Uub, solver='quadprog')
        # u = qpsolvers.quadprog_solve_qp(self.H, self.f, self.Aieq, self.bieq)
        # u = qpsolvers.solve_qp(self.H, self.f, self.Aieq, self.bieq)
        P = cvxopt.matrix(2*self.H)
        q = cvxopt.matrix(self.f)
        G = cvxopt.matrix(self.Aieq)
        h = cvxopt.matrix(self.bieq)
        try:
            sol = cvxopt.solvers.qp(P,q,G,h)
            u0 = np.asarray(sol['x'][:self.m])
        except:
            print("Error Constraints!") # why there is no Ulb/Uub effect???
            u0 = np.zeros((self.m,1))
        
        return u0