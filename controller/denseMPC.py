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
            qlin = np.ones((self.n,1))
        if np.size(qlin) == self.n:
            self.qlin = np.tile(qlin.reshape(self.n,1),(self.nh,1))
        elif np.size(qlin) == self.n*(self.nh):
            self.qlin = qlin.reshape(self.m*(self.nh),1) 

        if ulin == None:
            ulin = 0.01*np.ones((self.m,1))
        if np.size(ulin) == self.m:
            self.ulin = np.tile(ulin.reshape(self.m,1),(self.nh,1))
        elif np.size(ulin) == self.m * self.nh:
            self.ulin = ulin.reshape(self.m*self.nh,1)

        # set MPC constraints
        if np.size(Ulb) == self.m:
            self.Ulb = np.tile(Ulb.reshape(self.m,1),(self.nh,1))
        elif np.size(Ulb) == self.m * self.nh:
            self.Ulb = Ulb.reshape(self.m*self.nh,1)

        if np.size(Uub) == self.m:
            self.Uub = np.tile(Uub.reshape(self.m,1),(self.nh,1))
        elif np.size(Uub) == self.m * self.nh:
            self.Uub = Uub.reshape(self.m*self.nh,1)

        if np.size(Zub) > 0:            
            if np.size(Zub) == self.nk:
                self.Zub = np.tile(Zub.reshape(self.nk,1),(self.nh,1))
            elif np.size(Zub) == self.nk * (self.nh):
                self.Zub = Zub.reshape(self.m*(self.nh),1)
        else:
            self.Zub = np.zeros((self.m*(self.nh),1))

        if np.size(Zlb) > 0:           
            if np.size(Zlb) == self.nk:
                self.Zlb = np.tile(Zlb.reshape(self.nk,1),(self.nh,1))
            elif np.size(Zlb) == self.nk * (self.nh):
                self.Zlb = Zlb.reshape(self.m*(self.nh ),1)
        else:
            self.Zlb = np.zeros((self.m*(self.nh ),1))

    def getMPC(self, z0,A,B,C,xr = None):
        # get the u0
        # QP_solver lib: https://pypi.org/project/qpsolvers/
        # LQR: https://github.com/i-abr/mpc-koopman/blob/master/cart_pole.ipynb

        # set MPC model matrices
        self.A = np.eye(self.nk*(self.nh + 1), self.nk)
        self.B = np.zeros((self.nk*(self.nh + 1), self.m*self.nh))
        for i in range(self.nh):
            self.A[(i + 1)*self.nk:(i + 2)*self.nk, :] = np.matmul(self.A[i*self.nk:(i + 1)*self.nk, :],A)
            self.B[(i + 1)*self.nk:(i + 2)*self.nk, :] = np.matmul(A,self.B[i*self.nk:(i + 1)*self.nk, :])
            self.B[(i + 1)*self.nk:(i + 2)*self.nk, i*self.m:(i + 1)*self.m] = B
        self.A = self.A[self.nk:,:]
        self.B = self.B[self.nk:,:]
        self.C = la.block_diag(*([C]*(self.nh)))
        if xr == None:
            xr = np.zeros((self.n*self.nh,1))

        # set MPC constraints matrices
        if np.size(self.Zub) > 0:
            self.Aieq = self.B
            self.bieq = self.Zub - np.matmul(self.A,z0) 
        if np.size(self.Zlb) > 0:
            self.Aieq = np.vstack((self.Aieq,-self.B))
            self.bieq = np.vstack((self.bieq,-self.Zlb + np.matmul(self.A,z0)))
        if np.size(self.Uub) > 0:
            self.Aieq = np.vstack((self.Aieq,np.eye(np.size(self.B,1))))
            self.bieq = np.vstack((self.bieq,self.Uub))
        if np.size(self.Ulb) > 0:
            self.Aieq = np.vstack((self.Aieq,-np.eye(np.size(self.B,1))))
            self.bieq = np.vstack((self.bieq,-self.Ulb))

        self.H = 2*((self.C@self.B).T @ self.Q @ (self.C@self.B) + self.R)
        self.H = (self.H + self.H.T)/2
        self.f = 2*(self.A.T@self.C.T@self.Q@self.C@self.B).T@z0 + (-2*self.Q@self.C@self.B).T@xr + self.ulin

        # u = qpsolvers.solve_qp(self.H, self.f, self.Aieq, self.bieq, lb = self.Ulb, ub = self.Uub, solver='quadprog')
        # u = qpsolvers.quadprog_solve_qp(self.H, self.f, self.Aieq, self.bieq)
        # u = qpsolvers.solve_qp(self.H, self.f, self.Aieq, self.bieq)
        P = cvxopt.matrix(2*self.H)
        q = cvxopt.matrix(self.f)
        G = cvxopt.matrix(self.Aieq)
        h = cvxopt.matrix(self.bieq)
        sol = cvxopt.solvers.qp(P,q,G,h)
        
        u0 = sol['x'][:self.m]
        return u0