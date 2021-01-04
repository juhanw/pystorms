import numpy as np
import scipy
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
    

    # def __init__(self,A, B, C, Q = None, R = None, Qh = None, ulin = None, qlin = None, 
    #             Ulb = None, Uub = None, Zub = None, Zlb = None, nh = 10, solver = "quadprog" ):
    def __init__(self,A, B, C, Q = None, R = None, Qh = None, ulin = None, qlin = None, 
                Ulb = None, Uub = None, Xub = None, Xlb = None, nh = 10, solver = "quadprog" ):
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
        self.Qbig = la.block_diag(*([Q]*(nh-1)), Qh)
        
        if R == None:
            R = 0.01*np.eye(self.m)
        self.Rbig = np.kron(np.eye(self.nh),R)
            
        # if qlin == None:
        #     qlin = np.zeros((self.n,1))
        # if np.size(qlin) == self.n:
        #     self.qlin = np.tile(qlin.reshape(self.n,1),(self.nh,1))
        # elif np.size(qlin) == self.n*(self.nh):
        #     self.qlin = qlin.reshape(self.n*(self.nh),1) 

        # if ulin == None:
        #     ulin = 0.01*np.zeros((self.m,1))
        # if np.size(ulin) == self.m:
        #     self.ulin = np.tile(ulin.reshape(self.m,1),(self.nh,1))
        # elif np.size(ulin) == self.m * self.nh:
        #     self.ulin = ulin.reshape(self.m*self.nh,1)

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
            self.bu = np.vstack((self.bu,1e9*np.ones((self.m,1))))
        self.bu = np.reshape(self.bu,(2*self.m,1))

        if np.size(Xlb) == self.n:
            self.bx = -1*Xlb
        else:
            self.bx = np.zeros((self.n,1))
        if np.size(Xub) == self.n:
            self.bx = np.vstack((self.bx,Xub))
        else:
            self.bx = np.vstack((self.bx,1e9*np.ones((self.n,1))))
        self.bx = np.reshape(self.bx,(2*self.n,1))

    def getMPC(self, z0,A,B,C,xr = None):

        self.Sz = np.eye(self.nk*(self.nh + 1), self.nk)
        self.Su = np.zeros((self.nk*(self.nh + 1), self.m*self.nh))
        for i in range(self.nh):
            self.Sz[(i + 1)*self.nk:(i + 2)*self.nk, :] = np.matmul(self.Sz[i*self.nk:(i + 1)*self.nk, :],A)
            self.Su[(i + 1)*self.nk:(i + 2)*self.nk, :] = np.matmul(A,self.Su[i*self.nk:(i + 1)*self.nk, :])
            self.Su[(i + 1)*self.nk:(i + 2)*self.nk, i*self.m:(i + 1)*self.m] = B
        self.Sz = self.Sz[self.nk:,:]
        self.Su = self.Su[self.nk:,:]
        
        self.Cbig = np.kron(np.eye(self.nh),C)
        self.H = np.matmul(np.matmul(self.Cbig,self.Su).T,np.matmul(self.Qbig,np.matmul(self.Cbig,self.Su))) + self.Rbig
        self.f = 2*np.matmul(np.matmul(self.Cbig,self.Sz).T,np.matmul(self.Qbig,np.matmul(self.Cbig,self.Su)))
        self.G = 2*np.matmul(np.matmul(self.Qbig,self.Cbig),self.Su) # for tracking

        self.Ax = -C
        self.Ax = np.vstack((self.Ax,C))
        self.F = np.kron(np.eye(self.nh),self.Au)
        self.E = np.kron(np.eye(self.nh),np.zeros((2*self.m,self.nk)))
        self.bin = np.kron(np.ones((self.nh,1)),self.bu)
        self.F = np.vstack((self.F, np.kron(np.eye(self.nh),np.zeros((2*self.n,self.m)))))
        self.E = np.vstack((self.E, np.kron(np.eye(self.nh),self.Ax)))
        self.bin = np.vstack((self.bin, np.kron(np.ones((self.nh,1)),self.bx)))

        # QP solver
        L = self.F+np.matmul(self.E,self.Su)
        M = np.matmul(self.E,self.Sz)
        P = cvxopt.matrix(2*self.H)
        q = cvxopt.matrix(np.matmul(z0.T,self.f))
        G = cvxopt.matrix(L)
        h = cvxopt.matrix(-np.matmul(M,z0)+self.bin)
        try:
            sol = cvxopt.solvers.qp(P,q,G,h)
            u0 = np.asarray(sol['x'][:self.m])
        except:
            print("Error Constraints!") # why there is no Ulb/Uub effect???
            u0 = np.zeros((self.m,1))
        
        return u0



