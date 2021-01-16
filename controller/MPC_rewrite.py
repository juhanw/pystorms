import numpy as np
import scipy.linalg as sci_la
import cvxopt

class MPC:
    """
    docstring
    """
    def __init__(self, Xub, Xlb, Uub, Ulb, num_horizon=10, Uslop=0.05, Usmooth=0.1):
        """
        X, U bounds have already been scaled down
        """
        dt = 1
        self.nh = num_horizon
        self.n = np.size(Xub)
        self.m = np.size(Uub)
        self.Xub = Xub.reshape(self.n,1)
        self.Xlb = Xlb.reshape(self.n,1)
        self.Uub = Uub.reshape(self.m,1)
        self.Ulb = Ulb.reshape(self.m,1)
        self.Uslop = Uslop*np.mean(self.Uub-self.Ulb)
        self.Usmooth = dt**2*Usmooth*np.mean(self.Uub-self.Ulb)
    

    def set_cost(self, z0, A, B, C, q=1, qh=1, r=0.1):
        """
        Cost = U*H*U' + f'*U
        U = [u0, u1, ..., uh-1]
        X = [x1, x2, ..., xh]
        Z = [z0, z1, ..., zh-1]
        Zh = [z1, z2, ..., zh]
        X = C*Zh
        Zh = Sz*z0 + Su*U
        """
        self.nk = np.size(A,0)
        self.Sz = np.kron(np.ones((self.nh,1)),A)
        self.Su = np.kron(np.eye(self.nh),B)

        for i in range(self.nh-1):
            self.Sz[(i+1)*self.nk:(i+2)*self.nk,:] = np.matmul(A,self.Sz[i*self.nk:(i+1)*self.nk,:])
            self.Su[(i+1)*self.nk:(i+2)*self.nk,:(i+1)*self.m] = np.matmul(A,self.Su[i*self.nk:(i+1)*self.nk,:(i+1)*self.m])
        self.Q = np.kron(np.eye(self.nh-1),q*np.eye(self.n))
        self.Q = sci_la.block_diag(self.Q,qh*np.eye(self.n))
        self.R = np.kron(np.eye(self.nh),r*np.eye(self.m))
        self.Cbig = np.kron(np.eye(self.nh),C)
        CSu = np.matmul(self.Cbig,self.Su)
        self.H = np.matmul(CSu.T,np.matmul(self.Q,CSu)) + self.R
        calc_easy = np.matmul(self.Cbig,np.matmul(self.Sz,z0))
        self.f = 2*np.matmul(calc_easy.T,np.matmul(self.Q,CSu))

    def set_constraints(self, z0, A, B, C):
        """
        Ulb < ui < Uub 
        Xlb < xi < Xub
        xi = zi[1:n,:]
        """
        # input constraints
        Au_unit = np.vstack([np.eye(self.m),-np.eye(self.m)])
        Au = np.kron(np.eye(self.nh),Au_unit)
        bu_unit = np.vstack([self.Uub,-self.Ulb])
        bu = np.kron(np.ones((self.nh,1)),bu_unit)
        
        # slop constraints: u1 - u0 < uslop
        Au_km = np.kron(np.eye(self.nh-1),-np.eye(self.m))
        Au_km = np.hstack([Au_km,np.zeros((np.size(Au_km,0),self.m))])
        Au_k = np.kron(np.eye(self.nh-1),np.eye(self.m))
        Au_k = np.hstack([np.zeros((np.size(Au_k,0),self.m)),Au_k])
        Au_slop_block = Au_km + Au_k
        Au_slop = np.vstack([Au_slop_block, -Au_slop_block])
        bu_slop = np.kron(np.ones((np.size(Au_slop,0),1)), self.Uslop)

        # smooth constraints: (u2-u1) - (u1-u0) < usmooth
        Au_i0 = np.kron(np.eye(self.nh-2),np.eye(self.m))
        Au_i0 = np.hstack([Au_i0,np.zeros((np.size(Au_i0,0),2*self.m))])
        Au_i1 = np.kron(np.eye(self.nh-2),-2*np.eye(self.m))
        Au_i1 = np.hstack([np.zeros((np.size(Au_i1,0),self.m)),Au_i1,np.zeros((np.size(Au_i1,0),self.m))])
        Au_i2 = np.kron(np.eye(self.nh-2),np.eye(self.m))
        Au_i2 = np.hstack([np.zeros((np.size(Au_i2,0),2*self.m)),Au_i2])
        Au_smooth_block = Au_i0 + Au_i1 + Au_i2
        Au_smooth = np.vstack([Au_smooth_block,-Au_smooth_block])
        bu_smooth = np.kron(np.ones((np.size(Au_smooth,0),1)),self.Usmooth)

        Au = np.vstack([Au, Au_slop, Au_smooth])
        bu = np.vstack([bu, bu_slop, bu_smooth])

        # state constraints
        Az_unit = np.vstack([np.eye(self.n,self.nk),-np.eye(self.n,self.nk)])
        Az = np.kron(np.eye(self.nh),Az_unit)
        bz_unit = np.vstack([self.Xub,-self.Xlb])
        bz = np.kron(np.ones((self.nh,1)),bz_unit)

        Az_U = np.matmul(Az,self.Su)
        Az_z0 = np.matmul(Az,self.Sz)
        self.L = np.vstack([Au,Az_U])
        self.w = np.vstack([bu,bz-np.matmul(Az_z0,z0)])

    def getMPC(self, z0, A, B, C):
        """
        z0 has already been scaled down
        u0 has not been scaled up 
        """
        self.set_cost(z0, A, B, C)
        self.set_constraints(z0, A, B, C)
        P = cvxopt.matrix(2*self.H)
        q = cvxopt.matrix(self.f.T)
        G = cvxopt.matrix(self.L)
        h = cvxopt.matrix(self.w)
        try:
            sol = cvxopt.solvers.qp(P,q,G,h)
            u0 = np.asarray(sol['x'][:self.m])
        except:
            try:
                opts = {'feastol',1e-5}
                sol = cvxopt.solvers.qp(P,q,G,h,options = opts)
                u0 = np.asarray(sol['x'][:self.m])
            except:
                print("Error Constraints!") # why there is no Ulb/Uub effect???
                u0 = None
        return u0
