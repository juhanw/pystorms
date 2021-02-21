import numpy as np
import scipy.linalg as sci_la
import cvxopt

class MPC:
    """
    docstring
    """
    def __init__(self, Uub=None, Ulb=None, Xub_soft=None, Xlb_soft=None, Xub_hard=None, Xlb_hard=None, num_horizon=5, Uslop=0.005, Usmooth=0.01,N = None):
        """
        X, U bounds have already been scaled down
        """
        self.nh = num_horizon
        dt = 1/self.nh
        self.m = np.size(Uub)
        if Uub is not None:
            self.Uub = Uub.reshape(self.m,1)
        else:
            self.Uub = None
        if Ulb is not None:
            self.Ulb = Ulb.reshape(self.m,1)
        else:
            self.Ulb = None
        if Uub is not None and Ulb is not None:
            self.Uslop = dt*Uslop*np.mean(self.Uub-self.Ulb)
            self.Usmooth = dt**2*Usmooth*np.mean(self.Uub-self.Ulb)
        else:
            self.Uslop = None
            self.Usmooth = None

        if Xub_soft is not None:
            self.n = np.size(Xub_soft)
            self.Xub_soft = Xub_soft.reshape(self.n,1)
            self.nslack = self.n*(self.nh+1)
        else:
            self.Xub_soft = None
            self.n = N
        if Xlb_soft is not None:
            self.Xlb_soft = Xlb_soft.reshape(self.n,1)
        else:
            self.Xlb_soft = None

        if Xub_hard is not None:  
            self.n = np.size(Xub_hard)
            self.Xub_hard = Xub_hard.reshape(self.n,1)
        else:
            self.Xub_hard = None
        if Xlb_hard is not None:
            self.Xlb_hard = Xlb_hard.reshape(self.n,1)
        else:
            self.Xlb_hard = None

    def set_cost(self, z0, ulast, A, B, C, q=10, qh=10, r=0.1):
        """
        Cost = U*H*U' + f'*U
        U = [ulast, u0, u1, ..., uh-1] -- h+1
        X = [x0, x1, x2, ..., xh] -- h+1
        Z = [z0, z1, z2, ..., zh] -- h+1
        X = C*Zh
        Z = Sz*z0 + Su*U
        """
        self.nk = np.size(A,0)
        self.Sz = np.kron(np.ones((self.nh,1)),A)
        self.Su = np.kron(np.eye(self.nh),B)
        for i in range(self.nh-1):
            self.Sz[(i+1)*self.nk:(i+2)*self.nk,:] = np.matmul(A,self.Sz[i*self.nk:(i+1)*self.nk,:])
            self.Su[(i+1)*self.nk:(i+2)*self.nk,:(i+1)*self.m] = np.matmul(A,self.Su[i*self.nk:(i+1)*self.nk,:(i+1)*self.m])
        self.Su = np.hstack([np.zeros((np.size(self.Su,0),self.m)),self.Su])
        self.Su = np.vstack([np.zeros((self.nk,np.size(self.Su,1))), self.Su])
        self.Sz = np.vstack([np.eye(self.nk),self.Sz])

        self.Q = np.kron(np.eye(self.nh),q*np.eye(self.n))
        self.Q = sci_la.block_diag(self.Q,qh*np.eye(self.n))
        self.R = np.kron(np.eye(self.nh+1),r*np.eye(self.m))
        # self.R = sci_la.block_diag(0*np.eye(self.m),self.R)
        self.Cbig = np.kron(np.eye(self.nh+1),C)
        CSu = np.matmul(self.Cbig,self.Su)
        self.H = np.matmul(CSu.T,np.matmul(self.Q,CSu)) + self.R
        calc_easy = np.matmul(self.Cbig,np.matmul(self.Sz,z0))
        self.f = 2*np.matmul(calc_easy.T,np.matmul(self.Q,CSu)) #row vector
        # self.f[:,:self.m] = np.zeros((1,self.m))

        if self.Xub_soft is not None:
            self.H = sci_la.block_diag(self.H,0*np.eye(self.nslack))
            self.f = np.hstack([self.f, 1e6*np.ones((1, self.nslack))])

    def set_constraints(self, z0, ulast, A, B, C):
        """
        Ulb < ui < Uub 
        Xlb < xi < Xub
        xi = zi[1:n,:]
        """
        # input constraints
        Au_unit = np.vstack([np.eye(self.m),-np.eye(self.m)])
        Au = np.kron(np.eye(self.nh+1),Au_unit)
        bu_unit = np.vstack([self.Uub,-self.Ulb])
        bu = np.kron(np.ones((self.nh+1,1)),bu_unit)
        
        # slop constraints: u1 - u0 < uslop
        Au_k0 = np.kron(np.eye(self.nh),-np.eye(self.m))
        Au_k0 = np.hstack([Au_k0,np.zeros((np.size(Au_k0,0),self.m))])
        Au_k1 = np.kron(np.eye(self.nh),np.eye(self.m))
        Au_k1 = np.hstack([np.zeros((np.size(Au_k1,0),self.m)),Au_k1])
        Au_slop_block = Au_k0 + Au_k1
        Au_slop = np.vstack([Au_slop_block, -Au_slop_block])
        bu_slop = np.kron(np.ones((np.size(Au_slop,0),1)), self.Uslop)

        # smooth constraints: (u2-u1) - (u1-u0) < usmooth
        Au_i0 = np.kron(np.eye(self.nh-1),np.eye(self.m))
        Au_i0 = np.hstack([Au_i0,np.zeros((np.size(Au_i0,0),2*self.m))])
        Au_i1 = np.kron(np.eye(self.nh-1),-2*np.eye(self.m))
        Au_i1 = np.hstack([np.zeros((np.size(Au_i1,0),self.m)),Au_i1,np.zeros((np.size(Au_i1,0),self.m))])
        Au_i2 = np.kron(np.eye(self.nh-1),np.eye(self.m))
        Au_i2 = np.hstack([np.zeros((np.size(Au_i2,0),2*self.m)),Au_i2])
        Au_smooth_block = Au_i0 + Au_i1 + Au_i2
        Au_smooth = np.vstack([Au_smooth_block,-Au_smooth_block])
        bu_smooth = np.kron(np.ones((np.size(Au_smooth,0),1)),self.Usmooth)

            # memory constraints: u0 = u0
        Au_mem = np.kron(np.zeros((2,self.nh+1)),0*np.eye(self.m))
        Au_mem[:,:self.m] = Au_unit
        bu_mem = np.vstack([ulast,-1*ulast])

        Au = np.vstack([Au, Au_slop, Au_smooth, Au_mem])        
        bu = np.vstack([bu, bu_slop, bu_smooth, bu_mem])

        if self.Xub_soft is not None:
            self.L = np.hstack([Au, np.zeros((np.size(Au,0),self.nslack))])
        else:
            self.L = Au
        self.w = bu

        # state constraints
        P_unit = np.vstack([np.eye(self.n,self.nk),-np.eye(self.n,self.nk)])
        P = np.kron(np.eye(self.nh+1),P_unit)
        Az_U = np.matmul(P,self.Su)
        Az_z0 = np.matmul(P,self.Sz)   

        if self.Xub_hard is not None:
            bz_unit = np.vstack([self.Xub_hard,-1*self.Xlb_hard])
            bz = np.kron(np.ones((self.nh+1,1)),bz_unit)
            self.L = np.vstack([self.L,Az_U])
            self.w = np.vstack([self.w,bz-np.matmul(Az_z0,z0)])

        if self.Xub_soft is not None:
            # [U; slack inequalities; positive slack]
            Aieq1 = self.L
            n_s = int(self.nslack/(self.nh+1))
            S_unit = np.vstack([-np.eye(n_s),-np.eye(n_s)])
            S = np.kron(np.eye(self.nh+1),S_unit)
            Aieq2 = np.hstack([Az_U, S])
            Aieq3 = np.hstack([np.zeros((self.nslack,self.m*(self.nh+1) )), -np.eye(self.nslack) ])
            bieq1 = self.w
            bsoft_unit = np.vstack([self.Xub_soft,-1*self.Xlb_soft])
            bsoft = np.kron(np.ones((self.nh+1,1)),bsoft_unit)
            bieq2 = bsoft - np.matmul(Az_z0,z0)
            bieq3 = np.zeros((self.nslack,1))
            self.L = np.vstack([Aieq1, Aieq2, Aieq3])
            self.w = np.vstack([bieq1, bieq2, bieq3])

    def getMPC(self, z0, ulast, A, B, C):
        """
        z0, ulast has already been scaled down
        u0 has not been scaled up 
        """
        ulast = ulast.reshape(self.m,1)
        self.set_cost(z0, ulast, A, B, C)
        self.set_constraints(z0, ulast, A, B, C)
        P = cvxopt.matrix(2*self.H)
        q = cvxopt.matrix(self.f.T)
        G = cvxopt.matrix(self.L)
        h = cvxopt.matrix(self.w)
        try:
            cvxopt.solvers.options['maxiters'] = 200  # default 100; https://cvxopt.org/userguide/coneprog.html#quadratic-programming
            cvxopt.solvers.options['abstol'] = 1e-7 # default
            cvxopt.solvers.options['reltol'] = 1e-6 
            cvxopt.solvers.options['feastol'] = 1e-7
            sol = cvxopt.solvers.qp(P,q,G,h)
            u0 = np.asarray(sol['x'][self.m:2*self.m])
        except:
            try:
                cvxopt.solvers.options['maxiters'] = 1000
                sol = cvxopt.solvers.qp(P,q,G,h)
                u0 = np.asarray(sol['x'][self.m:2*self.m])
            except:
                try:
                    cvxopt.solvers.options['abstol'] = 1e-5 
                    cvxopt.solvers.options['reltol'] = 1e-4 
                    cvxopt.solvers.options['feastol'] = 1e-5 
                    sol = cvxopt.solvers.qp(P,q,G,h)
                    u0 = np.asarray(sol['x'][self.m:2*self.m])
                except:
                    print("Error Constraints!")
                    u0 = None
        return u0
