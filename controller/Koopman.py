import numpy as np
import scipy.linalg as LA
from sklearn.metrics import mean_squared_error

class Koopman:
    """
    docstring
    """
    def __init__(self,N_basis = 2,forgetCoeff = 0.97):
        """
        initialization
        """
        self.nk = N_basis
        self.weighting = forgetCoeff

    # initialization (need 2 timesnap data)
    def initOperator(self,xy,u):
        """
        change: nk = nk - N
        data = [x0;u] half-half
        Koopman model:
        [psi_x1]  = [A, B] * [psi_x0]    
        [x0    ]    [C, 0]   [u0    ] 
        
        as starting from 1 data pair, thus there is no need to weight date
        the weighting process is contained in the recursive updates  
        """
        self.n = np.size(xy,1)
        self.m = np.size(u,1)
        self.nbasis = self.nk - self.n
        
        self.x = xy[:-1,:].T
        self.y = xy[1:,:].T
        self.u = u.T
        self.stdData = np.std(xy)
        
        self.Z = np.random.randn(self.n,self.nbasis)/self.stdData
        self.basis = lambda x:rff(x,self.Z) # where x = (n x time)
        self.psi_x = self.basis(self.x)
        self.psi_y = self.basis(self.y)
        
        Zeta = np.vstack((self.psi_x,self.u))
        self.Q = np.matmul(self.psi_y,Zeta.T)
        self.G = LA.inv(np.matmul(Zeta,Zeta.T))
        self.M = np.matmul(self.x,self.psi_x.T)
        self.P = LA.inv(np.matmul(self.psi_x,self.psi_x.T))
        self.AB = np.matmul(self.Q,self.G)
        self.A = self.AB[:,:self.nk]
        self.B = self.AB[:,self.nk:]
        self.C = np.matmul(self.M,self.P)
        operator = np.vstack((self.AB,np.hstack((self.C,np.zeros((self.n,self.m))))))
        
        return operator

    # weighted RLS update
    def updateOperator(self,state,action):
        """
        docstring

        """
        xx = self.y[:,-1].reshape(self.n,1)
        uu = action.reshape(self.m,1)
        yy = state.reshape(self.n,1)
        psi_xx = self.psi_y[:,-1].reshape(self.nk,1)
        psi_yy = self.basis(yy)
        delta = np.vstack((psi_xx,uu))
        self.u = np.hstack((self.u, uu))
        self.x = np.hstack((self.x, xx))
        self.y = np.hstack((self.y, yy))
        self.psi_x = np.hstack((self.psi_x, psi_xx))
        self.psi_y = np.hstack((self.psi_y, psi_yy))
        
        beta = 1/(1 + np.matmul(np.matmul(delta.T,self.G/self.weighting), delta))
        inside1 = np.matmul(self.G/self.weighting,delta)
        self.G = self.G/self.weighting - beta*np.matmul(inside1,inside1.T)
        gamma = 1/(1 + np.matmul(np.matmul(psi_xx.T,self.P/self.weighting), psi_xx))
        inside2 = np.matmul(self.P/self.weighting,psi_xx)
        self.P = self.P/self.weighting - gamma*np.matmul(inside2,inside2.T)

        innovation1 = psi_yy - np.matmul(self.AB,delta)
        self.AB = self.AB + beta*np.matmul(innovation1,inside1.T)
        self.A = self.AB[:,:self.nk]
        self.B = self.AB[:,self.nk:]
        innovation2 = xx - np.matmul(self.C,psi_xx)
        self.C = self.C + gamma*np.matmul(innovation2,inside2.T)
        operator = np.vstack((self.AB,np.hstack((self.C,np.zeros((self.n,self.m))))))
        
        return operator
    
    def predict(self,x0,u):
        lift = np.matmul(self.A,self.basis(x0.reshape(self.n,1))) + np.matmul(self.B,u.reshape(self.m,1))
        x_kp = np.matmul(self.C,lift)
        return x_kp

# RFF basis
def rff(X,Z):
    """
    X is a n x 1 vector
    return Psi as a Nk x t matrix
    """
    Nk = np.size(Z,1)
    Zc = Z[:,:int(Nk/2)]
    Zs = Z[:,int(Nk/2):]
    cos_psi = np.cos(np.matmul(Zc.T,X))
    sin_psi = np.sin(np.matmul(Zs.T,X))
    if np.size(cos_psi) == int(Nk/2):
        cos_psi = cos_psi.reshape(int(Nk/2),1)
    if np.size(sin_psi ) == int(Nk/2):
        sin_psi = sin_psi.reshape(int(Nk/2),1)
    # concecation: 
    if np.size(X) == 5:
        Psi = np.vstack((X.reshape(5,1),np.vstack((cos_psi,sin_psi))))  # Nk x t
    else:    
        Psi = np.vstack((X,np.vstack((cos_psi,sin_psi))))  # Nk x t
    return Psi

