import numpy as np
import scipy.linalg as LA
from sklearn.metrics import mean_squared_error

class Koopman:
    """
    docstring
    """
    def __init__(self,N_basis = 2,forgetCoeff = 1):
        """
        initialization
        """
        self.nk = N_basis
        self.weighting = forgetCoeff**2

    # initialization (need 2 timesnap data)
    def initOperator(self,data):
        """
        change: nk = nk - N
        data = [x0;u] half-half
        Koopman model:
        [psi_x1]  = [A, B] * [psi_x0]    
        [x0    ]    [C, 0]   [u0    ] 
        
        as starting from 1 data pair, thus there is no need to weight date
        the weighting process is contained in the recursive updates  
        """
        self.n = np.size(data[0,:])
        self.m = self.n
        self.nbasis = self.nk - self.n
        
        self.nt = np.size(data,0)
        self.x = data[:int(self.nt/2)-1,:].T
        self.y = data[1:int(self.nt/2),:].T
        self.u = data[int(self.nt/2)+1:,:].T  # discard u0 that got x0

        temp = np.hstack((self.x,self.y))
        self.stdData = np.std(temp)
        self.Z = np.random.randn(self.n,self.nbasis)/self.stdData
        self.basis = lambda x:rff(x,self.Z) # where x = (n * time)

        self.psi_x = self.basis(self.x)
        self.psi_y = self.basis(self.y)
        
        Mc = np.matmul(self.psi_x,self.psi_x.T)
        rVec = np.vstack((self.psi_x,self.u))
        lVec = np.vstack((self.psi_y,self.x))
        M1 = np.matmul(rVec,rVec.T)
        M2 = np.matmul(lVec,rVec.T)
        lam = 0.2
        self.operator = np.matmul(M2,LA.inv(M1 + lam*np.eye(np.size(M1,0))))
        self.A = self.operator[:self.nk,:self.nk]
        self.B = self.operator[:self.nk,self.nk:]
        self.C = self.operator[self.nk:,:self.nk]
        self.P = LA.inv(M1)/self.weighting
        self.Pc = LA.inv(Mc)/self.weighting
    
        return self.operator

    # weighted RLS update
    def updateOperator(self,state,action):
        """
        docstring

        """
        self.Aaug = np.hstack((self.A, self.B))
        self.xx = self.y[:,-1].reshape(self.n,1)
        self.uu = action
        self.yy = state.reshape(self.n,1)
        self.psi_xx = self.psi_y[:,-1].reshape(self.nk,1)
        self.psi_yy = self.basis(self.yy)
        self.psi_xaug = np.vstack((self.psi_xx,self.uu))

        # issue: Aaug(AND P) updates w/ [x;u], but C w/ [x]
        P_psi_xaug = self.P@(self.psi_xaug)
        Pc_psi_xx = self.Pc@(self.psi_xx)
        gamma = 1/(1 + self.psi_xaug.T@(P_psi_xaug))
        gammac = 1/(1 + self.psi_xx.T@(Pc_psi_xx))
        self.Aaug += np.outer(gamma*(self.psi_yy - self.Aaug@self.psi_xaug), P_psi_xaug)
        self.C += np.outer(gammac*(self.xx - self.C@self.psi_xx), self.Pc@self.psi_xx)
        self.P = (self.P - gamma*np.outer(P_psi_xaug,P_psi_xaug))/self.weighting
        self.P = (self.P + self.P.T)/2
        self.Pc = (self.Pc - gammac*np.outer(Pc_psi_xx,Pc_psi_xx))/self.weighting
        self.Pc = (self.Pc + self.Pc.T)/2
        self.A = self.Aaug[:,:self.nk]
        self.B = self.Aaug[:,self.nk:]
        CD = np.hstack((self.C,np.zeros((np.size(self.C,0),np.size(self.B,1)))))
        self.operator = np.vstack((self.Aaug,CD))

        return self.operator
    
    def predict(self,x0,u):
        lift = self.A@self.basis(x0) + self.B@u
        return self.C@lift

# RFF basis
def rff(X,Z):
    """
    docstring
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
        Psi = np.vstack((X.reshape(5,1),np.vstack((cos_psi,sin_psi))))  # Nk x m
    else:    
        Psi = np.vstack((X,np.vstack((cos_psi,sin_psi))))  # Nk x m
    return Psi

