import numpy as np
import scipy.linalg as LA
from sklearn.metrics import mean_squared_error

class Koopman:
    """
    docstring
    """
    def __init__(self,N_basis = 2,forgetCoeff = 0.9):
        """
        initialization
        """
        self.nk = N_basis
        self.weighting = forgetCoeff

    # initialization (need 2 timesnap data)
    def initOperator(self,xy,u,Xub,Xlb,Uub,Ulb):
        """
        change: nk = nk - N
        data = [x0;u] half-half
        Koopman model:
        [psi_x1]  = [A, B] * [psi_x0]    
        [x0    ]    [C, 0]   [u0    ] 
        
        as starting from 1 data pair, thus there is no need to weight date
        the weighting process is contained in the recursive updates  
        """
        self.Xub = Xub
        self.Xlb = Xlb
        self.Uub = Uub
        self.Ulb =Ulb
        self.n = np.size(xy,1)
        self.m = np.size(u,1)
        self.nbasis = self.nk - self.n
        
        x_init = xy[:-1,:].T
        y_init = xy[1:,:].T
        u_init = u[1:,:].T

        Nt = len(x_init[0,:])
        weight = np.sqrt(self.weighting)**range(Nt-1,-1,-1)
        self.x, self.y, self.u = weight*x_init, weight*y_init, weight*u_init
        self.x = self.scale(self.x,state_scale=True)
        self.y = self.scale(self.y,state_scale=True)
        self.u = self.scale(self.u,action_scale=True)

        mu = np.mean(xy,0)
        sum_norm2 = 0
        N = np.size(xy,0)
        for i in range(N):
            sum_norm2 = sum_norm2 + np.linalg.norm(xy[i,:]-mu)**2
        std_k = np.sqrt(sum_norm2 / N)
        self.stdData = std_k
        # self.stdData = np.std(xy)
        
        self.Z = np.random.randn(self.n,self.nbasis)/self.stdData
        self.basis = lambda x:rff(x,self.Z) # where x = (n x time)
        self.psi_x = self.basis(self.x)
        self.psi_y = self.basis(self.y)
        
        Zeta = np.vstack((self.psi_x,self.u))
        self.Q = np.matmul(self.psi_y,Zeta.T)
        self.G = LA.inv(np.matmul(Zeta,Zeta.T))/self.weighting
        self.M = np.matmul(self.x,self.psi_x.T)
        self.P = LA.inv(np.matmul(self.psi_x,self.psi_x.T))/self.weighting
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
        yy = self.scale(state,state_scale=True).reshape(self.n,1)
        uu = self.scale(action, action_scale=True).reshape(self.m,1)
        xx = self.y[:,-1].reshape(self.n,1)
        # uu = action.reshape(self.m,1)
        # yy = state.reshape(self.n,1)
        psi_xx = self.psi_y[:,-1].reshape(self.nk,1)
        psi_yy = self.basis(yy)
        delta = np.vstack((psi_xx,uu))
        self.u = np.hstack((self.u, uu))
        self.x = np.hstack((self.x, xx))
        self.y = np.hstack((self.y, yy))
        self.psi_x = np.hstack((self.psi_x, psi_xx))
        self.psi_y = np.hstack((self.psi_y, psi_yy))
        
        self.G = (self.G + self.G.T)/2
        self.P = (self.P + self.P.T)/2
        beta = 1/(1 + np.matmul(np.matmul(delta.T,self.G), delta))
        gamma = 1/(1 + np.matmul(np.matmul(psi_xx.T,self.P), psi_xx))
        inside1 = np.matmul(self.G,delta)
        inside2 = np.matmul(self.P,psi_xx)
        innovation1 = psi_yy - np.matmul(self.AB,delta)
        innovation2 = xx - np.matmul(self.C,psi_xx)
        self.AB = self.AB + beta*np.matmul(innovation1,inside1.T)
        self.A = self.AB[:,:self.nk]
        self.B = self.AB[:,self.nk:]
        self.C = self.C + gamma*np.matmul(innovation2,inside2.T)
        self.G = (self.G - beta*np.matmul(inside1,inside1.T))/self.weighting
        self.P = (self.P - gamma*np.matmul(inside2,inside2.T))/self.weighting

        operator = np.vstack((self.AB,np.hstack((self.C,np.zeros((self.n,self.m))))))
        
        return operator
    
    def predict(self,x0,u):
        lift = np.matmul(self.A,self.basis(x0.reshape(self.n,1))) + np.matmul(self.B,u.reshape(self.m,1))
        x_kp = np.matmul(self.C,lift)
        # return x_kp.reshape(1,self.n)
        x_kp_record = self.scale(x_kp,down=False,state_scale=True)
        return x_kp_record.reshape(1,self.n)

    def scale(self, data, down = True, state_scale = False, action_scale = False):
        if np.size(data) == self.n or np.size(data) == self.m:
            data = data.reshape(np.size(data),1)
        
        if np.min(np.shape(data)) > 1:
            if state_scale:
                self.state_center = (self.Xub + self.Xlb)/2     #Xub = 1 x n
                self.state_range = (self.Xub - self.Xlb)/2
                self.state_center = self.state_center.reshape(self.n,1)
                self.state_range = self.state_range.reshape(self.n,1)
            elif action_scale:
                self.action_center = (self.Uub + self.Ulb)/2
                self.action_range = (self.Uub - self.Ulb)/2
                self.action_center = self.action_center.reshape(self.m,1)
                self.action_range = self.action_range.reshape(self.m,1)

        if np.shape(data)[0] < np.shape(data)[1]:
            data = data.T
        if down:
            if state_scale:
                scaled = (data - self.state_center)/self.state_range
            elif action_scale:
                scaled = (data - self.action_center)/self.action_range
            else:
                print("ERROR scaling parameter!")
                scaled = data
        else:
            if state_scale:
                scaled = data*self.state_range + self.state_center
            elif action_scale:
                scaled = data*self.action_range + self.action_center
            else:
                print("ERROR scaling parameter!")
                scaled = data

        return scaled

# RFF basis
def rff(X,Z):
    """
    X is a n x 1 vector
    return Psi as a Nk x t matrix
    """
    if np.size(X,0) != 5:
        X = X.reshape(5,int(np.size(X)/5))
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