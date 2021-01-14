import numpy as np

class Koopman:
    def __init__(self,num_basis, weighting=1):
        """
        potential improvements:
        1. delay
        2. sparse regression to denoise
        """
        self.nk = num_basis
        self.weighting = weighting


    def initialization(self, states, actions, Xub, Xlb, Uub, Ulb):
        """
        scale data to [-1,1]
        data = Nt x N
        construct Koopman matrices A, B, C
        initialize recursive update matrices G, beta
        z = A*x + B*u
        x = C*z
        Koopman library functions: RFF
        """
        self.n = np.size(states,axis=1)
        self.m = np.size(actions,axis=1)
        self.Xub = Xub.reshape(1,self.n)
        self.Xlb = Xlb.reshape(1,self.n)
        self.Uub = Uub.reshape(1,self.m)
        self.Ulb = Ulb.reshape(1,self.m)
        states_scaled = self.scale(states,state_scale=True)
        actions_scaled = self.scale(actions,action_scale=True)
        Nt = np.size(actions,0)
        Weights = np.sqrt(self.weighting)**range(Nt-1,-1,-1)
        Weights = Weights.reshape(Nt,1)
        states_weighted = Weights*states_scaled
        actions_weighted = Weights*actions_scaled
        self.X = states_weighted[:-1,:]
        self.Y = states_weighted[1:,:]
        self.U = actions_weighted[1:,:]
        self.PsiX = self.lift(self.X)
        self.PsiY = self.lift(self.Y)
        self.X = self.X.reshape(self.n,Nt-1)
        self.Y = self.Y.reshape(self.n,Nt-1)
        self.U = self.U.reshape(self.m,Nt-1)
        self.Zeta = np.vstack((self.PsiX,self.U))
        non_singular = 0.1
        self.Q = np.matmul(self.PsiY,self.Zeta.T)
        self.G = np.linalg.inv(np.matmul(self.Zeta,self.Zeta.T) + non_singular*np.eye(len(self.Zeta)))
        # self.G = np.linalg.inv(np.matmul(self.Zeta,self.Zeta.T))
        self.AB = np.matmul(self.Q,self.G)
        self.A = self.AB[:,:self.nk]
        self.B = self.AB[:,self.nk:]
        self.C = np.hstack((np.eye(self.n),np.zeros((self.n,self.nk-self.n))))
        self.G = self.G /self.weighting
        # Evaluate regression accuracy:
        error = self.Y - np.matmul(self.C,np.matmul(self.AB,self.Zeta))
        NRMSE_Koopman = 100*np.sqrt(sum(np.linalg.norm(error,axis=0)**2)) / np.sqrt(sum(np.linalg.norm(self.Y,axis=0)**2))
        print(NRMSE_Koopman,"%")

        return self.A, self.B, self.C

    def scale(self, data, scale_down=True, state_scale=False, action_scale=False):
        """
        data is a Nt x N matrix
        scale down to [-1,1], scale up to stored scaling range
        initialize scaling range with first initialization data set
        """
        if data.size != self.n and  data.size != self.m: 
            if state_scale:
                max_data = self.Xub
                min_data = self.Xlb
                self.state_range = (max_data + min_data) / 2
                self.state_center = (max_data - min_data) / 2
            elif action_scale:
                max_data = self.Uub
                min_data = self.Ulb
                self.action_range = (max_data + min_data) / 2
                self.action_center = (max_data - min_data) / 2
            else:
                print("Error scaling data!")
            
        if state_scale:
            if scale_down:
                scaled = (data - self.state_center) / self.state_range
            else:
                scaled = data*self.state_range + self.state_center
        elif action_scale:
            if scale_down:
                scaled = (data - self.action_center) / self.action_range
            else:
                scaled = data*self.action_range + self.action_center
        else:
            print("Error scaling direction!")
            scaled = data

        return scaled

    def lift(self, data):
        """
        data is a Nt x N matrix
        return lifted states Nk x Nt matrix (s.t A*Psi)
        lift the state space to a Koopman subspace
        lifted = [states; (actions?); lift(states)]
        RFF sampling rff_z ~ N(0, sigma^2*I_n)
        # RBF
        """
        if data.shape[1] > 1: 
            # set rff:
            rff_sigma_gaussian = np.std(data)
            num_features = int((self.nk - self.n)/2)
            rff_z = np.random.normal(0, 1.0/rff_sigma_gaussian, (data.shape[1], num_features))
            # print('Set up Random Fourier Features successfully!')
        Q = np.matmul(data,rff_z)
        Fcos = np.cos(Q)
        Fsin = np.sin(Q)
        F = np.hstack((Fcos, Fsin))
        Psi = np.hstack((data,F))
        Psi = Psi.T

        return Psi

    def update(self, states, actions):
        """
        recursive update AB, G 
        states is 1 x n
        actions is 1 x m
        scale_down -> update
        """
        states = states.reshape(1,self.n)
        actions = actions.reshape(1,self.m)
        states_scaled = self.scale(states,state_scale=True)
        actions_scaled = self.scale(actions,action_scale=True)
        y_new = states_scaled.reshape(self.n,1)
        u_new = actions_scaled.reshape(self.m,1)
        x_new = self.Y[:,-1].reshape(self.n,1)
        delta = np.vstack((self.lift(x_new.reshape(1,self.n)),u_new))
        calc_easy = np.matmul(self.G,delta)
        beta = 1/(1 + np.matmul(delta.T,calc_easy))
        innovation = self.lift(y_new.reshape(1,self.n)) - np.matmul(self.AB,delta)
        self.AB += beta*np.matmul(innovation,calc_easy.T)
        self.A = self.AB[:,:self.nk]
        self.B = self.AB[:,self.nk:]
        self.G = (self.G - beta*np.matmul(calc_easy,calc_easy.T))/self.weighting
        self.X = np.hstack((self.X,x_new))
        self.Y = np.hstack((self.Y,y_new))
        self.U= np.hstack((self.U,u_new))
        return self.A, self.B, self.C

    def predict(self, states, actions):
        """
        scale_down -> Koopman_predict -> scale_up
        """
        states = states.reshape(1,self.n)
        actions = actions.reshape(1,self.m)
        states_scaled = self.scale(states,state_scale=True)
        actions_scaled = self.scale(actions,action_scale=True)
        delta_new = np.vstack([self.lift(states_scaled),actions_scaled.reshape(self.m,1)])
        lift_predicted = np.matmul(self.AB,delta_new)
        predicts_scaled = np.matmul(self.C, lift_predicted)
        predicted = self.scale(predicts_scaled.reshape(1,self.n),scale_down=False,state_scale=True)

        return predicted