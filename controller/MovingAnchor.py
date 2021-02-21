import numpy as np
# change init in test!
class MovingAnchor:
    def __init__(self, Xub, Xlb, Uub, Ulb, num_lift=2, weighting=0.96, Mub=None, Mlb=None):
        """
        potential improvements:
        1. delay
        2. sparse regression to denoise
        """
        self.n = np.size(Xub)
        self.m = np.size(Uub)
        self.weighting = weighting
        self.Xub = Xub.reshape(1,self.n)
        self.Xlb = Xlb.reshape(1,self.n)
        self.Uub = Uub.reshape(1,self.m)
        self.Ulb = Ulb.reshape(1,self.m)

        # set scale center and range
        Xub_scale = self.Xub + 0.3*(self.Xub-self.Xlb)
        Xlb_scale = self.Xlb - 0.3*(self.Xub-self.Xlb)
        Uub_scale = self.Uub + 0.3*(self.Uub-self.Ulb)
        Ulb_scale = self.Ulb - 0.3*(self.Uub-self.Ulb)
        self.state_range = (Xub_scale - Xlb_scale) / 2
        self.state_center = (Xub_scale + Xlb_scale) / 2
        self.action_range = (Uub_scale - Ulb_scale) / 2
        self.action_center = (Uub_scale + Ulb_scale) / 2
        if Mub is None:
            self.metric_range = self.state_range
            self.metric_center = self.state_center
            self.ncost = 0
        else:
            self.metric_range = (Mub - Mlb) / 2
            self.metric_center = (Mub + Mlb) / 2
            self.ncost = np.size(Mub)
        # self.nk = num_lift + self.n + self.m + self.ncost
        self.nk = num_lift + self.n + self.ncost
        print('Set up scale ranges and centers successfully!')
    
    def initialization(self, states, actions, costs=None, Weights = None):
        """
        scale data to [-1,1]
        data = Nt x N
        construct Koopman matrices A, B, C
        initialize recursive update matrices G, beta
        z = A*x + B*u
        x = C*z
        Koopman library functions: RFF
        psi = [x; cost; liftx]
        """
        # set rff:
        rff_sigma_gaussian = np.std(states)
        num_features = int((self.nk - self.n - self.ncost)/2)
        # num_features = int((self.nk - self.n- self.m - self.ncost)/2)
        # np.random.get_state()[1][0]
        np.random.seed(878528420)  # 1536292545 1536292545
        self.rff_z = np.random.randn(self.n, num_features)/rff_sigma_gaussian
        # self.rff_z = np.random.randn(self.n+self.m, num_features)/rff_sigma_gaussian
        print('Set up Random Fourier Features successfully!')
        
        Nt = np.size(states,0) - 1
        Weights = np.sqrt(self.weighting)**range(Nt-1,-1,-1)
        self.regression(states, actions, Weights, costs)

        return self.Theta

    def update(self, states, actions, costs=None, alpha=0.05):
        """
        windowed updates
        """
        window = np.size(actions,0) - 1
        # beta = (1-1.5*alpha)/(window-1)
        Weights = alpha*np.ones([1,window+1])
        Weights[:,0] = self.weighting
        Weights[:,-1] = 1
        if costs is None:
            self.regression(states, actions, Weights)
        else:
            self.regression(states, actions, Weights, costs)
        return self.Theta
        

    def predict(self, states, actions, costs=None):
        """
        scale_down -> predict -> scale_up
        """
        states = states.reshape(1,self.n)
        actions = actions.reshape(1,self.m)
        states_scaled = self.scale(states)
        actions_scaled = self.scale(actions,state_scale=False)
        # zeta_new = np.hstack([states_scaled,actions_scaled])
        if costs is None:
            # delta_new = self.lift(zeta_new).T
            delta_new = np.hstack([self.lift(states_scaled).T,actions_scaled.reshape(1,self.m)])
        else:
            costs_scaled = self.scale_lift(costs)
            delta_new = np.hstack([self.lift(states_scaled,costs_scaled).T,actions_scaled.reshape(1,self.m)])
        y_predicted = np.matmul(delta_new,self.Theta)
        predicted = self.scale(y_predicted,scale_down=False)

        return predicted

    def regression(self, states, actions, Weights, costs=None):
        states_scaled = self.scale(states)
        actions_scaled = self.scale(actions,state_scale=False)
        X_scaled = states_scaled[:-1,:]
        Y_scaled = states_scaled[1:,:]
        U_scaled = actions_scaled
        self.X = Weights*X_scaled.T
        self.Y = Weights*Y_scaled.T
        self.U = Weights*U_scaled.T
        # self.Zeta = np.hstack((self.X.T,self.U.T))
        if costs is None:
            # self.PsiX = self.lift(self.Zeta).T
            self.PsiX = self.lift(self.X.T).T
        else:
            costs_scaled = self.scale_lift(costs)
            CX_scaled = costs_scaled[:-1,:]
            CY_scaled = costs_scaled[1:,:]
            if self.ncost == 1:
                self.CX = np.multiply(Weights.reshape(np.size(Weights),1),CX_scaled)
                self.CY = np.multiply(Weights.reshape(np.size(Weights),1),CY_scaled)
            else:
                self.CX = Weights*CX_scaled.T
                self.CY = Weights*CY_scaled.T
            self.PsiX = self.lift(self.X.T,self.CX).T
        self.Zeta = np.hstack((self.PsiX,self.U.T))
        # self.Zeta = self.PsiX
        self.non_singular = 0.05
        self.Q = np.matmul(self.Zeta.T,self.Y.T)
        self.G = np.linalg.inv(np.matmul(self.Zeta.T,self.Zeta) + self.non_singular*np.eye(np.size(self.Zeta,1)))
        # self.G = np.linalg.inv(np.matmul(self.Zeta,self.Zeta.T))
        self.Theta = np.matmul(self.G,self.Q)
        self.G = self.G /self.weighting
        self.G = (self.G + self.G.T)/2
        # Evaluate regression accuracy:
        error = self.Y.T - np.matmul(self.Zeta,self.Theta)
        NRMSE_Koopman = 100*np.sqrt(sum(np.linalg.norm(error,axis=0)**2)) / np.sqrt(sum(np.linalg.norm(self.Y.T,axis=0)**2))
        print(NRMSE_Koopman,"%")

    def scale_lift(self,data, scale_down=True,metric_scale=True):
        '''
        This scaling would only be used when there have non-convex tracking or constraints,
        where those metrics are appended into the lifted states and will be tackled in MPC. 
        '''
        if metric_scale:
            if scale_down:
                scaled = (data - self.metric_center)/self.metric_range
                # scaled = 2/(1+np.exp(-data)) - 1
            else:
                scaled = data*self.metric_range + self.metric_center
                # scaled = -np.log(2/(data+1)-1)
        else:
            scaled = data
        return scaled

    def scale(self, data, scale_down=True, state_scale=True):
        """
        data is a Nt x N matrix
        scale down to [-1,1], scale up to stored scaling range
        initialize scaling range with first initialization data set
        """ 
        if np.shape(data) == (self.m,1) or np.shape(data) == (self.n,1):
            data = data.T
        if state_scale:
            if scale_down:
                scaled = (data - self.state_center) / self.state_range
                # scaled = 2/(1+np.exp(-data)) - 1
            else:
                scaled = data*self.state_range + self.state_center
                # scaled = -np.log(2/(data+1)-1)
        else:
            if scale_down:
                scaled = (data - self.action_center) / self.action_range
                # scaled = 2/(1+np.exp(-data)) - 1
            else:
                scaled = data*self.action_range + self.action_center
                # scaled = -np.log(2/(data+1)-1)
        
        return scaled

    def lift(self, data, cost=None):
        """
        data is a Nt x N matrix
        cost is a Nt x N matrix
        return lifted states Nk x Nt matrix (s.t A*Psi)
        lift the state space to a Koopman subspace
        lifted = [states; (actions?); lift(states)]
        RFF sampling rff_z ~ N(0, sigma^2*I_n)
        """
        # Quadratic regression (50% NRMSE)
        # Psi = np.ones([np.size(data,0),self.nk])
        # for k in range(self.nk):
        #     for i in range(np.size(data,1)):
        #         for j in range(i,np.size(data,1)):
        #             Psi[:,k] = np.matmul(data[:,i].T,data[:,j])

        if cost is not None:
            if np.size(cost) == self.ncost:
                cost = cost.reshape(1, np.size(cost))
                data = data.reshape(1, np.size(data))
        Q = np.matmul(data,self.rff_z)
        Fcos = np.cos(Q)
        Fsin = np.sin(Q)
        F = np.hstack((Fcos, Fsin))/ np.sqrt(np.size(self.rff_z,1))
        if cost is None:
            Psi = np.hstack((data,F))
        else:
            if np.size(cost) == 1:
                Psi = np.hstack((np.append(data,cost).reshape(1,self.n+self.ncost),F))
            else:
                Psi = np.hstack((data,cost,F))
        Psi = Psi.T

        return Psi

