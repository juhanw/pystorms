import numpy as np
# change init in test!
class Koopman2:
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
        self.nk = num_lift + self.n + self.m + self.ncost
        print('Set up scale ranges and centers successfully!')

    def initialization(self, states, actions, costs=None):
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
        num_features = int((self.nk - self.n- self.m- self.ncost)/2)
        # np.random.get_state()[1][0]
        np.random.seed(878528420)  # 1536292545 1536292545
        self.rff_z = np.random.randn(self.n + self.m, num_features)/rff_sigma_gaussian
        print('Set up Random Fourier Features successfully!')

        states_scaled = self.scale(states)
        actions_scaled = self.scale(actions,state_scale=False)
        X_scaled = states_scaled[:-1,:]
        U_scaled = actions_scaled
        Nt = np.size(X_scaled,0)
        Weights = np.sqrt(self.weighting)**range(Nt-1,-1,-1)
        # Weights = Weights.reshape(Nt,1)
        X = Weights*X_scaled.T
        self.X = X[:,:-1]
        self.Y = X[:,1:]
        self.U = Weights*U_scaled.T
        self.XU = np.vstack([self.X,self.U[:,:-1]])
        self.YU = np.vstack([self.Y,self.U[:,1:]])
        
        if costs is None:
            self.PsiXU = self.lift(self.XU.T)
            self.PsiYU = self.lift(self.YU.T)
        else:
            costs_scaled = self.scale_lift(costs)
            CX_scaled = costs_scaled[:-1,:]
            if self.ncost == 1:
                CX = np.multiply(Weights.reshape(len(Weights),1),CX_scaled)
                self.CX = CX[:-1].T
                self.CY = CX[1:].T
            else:
                self.CX = Weights*CX_scaled[:,:-1].T
                self.CY = Weights*CX_scaled[:,1:].T
            self.PsiXU = self.lift(self.XU.T,self.CX.T)
            self.PsiYU = self.lift(self.YU.T,self.CY.T)
        self.non_singular = 0.05
        self.Q = np.matmul(self.PsiYU,self.PsiXU.T)
        self.G = np.linalg.inv(np.matmul(self.PsiXU,self.PsiXU.T) + self.non_singular*np.eye(len(self.PsiXU)))
        # self.G = np.linalg.inv(np.matmul(self.Zeta,self.Zeta.T))
        self.AB = np.matmul(self.Q,self.G)
        self.C = np.hstack((np.eye(self.n),np.zeros((self.n,self.nk-self.n))))
        self.G = self.G /self.weighting
        self.G = (self.G + self.G.T)/2
        # Evaluate regression accuracy:
        error = self.Y - np.matmul(self.C,np.matmul(self.AB,self.PsiXU))
        NRMSE_Koopman = 100*np.sqrt(sum(np.linalg.norm(error,axis=0)**2)) / np.sqrt(sum(np.linalg.norm(self.Y,axis=0)**2))
        print(NRMSE_Koopman,"%")

        return self.AB, self.C

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
                Psi = np.hstack((np.append(data,cost).reshape(1,self.n+self.m+self.ncost),F))
            else:
                Psi = np.hstack((data,cost,F))
        Psi = Psi.T

        return Psi

    def update(self, states_x, states_y, action_x, action_y, costs_x=None, costs_y= None):
        """
        recursive update AB, G 
        states is 1 x n
        actions is 1 x m
        scale_down -> update
        """
        statesx_scaled = self.scale(states_x.reshape(1,self.n))
        statesy_scaled = self.scale(states_y.reshape(1,self.n))
        actionx_scaled = self.scale(action_x.reshape(1,self.m),state_scale=False)
        actiony_scaled = self.scale(action_y.reshape(1,self.m),state_scale=False)
        x_new = statesx_scaled.reshape(self.n,1)
        y_new = statesy_scaled.reshape(self.n,1)
        ux_new = actionx_scaled.reshape(self.m,1)
        uy_new = actiony_scaled.reshape(self.m,1)
        xu_new = np.vstack([x_new,ux_new])
        yu_new = np.vstack([y_new,uy_new])
        
        if costs_x is None:
            delta = self.lift(xu_new.reshape(1,self.n+self.m))
        else:
            costsx_scaled = self.scale_lift(costs_x)
            costsy_scaled = self.scale_lift(costs_y)
            delta = self.lift(xu_new.reshape(1,self.n+self.m),costsx_scaled)
        calc_easy = np.matmul(self.G,delta)
        beta = 1/(1 + np.matmul(delta.T,calc_easy))
        if costs_x is None:
            innovation = self.lift(yu_new.reshape(1,self.n+self.m)) - np.matmul(self.AB,delta)
        else:
            innovation = self.lift(yu_new.reshape(1,self.n+self.m),costsy_scaled) - np.matmul(self.AB,delta)
        self.AB += beta*np.matmul(innovation,calc_easy.T)
        if np.max(self.AB) > 1000:
            print("SOmething wrong") # G has become negative definite det = -5e136 <-- calc_easy exp grow
        self.G = (self.G - beta*np.matmul(calc_easy,calc_easy.T))/self.weighting
        self.G = (self.G + self.G.T)/2
        # print(np.linalg.det(self.G))
        if np.linalg.det(self.G) < 0:
            temp = beta*np.matmul(calc_easy,calc_easy.T)
            rank = np.linalg.eigvals(temp)
            print(min(rank))
            GG = np.linalg.inv(np.matmul(self.PsiXU,self.PsiXU.T) + self.non_singular*np.eye(len(self.PsiXU)))
            GG = GG /self.weighting
            GG = (GG + GG.T)/2
            self.G = GG
        
        self.XU = np.hstack((self.XU,xu_new))
        self.YU = np.hstack((self.YU,yu_new))
        return self.AB, self.C

    def predict(self, states, actions, costs=None):
        """
        scale_down -> Koopman_predict -> scale_up
        """
        states = states.reshape(1,self.n)
        actions = actions.reshape(1,self.m)
        states_scaled = self.scale(states)
        actions_scaled = self.scale(actions,state_scale=False)
        xu_scaled = np.hstack([states_scaled,actions_scaled])
        if costs is None:
            delta_new = self.lift(xu_scaled)
        else:
            costs_scaled = self.scale_lift(costs)
            delta_new = self.lift(xu_scaled,costs_scaled)
        lift_predicted = np.matmul(self.AB,delta_new)
        predicted = self.scale(lift_predicted[:self.n].reshape(1,self.n),scale_down=False)
        # u_pred = self.scale(lift_predicted[self.n:self.n+self.m].reshape(1,self.n),scale_down=False,state_scale=False)
        u_pred = lift_predicted[self.n:self.n+self.m]

        return predicted,u_pred