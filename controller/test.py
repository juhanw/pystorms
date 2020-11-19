# Import python libraries 
import pystorms
# import sys
# sys.path.append("/controller")
from denseMPC import denseMPC
from Koopman import Koopman 
# Python Scientific Computing Stack
import numpy as np
import pandas as pd
# Plotting tools
import seaborn as sns
from matplotlib import pyplot as plt

# KMPC
env_equalfilling = pystorms.scenarios.delta()
done = False

u = []
x0 = []
actions_north3 = []
actions_north2 = []
actions_north1 = []
actions_central = []
actions_south = []
settings = np.ones(5)
Xub = np.array([11.99, 6.59, 5.92, 5.7, 9.5])
Xlb = np.array([5.28, 4.04, 2.11, 2.21, 0])
Uub = np.ones((1,5))
Ulb = np.zeros((1,5))

t = 0
nk = 2 + 5
n0 = 10
KPmodel = Koopman(nk)

while not done:
    if  t <= n0:
        actions =  np.random.uniform(0,1,5)
        u.append(actions) 
        actions_north3.append(actions[0])
        actions_north2.append(actions[1])
        actions_north1.append(actions[2])
        actions_central.append(actions[3])
        actions_south.append(actions[4])

        done = env_equalfilling.step(actions)
        state = env_equalfilling.state()
        x0.append(state[1:])

        if t == n0:
            u = np.asarray(u)
            x0 = np.asarray(x0)
            u.reshape(n0+1,5)
            x0.reshape(n0+1,5)
            
            # initialize Koopman model
            initData = np.vstack((x0,u))
            operator = KPmodel.initOperator(initData)
            A = operator[:nk,:nk]
            B = operator[:nk,nk:]
            C = operator[nk:,:nk]
            
            # initialize controller
            Zub = KPmodel.basis(Xub)
            Zlb = KPmodel.basis(Xlb)
            KMPC = denseMPC(A,B,C,Uub=Uub,Ulb=Ulb,Zub=Zub,Zlb=Zlb)
            
            # initialize containers
            xtrue = state[1:].reshape(5,1)
            xkp = xtrue
            # umpc = np.empty(np.shape(state[1:]),float)
    
            # get control input
            z0 = KPmodel.basis(state[1:]) # x
            actions = KMPC.getMPC(z0,A,B,C)
            actions_north3.append(actions[0])
            actions_north2.append(actions[1])
            actions_north1.append(actions[2])
            actions_central.append(actions[3])
            actions_south.append(actions[4])

            # record
            # umpc = np.vstack((umpc,actions ))
            umpc = actions
            xkp = np.hstack((xkp,KPmodel.predict(xkp[:,-1],actions) ))
            done = env_equalfilling.step(actions)
            state = env_equalfilling.state() # y
            xtrue = np.hstack((xtrue,state[1:].reshape(5,1) ))

    else:
        # update Koopman model
        operator = KPmodel.updateOperator(state[1:],actions)
        A = operator[:nk,:nk]
        B = operator[:nk,nk:]
        C = operator[nk:,:nk]

        # get control input
        z0 = KPmodel.basis(state[1:])
        actions = KMPC.getMPC(z0,None,A,B,C)
        actions_north3.append(actions[0])
        actions_north2.append(actions[1])
        actions_north1.append(actions[2])
        actions_central.append(actions[3])
        actions_south.append(actions[4])

        # record
        umpc = np.hstack((umpc,actions ))
        xkp = np.hstack((xkp,KPmodel.predict(xkp[:,-1],actions) ))
        done = env_equalfilling.step(actions)
        state = env_equalfilling.state() # y
        xtrue = np.hstack((xtrue,state[1:].reshape(5,1) ))

    print(t, "is time")
    t = t + 1
    
equalfilling_perf = sum(env_equalfilling.data_log["performance_measure"])