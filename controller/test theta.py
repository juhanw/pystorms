# Import python libraries 
import pystorms
# import sys
# sys.path.append("/controller")
# from denseMPC import MPC
# from MPC_rewrite import MPC
# from MPC_soft import MPC
# from MPC_smooth import MPC
from MPC_cvx import MPC
# from Koopman import Koopman 
# from Koopman_rewrite import Koopman
# from Koopman_multisteps import Koopman
from Koopman_performance import Koopman
# Python Scientific Computing Stack
import numpy as np
import pandas as pd
# Plotting tools
import seaborn as sns
from matplotlib import pyplot as plt

# KMPC
env_equalfilling = pystorms.scenarios.theta() # states = [BC, BS, N1, N2, N3, N4]
done = False

u = []
x0 = []
metric = []
settings = np.ones(2)
Xub_extreme = np.asarray([0.5, 0.5])
Xlb_extreme = np.asarray([0, 0])
Uub = np.asarray([1,1])
Ulb = np.zeros((1,2))
# Mub = sum(Xub_extreme)
# Mlb = sum(Xlb_extreme)
Mub = 0.5
Mlb = 0

t = 0
tmulti = 10
n_basis = 4
n = Xub_extreme.size
m = Uub.size
ncost = 1
nk = n + n_basis
n0 = 150    #n0 > nk + m
KPmodel = Koopman(Xub_extreme,Xlb_extreme,Uub,Ulb,Mub,Mlb,nk)
Xub_scaled = KPmodel.scale(Xub_extreme)
Xlb_scaled = KPmodel.scale(Xlb_extreme)
Uub_scaled = KPmodel.scale(Uub,state_scale=False)
Ulb_scaled = KPmodel.scale(Ulb,state_scale=False)
Mub_scaled = KPmodel.scale_lift(Mub)
Mlb_scaled = KPmodel.scale_lift(Mlb)
KMPC = MPC(Uub_scaled,Ulb_scaled,Mub=Mub_scaled,Mlb=Mlb_scaled,n=n)
# KMPC = MPC(Uub_scaled,Ulb_scaled, Xub_soft = Xub_scaled, Xlb_soft = Xlb_scaled)

while not done:
    if  t <= n0:
        actions =  np.random.uniform(0,0.5,m)
        u.append(actions)

        done = env_equalfilling.step(np.ones(n))
        state = env_equalfilling.state()
        flow = env_equalfilling.data_log["flow"]["8"]
        x0.append(state)
        metric.append(flow[-1])

        if t == n0:
            u = np.asarray(u)
            x0 = np.asarray(x0)
            u = u.reshape(n0+1,m)
            x0 = x0.reshape(n0+1,n)
            metric = np.asarray(metric).reshape(n0+1,ncost)
            
            # initialize Koopman model:
            A,B,C = KPmodel.initialization(x0,u,metric)
            t1 = t

            # get control input
            z0 = KPmodel.lift(KPmodel.scale(x0[-1,:]),KPmodel.scale_lift(metric[-1,:])) # x
            ulast_scaled = KPmodel.scale(u[-1,:],state_scale=False)
            actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
            if actions_mpc is None:
                actions_mpc = u[-1,:]
            actions = KPmodel.scale(actions_mpc,scale_down=False,state_scale=False)
            actions = actions.T
            
            # initialize containers
            xtrue = x0[-1,:].reshape(1,n)
            xkp = xtrue
            umpc = actions.reshape(1,m)
            xkp = np.vstack((xkp,KPmodel.predict(xtrue,umpc,metric[-1,:]) ))

    else:
        done = env_equalfilling.step(umpc[-1,:])
        state = env_equalfilling.state() # y
        flow = env_equalfilling.data_log["flow"]["8"]
        metric = np.vstack((metric,flow[-1]))
        xtrue = np.vstack((xtrue,state ))
        # xtrue = np.vstack((xtrue,flow[-1] ))

        if t-t1 == tmulti:
            # update Koopman model
            A,B,C = KPmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:],metric[-2,:],metric[-1,:])
            t1 = t
            xkp_new = KPmodel.predict(xtrue[-1,:],umpc[-1,:],metric[-1,:]).reshape(1,np.size(xkp,1))
            xkp[-1,:] = xtrue[-1,:]
            xkp = np.vstack((xkp, xkp_new))
            # get control input
            z0 = KPmodel.lift(KPmodel.scale(xtrue[-1,:]),KPmodel.scale_lift(metric[-1,:]))
            ulast_scaled = KPmodel.scale(umpc[-1,:],state_scale=False)
        else:
            xkp_new = KPmodel.predict(xkp[-1,:],umpc[-1,:], metric[-1,:]).reshape(1,np.size(xkp,1))
            xkp = np.vstack((xkp, xkp_new))
            # get control input
            z0 = KPmodel.lift(KPmodel.scale(xkp[-1,:]),KPmodel.scale_lift(metric[-1,:]))
            ulast_scaled = KPmodel.scale(umpc[-1,:],state_scale=False)
        
        actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
        if actions_mpc is None:
            actions_mpc = umpc[-1,:]
        actions = KPmodel.scale(actions_mpc,scale_down=False,state_scale=False)
        actions = actions.T
        umpc = np.vstack((umpc,actions.reshape(1,np.size(umpc,1)) ))
        xkp_all = xkp[:-1,:]
        qoi = t-50
        error = xtrue - xkp_all
        rmse_square = error**2
        rmse_each = np.sqrt(np.sum(rmse_square,0)/np.size(rmse_square,0))
        rmse_mean = np.mean(xtrue,0)
        nrmse_each = rmse_each/rmse_mean * 100
        # NRMSE_predict = 100*np.sqrt(sum(np.linalg.norm(error[qoi:,:],axis=0)**2)) / np.sqrt(sum(np.linalg.norm(xtrue[qoi:,:],axis=0)**2))
        if nrmse_each.any() < 0.5 and qoi > 0:
            print(nrmse_each)

    print(t, "is time")
    t = t + 1
    if t > 12400:
        # break
        print(t, "is time")
    
equalfilling_perf = sum(env_equalfilling.data_log["performance_measure"])



################################################################################
# Plotting
################################################################################
# set seaborn figure preferences and colors
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style("darkgrid")
colorpalette = sns.color_palette("colorblind")
colors_hex = colorpalette.as_hex()
plt.rcParams['figure.figsize'] = [20, 15]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
xkp_all = xkp[:-1,:]
qoi = 0
# error = xtrue - xkp_all
# rmse_mean = np.mean(xtrue,0)
error = env_equalfilling.data_log["flow"]["8"][n0:] - metric[n0:]
rmse_mean = np.mean(env_equalfilling.data_log["flow"]["8"][n0:])
rmse_square = error**2
rmse_each = np.sqrt(np.sum(rmse_square,0)/np.size(rmse_square,0))
nrmse_each = rmse_each/rmse_mean * 100
# NRMSE_predict = 100*np.sqrt(sum(np.linalg.norm(error[qoi:,:],axis=0)**2)) / np.sqrt(sum(np.linalg.norm(xtrue[qoi:,:],axis=0)**2))
print(nrmse_each,"%")

plotenvironment = env_equalfilling

fig = plt.figure(figsize=(8,8))
fig.add_subplot(2,1, 1)
# plt.ylabel("Depth")
# for i in range(n):
#     label1 = "P"+str(i+1)+" Koopman NRMSE = "+str(round(nrmse_each[0],3))+"%"
#     label2 = "P"+str(i+1)+" Ground True, Sampling T = "+ str(tmulti)+" steps"
#     plt.plot(xtrue[:,i],label=label2)
#     plt.plot(xkp_all[:,i],label=label1)
label3 = "Koopman NRMSE = "+str(round(nrmse_each[0],3))+"%"
plt.plot(env_equalfilling.data_log["flow"]["8"][n0:], label="Ground True")
plt.plot(metric[n0:],label=label3)
plt.axhline(0.5, color="r",label="Limit = 0.5")
plt.ylabel('Outflows')
plt.legend(loc='upper right',prop={'size': 12})
# new plot on control =============================================================
# plt.subplot(2, 2, 3)
fig.add_subplot(2,2,3)
# plt.rcParams['figure.figsize'] = [6, 6]
for i in range(m):
    labelu = "MPC - P"+str(i+1)
    plt.plot(umpc[:,i], label=labelu, linestyle='--', linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.legend(loc='upper right',prop={'size': 12})
plt.ylabel('Control asset setting')
plt.xlabel('Simulation step')
# plt.show()
# '''
fig.add_subplot(2,2,4)
# plt.subplot(2, 2, 4)
dth = np.linspace(0,2*np.pi,100)
evalA = np.linalg.eigvals(A)
plt.plot(evalA.real,evalA.imag,'o',label = 'Koopman Eigenvalues')
plt.plot(np.cos(dth),np.sin(dth),'-',color='r',label='Unit Circle')
plt.axis('equal')
plt.legend(loc='upper right',prop={'size': 12})
plt.ylabel('Imaginary Axis')
plt.xlabel('Real Axis')
plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.show()