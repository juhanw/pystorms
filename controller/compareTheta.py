# Import python libraries 
import pystorms
from MPC_cvx import MPC
# from MPC_smooth import MPC
from Koopman_performance import Koopman
from Koopman_liftCtrl import Koopman2
from DMD_MIL import DMD
from NARX import NARX
from MovingAnchor import MovingAnchor
# Python Scientific Computing Stack
import numpy as np
import pandas as pd
# Plotting tools
import seaborn as sns
from matplotlib import pyplot as plt

# KMPC
env_equalfilling = pystorms.scenarios.theta() # states = [BC, BS, N1, N2, N3, N4]
done = False

settings = np.ones(2)
Xub_extreme = np.asarray([0.5, 0.5])
Xlb_extreme = np.asarray([-0.1, -0.1])
Uub = np.asarray([1,1])
Ulb = np.asarray([0,0])
Mub = 0.5
Mlb = 0
# np.random.get_state()[1][0]
np.random.seed(878528420)  # 1536292545 1536292545

t = 0
tmulti = 50
n_basis = 6
n = Xub_extreme.size
m = Uub.size
ncost = 1
n0 = 150    #n0 > nk + m 150?
KPmodel = Koopman(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis,Mub=Mub,Mlb=Mlb)
KCmodel = Koopman2(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis,Mub=Mub,Mlb=Mlb)
DMDmodel = DMD(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis,Mub=Mub,Mlb=Mlb)
NARXmodel = NARX(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis,Mub=Mub,Mlb=Mlb)
MAmodel = MovingAnchor(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis,Mub=Mub,Mlb=Mlb)
# KPmodel = Koopman(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
# KCmodel = Koopman2(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
# DMDmodel = DMD(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
# NARXmodel = NARX(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
# MAmodel = MovingAnchor(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
Xub_scaled = KPmodel.scale(Xub_extreme)
Xlb_scaled = KPmodel.scale(Xlb_extreme)
Uub_scaled = KPmodel.scale(Uub,state_scale=False)
Ulb_scaled = KPmodel.scale(Ulb,state_scale=False)
Mub_scaled = KPmodel.scale_lift(Mub)
Mlb_scaled = KPmodel.scale_lift(Mlb)
KMPC = MPC(Uub_scaled,Ulb_scaled,Mub=Mub_scaled,Mlb=Mlb_scaled,n=n)
# KMPC = MPC(Uub_scaled,Ulb_scaled,N=n)

u = []
x0 = 0*np.ones(n)
metric = 0*np.ones(ncost)
while not done:
    if  t <= n0:
        actions = np.ones(m)
        u.append(actions)

        done = env_equalfilling.step(actions)
        state = env_equalfilling.state()
        flow = env_equalfilling.data_log["flow"]["8"]
        x0 = np.vstack([x0,state.reshape(1,n)])
        metric = np.vstack([metric,flow[-1]])

        if t == n0:
            u = np.asarray(u)
            x0 = np.asarray(x0)
            u = u.reshape(n0+1,m)
            
            # initialize Koopman model:
            t1 = t
            A,B,C = KPmodel.initialization(x0,u,metric)
            Admd, Bdmd, Cdmd = DMDmodel.initialization(x0,u,metric)
            ABc,Cc = KCmodel.initialization(x0,u,metric)
            ThetaNARX = NARXmodel.initialization(x0,u,metric)
            ThetaMA = MAmodel.initialization(x0,u,metric)
            # A,B,C = KPmodel.initialization(x0,u)
            # Admd, Bdmd, Cdmd = DMDmodel.initialization(x0,u)
            # ABc,Cc = KCmodel.initialization(x0,u)
            # ThetaNARX = NARXmodel.initialization(x0,u)
            # ThetaMA = MAmodel.initialization(x0,u)

            # get control input
            z0 = KPmodel.lift(KPmodel.scale(x0[-1,:]),KPmodel.scale_lift(metric[-1,:])) # x
            # z0 = KPmodel.lift(KPmodel.scale(x0[-1,:])) # x
            ulast_scaled = KPmodel.scale(u[-1,:],state_scale=False)
            actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
            if actions_mpc is None:
                actions_mpc = u[-1,:]
            actions = KPmodel.scale(actions_mpc,scale_down=False,state_scale=False)
            actions = actions.T
            xc_new,uc_new = KCmodel.predict(x0[-1,:].reshape(1,n),u[-1,:].reshape(1,m),metric[-1,:])
            # xc_new,uc_new = KCmodel.predict(x0[-1,:].reshape(1,n),u[-1,:].reshape(1,m))
            # actions = uc_new
            
            # initialize containers
            xtrue = x0[-1,:].reshape(1,n)
            metric = metric[-1,:].reshape(1,ncost)
            umpc = actions.reshape(1,m)
            xkp = xtrue
            xkp = np.vstack((xkp,KPmodel.predict(xtrue,umpc,metric[-1,:]) ))
            xdmd = xtrue
            xdmd = np.vstack((xdmd,DMDmodel.predict(xtrue,umpc,metric[-1,:]) ))
            xc = xtrue
            xc = np.vstack((xc, xc_new))
            xnarx = xtrue
            xnarx = np.vstack((xnarx,NARXmodel.predict(xtrue,umpc,metric[-1,:]) ))
            xma = xtrue
            xma = np.vstack((xma,MAmodel.predict(xtrue,umpc,metric[-1,:]) ))
            # xkp = xtrue
            # xkp = np.vstack((xkp,KPmodel.predict(xtrue,umpc) ))
            # xdmd = xtrue
            # xdmd = np.vstack((xdmd,DMDmodel.predict(xtrue,umpc) ))
            # xc = xtrue
            # xc = np.vstack((xc, xc_new))
            # xnarx = xtrue
            # xnarx = np.vstack((xnarx,NARXmodel.predict(xtrue,umpc) ))
            # xma = xtrue
            # xma = np.vstack((xma,MAmodel.predict(xtrue,umpc) ))

    else:
        done = env_equalfilling.step(umpc[-1,:])
        state = env_equalfilling.state() # y
        flow = env_equalfilling.data_log["flow"]["8"]
        metric = np.vstack((metric,flow[-1]))
        xtrue = np.vstack((xtrue,state ))

        if t-t1 == tmulti:
            # update Koopman model
            A,B,C = KPmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:],metric[-2,:],metric[-1,:])
            xkp_new = KPmodel.predict(xtrue[-1,:],umpc[-1,:],metric[-1,:]).reshape(1,np.size(xkp,1))
            xkp[-1,:] = xtrue[-1,:]
            xkp = np.vstack((xkp, xkp_new))
            Admd,Bdmd,Cdmd = DMDmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:],metric[-2,:],metric[-1,:])
            xdmd_new = DMDmodel.predict(xtrue[-1,:],umpc[-1,:],metric[-1,:]).reshape(1,np.size(xdmd,1))
            xdmd[-1,:] = xtrue[-1,:]
            xdmd = np.vstack([xdmd, xdmd_new])
            ABc,Cc = KCmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-2,:],umpc[-1,:],metric[-2,:],metric[-1,:])
            xc_new,uc_new = KCmodel.predict(xtrue[-1,:],umpc[-1,:],metric[-1,:])
            xc[-1,:] = xtrue[-1,:]
            xc = np.vstack((xc, xc_new))
            ThetaNARX = NARXmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:],metric[-2,:],metric[-1,:])
            xnarx_new = NARXmodel.predict(xtrue[-1,:],umpc[-1,:],metric[-1,:]).reshape(1,np.size(xnarx,1))
            xnarx[-1,:] = xtrue[-1,:]
            xnarx = np.vstack((xnarx, xnarx_new))
            ThetaMA = MAmodel.update(xtrue[-(tmulti+2):,:],umpc[-(tmulti+1):,:],metric[-(tmulti+2):,:])
            xma_new = MAmodel.predict(xtrue[-1,:],umpc[-1,:],metric[-1,:]).reshape(1,np.size(xma,1))
            xma[-1,:] = xtrue[-1,:]
            xma = np.vstack((xma, xma_new))
            # A,B,C = KPmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:])
            # xkp_new = KPmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xkp,1))
            # xkp[-1,:] = xtrue[-1,:]
            # xkp = np.vstack((xkp, xkp_new))
            # Admd,Bdmd,Cdmd = DMDmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:])
            # xdmd_new = DMDmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xdmd,1))
            # xdmd[-1,:] = xtrue[-1,:]
            # xdmd = np.vstack([xdmd, xdmd_new])
            # ABc,Cc = KCmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-2,:],umpc[-1,:])
            # xc_new,uc_new = KCmodel.predict(xtrue[-1,:],umpc[-1,:])
            # xc[-1,:] = xtrue[-1,:]
            # xc = np.vstack((xc, xc_new))
            # ThetaNARX = NARXmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:])
            # xnarx_new = NARXmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xnarx,1))
            # xnarx[-1,:] = xtrue[-1,:]
            # xnarx = np.vstack((xnarx, xnarx_new))
            # ThetaMA = MAmodel.update(xtrue[-(tmulti+2):,:],umpc[-(tmulti+1):,:])
            # xma_new = MAmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xma,1))
            # xma[-1,:] = xtrue[-1,:]
            # xma = np.vstack((xma, xma_new))

            # get control input
            z0 = KPmodel.lift(KPmodel.scale(xtrue[-1,:]),KPmodel.scale_lift(metric[-1,:]))
            # z0 = KPmodel.lift(KPmodel.scale(xtrue[-1,:]))
            ulast_scaled = KPmodel.scale(umpc[-1,:],state_scale=False)
            t1 = t
        else:
            xkp_new = KPmodel.predict(xkp[-1,:],umpc[-1,:], metric[-1,:]).reshape(1,np.size(xkp,1))
            xkp = np.vstack((xkp, xkp_new))
            xdmd_new = DMDmodel.predict(xdmd[-1,:],umpc[-1,:], metric[-1,:]).reshape(1,np.size(xdmd,1))
            xdmd = np.vstack((xdmd, xdmd_new))
            xc_new,uc_new = KCmodel.predict(xc[-1,:],umpc[-1,:], metric[-1,:])
            xc = np.vstack((xc, xc_new))
            xnarx_new = NARXmodel.predict(xnarx[-1,:],umpc[-1,:], metric[-1,:]).reshape(1,np.size(xnarx,1))
            xnarx = np.vstack((xnarx, xnarx_new))
            xma_new = MAmodel.predict(xma[-1,:],umpc[-1,:], metric[-1,:]).reshape(1,np.size(xma,1))
            xma = np.vstack((xma, xma_new))
            # xkp_new = KPmodel.predict(xkp[-1,:],umpc[-1,:]).reshape(1,np.size(xkp,1))
            # xkp = np.vstack((xkp, xkp_new))
            # xdmd_new = DMDmodel.predict(xdmd[-1,:],umpc[-1,:]).reshape(1,np.size(xdmd,1))
            # xdmd = np.vstack((xdmd, xdmd_new))
            # xc_new,uc_new = KCmodel.predict(xc[-1,:],umpc[-1,:])
            # xc = np.vstack((xc, xc_new))
            # xnarx_new = NARXmodel.predict(xnarx[-1,:],umpc[-1,:]).reshape(1,np.size(xnarx,1))
            # xnarx = np.vstack((xnarx, xnarx_new))
            # xma_new = MAmodel.predict(xma[-1,:],umpc[-1,:]).reshape(1,np.size(xma,1))
            # xma = np.vstack((xma, xma_new))

            # get control input
            z0 = KPmodel.lift(KPmodel.scale(xkp[-1,:]),KPmodel.scale_lift(metric[-1,:]))
            # z0 = KPmodel.lift(KPmodel.scale(xkp[-1,:]))
            ulast_scaled = KPmodel.scale(umpc[-1,:],state_scale=False)
        
        actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
        # actions_mpc = uc_new
        if actions_mpc is None:
            actions_mpc = umpc[-1,:]
        actions = KPmodel.scale(actions_mpc,scale_down=False,state_scale=False)
        actions = actions.T
        umpc = np.vstack((umpc,actions.reshape(1,np.size(umpc,1)) ))

    print(t, "is time")
    t = t + 1
    if t > 6500:
        # break
        print(t, "is time")
    
equalfilling_perf = sum(env_equalfilling.data_log["performance_measure"])



################################################################################
# Plotting
################################################################################
# set seaborn figure preferences and colors
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1.5})
sns.set_style("darkgrid")
colorpalette = sns.color_palette("colorblind") # Paired
colors_hex = colorpalette.as_hex()
# colors_hex = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'w']
plt.rcParams['figure.figsize'] = [20, 15]
plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower
# Koopman MIL NRMSE
xkp_all = xkp[:-1,:]
qoi = 0
ind = xtrue > 0.001
error = xtrue[ind] - xkp_all[ind]
rmse_mean = np.mean(xtrue[ind],0)
rmse_square = error**2
rmse_each = np.sqrt(np.sum(rmse_square)/np.size(rmse_square))
rmse_mean = np.sqrt(np.sum(xtrue[ind]**2)/np.size(xtrue[ind]))
nrmse_each1 = rmse_each/rmse_mean * 100
print("Koopman NRMSE = ",nrmse_each1,"%")
# DMD MIL NRMSE
xdmd_all = xdmd[:-1,:]
qoi = 0
ind = xtrue > 0.001
error = xtrue[ind] - xdmd_all[ind]
rmse_mean = np.mean(xtrue[ind],0)
rmse_square = error**2
rmse_each = np.sqrt(np.sum(rmse_square)/np.size(rmse_square))
rmse_mean = np.sqrt(np.sum(xtrue[ind]**2)/np.size(xtrue[ind]))
nrmse_each2 = rmse_each/rmse_mean * 100
print("DMD NRMSE = ",nrmse_each2,"%")
# Koopman2 MIL NRMSE
xc_all = xc[:-1,:]
qoi = 0
ind = xtrue > 0.001
error = xtrue[ind] - xc_all[ind]
rmse_mean = np.mean(xtrue[ind],0)
rmse_square = error**2
rmse_each = np.sqrt(np.sum(rmse_square)/np.size(rmse_square))
rmse_mean = np.sqrt(np.sum(xtrue[ind]**2)/np.size(xtrue[ind]))
nrmse_each3 = rmse_each/rmse_mean * 100
print("Koopman2 NRMSE = ",nrmse_each3,"%")
timeFromN0 = np.linspace(n0+1,t,t-n0)
timeFromN0 = timeFromN0.astype(int)
# NARX MIL NRMSE
xnarx_all = xnarx[:-1,:]
qoi = 0
ind = xtrue > 0.001
error = xtrue[ind] - xnarx_all[ind]
rmse_mean = np.mean(xtrue[ind],0)
rmse_square = error**2
rmse_each = np.sqrt(np.sum(rmse_square)/np.size(rmse_square))
rmse_mean = np.sqrt(np.sum(xtrue[ind]**2)/np.size(xtrue[ind]))
nrmse_each4 = rmse_each/rmse_mean * 100
print("NARX NRMSE = ",nrmse_each4,"%")
# MovingAncore NRMSE
xma_all = xma[:-1,:]
qoi = 0
ind = xtrue > 0.001
error = xtrue[ind] - xma_all[ind]
rmse_mean = np.mean(xtrue[ind],0)
rmse_square = error**2
rmse_each = np.sqrt(np.sum(rmse_square)/np.size(rmse_square))
rmse_mean = np.sqrt(np.sum(xtrue[ind]**2)/np.size(xtrue[ind]))
nrmse_each5 = rmse_each/rmse_mean * 100
print("MovingAncore NRMSE = ",nrmse_each5,"%")
# Plot comparisons
plotenvironment = env_equalfilling
fig = plt.figure(figsize=(8,8))
for i in range(n):
    fig.add_subplot(n,1, i+1)
    label0 = "P"+str(i+1)+" Ground True"
    label1 = "P"+str(i+1)+" Koopman"
    label2 = "P"+str(i+1)+" DMD"
    label3 = "P"+str(i+1)+" Koopman2"
    label4 = "P"+str(i+1)+" NARX"
    label5 = "P"+str(i+1)+" MovingAnchor"
    plt.plot(timeFromN0, xtrue[qoi:,i],'-',label=label0,color=colors_hex[0])
    plt.plot(timeFromN0, xkp_all[qoi:,i],'-',label=label1,color=colors_hex[1])
    plt.plot(timeFromN0, xdmd_all[qoi:,i],'-',label=label2,color=colors_hex[2])
    plt.plot(timeFromN0, xc_all[qoi:,i],'-',label=label3,color=colors_hex[3])
    plt.plot(timeFromN0, xnarx_all[qoi:,i],'-',label=label4,color=colors_hex[4])
    plt.plot(timeFromN0, xma_all[qoi:,i],'-',label=label5,color='y')
    plt.legend(loc='upper right',prop={'size': 12})
title = "NRMSE_Koopman = "+str(round(nrmse_each1,3))
title += "%, NRMSE_DMD = "+str(round(nrmse_each2,3))
title += "%, NRMSE_Koopman2 = "+str(round(nrmse_each3,3))
title += "%, NRMSE_NARX = "+str(round(nrmse_each4,3))
title += "%, NRMSE_MA = "+str(round(nrmse_each5,3))
title += "%, Sampling T = "+ str(tmulti) + ", InitialDate size = " + str(int(n0))
plt.suptitle(title, fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Simulation step',fontsize=14)
plt.ylabel("Depth",fontsize=14)
plt.show()

fig.add_subplot(2,1, 2)
plt.plot(env_equalfilling.data_log["flow"]["8"][n0:], label="Outflow")
plt.axhline(0.5, color="r",label="Limit = 0.5")
plt.ylabel('Outflows')
plt.legend(loc='upper right',prop={'size': 12})
plt.show()

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
plt.show()
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