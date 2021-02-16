# Import python libraries 
import pystorms
from MPC_smooth import MPC
from Koopman_performance import Koopman
from Koopman_liftCtrl import Koopman2
from DMD_MIL import DMD
# Python Scientific Computing Stack
import numpy as np
import pandas as pd
# Plotting tools
import seaborn as sns
from matplotlib import pyplot as plt

# KMPC
env_equalfilling = pystorms.scenarios.delta() # states = [BC, BS, N1, N2, N3, N4]
done = False

u = []
x0 = []
actions_north3 = []
actions_north2 = []
actions_north1 = []
actions_central = []
actions_south = []
settings = np.ones(5)
Xub_extreme = np.asarray([5.7, 9.5, 5.92, 6.59, 11.99]) # [BC, BS, N1, N2, N3]
Xlb_extreme = np.asarray([2.21, 0, 2.11, 4.04, 5.28])
Xub_allowed = np.asarray([3.8, 6.55, 5.8, 5.04, 5.92])
Xlb_allowed = np.asarray([3.28, 0, 5.2, 4.44, 5.32])
Uub = np.ones((1,5))
Ulb = np.zeros((1,5))

t = 0
tmulti = 50
n_basis = 6
n = Xub_extreme.size
m = Uub.size
n0 = 20    #n0 > nk + m
KPmodel = Koopman(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
KCmodel = Koopman2(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
DMDmodel = DMD(Xub_extreme,Xlb_extreme,Uub,Ulb,n_basis)
Xub_scaled = KPmodel.scale(Xub_extreme)
Xlb_scaled = KPmodel.scale(Xlb_extreme)
Xub_allowed_scaled = KPmodel.scale(Xub_allowed)
Xlb_allowed_scaled = KPmodel.scale(Xlb_allowed)
Uub_scaled = KPmodel.scale(Uub,state_scale=False)
Ulb_scaled = KPmodel.scale(Ulb,state_scale=False)
KMPC = MPC(Uub_scaled,Ulb_scaled, Xub_soft= Xub_scaled, Xlb_soft = Xlb_scaled)

while not done:
    if  t <= n0:
        actions =  np.random.uniform(0,0.5,5)
        u.append(actions)
        actions_north3.append(actions[4])
        actions_north2.append(actions[3])
        actions_north1.append(actions[2])
        actions_central.append(actions[0])
        actions_south.append(actions[1])

        done = env_equalfilling.step(actions)
        state = env_equalfilling.state()
        x0.append(state[:-1])

        if t == n0:
            u = np.asarray(u)
            x0 = np.asarray(x0)
            u.reshape(n0+1,m)
            x0.reshape(n0+1,n)
            
            # initialize Koopman model:
            A,B,C = KPmodel.initialization(x0,u)
            Admd, Bdmd, Cdmd = DMDmodel.initialization(x0,u)
            ABc,Cc = KCmodel.initialization(x0,u)
            t1 = t

            # get control input
            z0 = KPmodel.lift(KPmodel.scale(x0[-1,:])) # x
            ulast_scaled = KPmodel.scale(u[-1,:],state_scale=False)
            # actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
            # if actions_mpc is None:
            #     actions_mpc = u[-1,:]
            # actions = KPmodel.scale(actions_mpc,scale_down=False,state_scale=False)
            # actions = actions.T
            xc_new,uc_new = KCmodel.predict(x0[-1,:].reshape(1,5),u[-1,:].reshape(1,5))
            actions = uc_new
            actions_north3.append(actions[4])
            actions_north2.append(actions[3])
            actions_north1.append(actions[2])
            actions_central.append(actions[0])
            actions_south.append(actions[1])
            
            # initialize containers
            xtrue = x0[-1,:].reshape(1,5)
            umpc = actions.reshape(1,5)
            xkp = xtrue
            xkp = np.vstack((xkp,KPmodel.predict(xtrue,umpc) ))
            xdmd = xtrue
            xdmd = np.vstack((xdmd,DMDmodel.predict(xtrue,umpc) ))
            xc = xtrue
            xc = np.vstack((xc, xc_new))

    else:
        done = env_equalfilling.step(umpc[-1,:])
        state = env_equalfilling.state() # y
        # states = [state[2], state[1], state[0], state[4], state[5]]
        xtrue = np.vstack((xtrue,state[:-1].reshape(1,5) ))

        if t-t1 == tmulti:
            # update Koopman model
            A,B,C = KPmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:])
            xkp_new = KPmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xkp,1))
            xkp[-1,:] = xtrue[-1,:]
            xkp = np.vstack((xkp, xkp_new))
            Admd,Bdmd,Cdmd = DMDmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-1,:])
            xdmd_new = DMDmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xdmd,1))
            xdmd[-1,:] = xtrue[-1,:]
            xdmd = np.vstack((xdmd, xdmd_new))
            ABc,Cc = KCmodel.update(xtrue[-2,:],xtrue[-1,:],umpc[-2,:],umpc[-1,:])
            xc_new,uc_new = KCmodel.predict(xtrue[-1,:],umpc[-1,:])
            xc[-1,:] = xtrue[-1,:]
            xc = np.vstack((xc, xc_new))
            # get control input
            z0 = KPmodel.lift(KPmodel.scale(xtrue[-1,:]))
            ulast_scaled = KPmodel.scale(umpc[-1,:],state_scale=False)
            t1 = t
        else:
            xkp_new = KPmodel.predict(xkp[-1,:],umpc[-1,:]).reshape(1,np.size(xkp,1))
            xkp = np.vstack((xkp, xkp_new))
            xdmd_new = DMDmodel.predict(xdmd[-1,:],umpc[-1,:]).reshape(1,np.size(xdmd,1))
            xdmd = np.vstack((xdmd, xdmd_new))
            xc_new,uc_new = KCmodel.predict(xc[-1,:],umpc[-1,:])
            xc = np.vstack((xc, xc_new))
            # get control input
            z0 = KPmodel.lift(KPmodel.scale(xkp[-1,:]))
            ulast_scaled = KPmodel.scale(umpc[-1,:],state_scale=False)
        
        # actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
        actions_mpc = uc_new
        if actions_mpc is None:
            actions_mpc = umpc[-1,:]
        actions = KPmodel.scale(actions_mpc,scale_down=False,state_scale=False)
        actions = actions.T
        actions_north3.append(actions[4])
        actions_north2.append(actions[3])
        actions_north1.append(actions[2])
        actions_central.append(actions[0])
        actions_south.append(actions[1])
        umpc = np.vstack((umpc,actions.reshape(1,np.size(umpc,1)) ))
        xkp_all = xkp[:-1,:]
        xdmd_all = xdmd[:-1,:]
        xc_all = xc[:-1,:]
        qoi = 0
        error = xtrue - xkp_all
        errDMD = xtrue - xdmd_all
        errC = xtrue - xc_all
        rmse_square = error**2
        rmse_each = np.sqrt(np.sum(rmse_square,0)/np.size(rmse_square,0))
        rmse_mean = np.mean(xtrue,0)
        nrmse_each = rmse_each/rmse_mean * 100
        # NRMSE_predict = 100*np.sqrt(sum(np.linalg.norm(error[qoi:,:],axis=0)**2)) / np.sqrt(sum(np.linalg.norm(xtrue[qoi:,:],axis=0)**2))
        if nrmse_each.any() < 0.5 and qoi > 0:
            print(nrmse_each)

    print(t, "is time")
    t = t + 1
    if t > 28000: #45957:
        # break
        print(t, "is time")
    
equalfilling_perf = sum(env_equalfilling.data_log["performance_measure"])



################################################################################
# Plotting
################################################################################
# set seaborn figure preferences and colors
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 1})
sns.set_style("darkgrid")
colorpalette = sns.color_palette("colorblind")
colors_hex = colorpalette.as_hex()
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
# Plot comparisons
plotenvironment = env_equalfilling
label0 = "Ground Truth"
label1 = "Koopman"
label2 = "DMD"
label3 = "Koopman2"
title = "NRMSE_Koopman = "+str(round(nrmse_each1,3))
title += "%, NRMSE_DMD = "+str(round(nrmse_each2,3))
title += "%, NRMSE_Koopman2 = "+str(round(nrmse_each3,3))
title += "%, Sampling T = "+ str(tmulti) + ", InitialDate size = " + str(int(n0))
plt.suptitle(title, fontsize=10)
plt.subplot(2, 3, 1)
plt.plot(timeFromN0, xtrue[qoi:,0], color=colors_hex[0],label = label0)
plt.plot(timeFromN0, xkp_all[qoi:,0], label = label1,color=colors_hex[2])
plt.plot(timeFromN0, xdmd_all[qoi:,0], label = label2,color=colors_hex[1])
plt.plot(timeFromN0, xc_all[qoi:,0], label = label3,color=colors_hex[3])
plt.legend(loc="upper right", borderaxespad=0.1,prop={'size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Simulation step',fontsize=14)
plt.axhline(5.7, color="r")
# plt.axhline(3.8, color="k")
# plt.axhline(3.28, color="k")
plt.axhline(2.21, color="r")
plt.ylim([0,11.5])
plt.ylabel("Depth",fontsize=14)
plt.title("Central Basin")

plt.subplot(2, 3, 2)
plt.plot(timeFromN0, xtrue[qoi:,1], color=colors_hex[0],label = label0)
plt.plot(timeFromN0, xkp_all[qoi:,1], label = label1,color=colors_hex[2])
plt.plot(timeFromN0, xdmd_all[qoi:,1], label = label2,color=colors_hex[1])
plt.plot(timeFromN0, xc_all[qoi:,1], label = label3,color=colors_hex[3])
plt.legend(loc="upper right", borderaxespad=0.1,prop={'size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Simulation step',fontsize=14)
plt.axhline(9.5, color="r")
# plt.axhline(6.55, color="k")
plt.ylim([0,12])
plt.ylabel("Depth",fontsize=14)
plt.title("South Basin")

plt.subplot(2, 3, 3)
plt.plot(timeFromN0, xtrue[qoi:,2], color=colors_hex[0],label = label0)
plt.plot(timeFromN0, xkp_all[qoi:,2], label = label1,color=colors_hex[2])
plt.plot(timeFromN0, xdmd_all[qoi:,2], label = label2,color=colors_hex[1])
plt.plot(timeFromN0, xc_all[qoi:,2], label = label3,color=colors_hex[3])
plt.legend(loc="upper right", borderaxespad=0.1,prop={'size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Simulation step',fontsize=14)
plt.axhline(5.92, color="r")
plt.axhline(2.11, color="r")
# plt.axhline(5.8, color="k")
# plt.axhline(5.2, color="k")
plt.ylim([0,12])
plt.ylabel("Depth",fontsize=14)
plt.title("Basin North1")

plt.subplot(2, 3, 4)
plt.plot(timeFromN0, xtrue[qoi:,3], color=colors_hex[0],label = label0)
plt.plot(timeFromN0, xkp_all[qoi:,3], label = label1,color=colors_hex[2])
plt.plot(timeFromN0, xdmd_all[qoi:,3], label = label2,color=colors_hex[1])
plt.plot(timeFromN0, xc_all[qoi:,3], label = label3,color=colors_hex[3])
plt.legend(loc="upper right", borderaxespad=0.1,prop={'size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Simulation step',fontsize=14)
plt.axhline(6.59, color="r")
# plt.axhline(5.04, color="k")
# plt.axhline(4.44, color="k")
plt.axhline(4.04, color="r")
plt.ylim([0,12])
plt.ylabel("Depth",fontsize=14)
plt.title("Basin North2")

plt.subplot(2, 3, 5)
plt.plot(timeFromN0, xtrue[qoi:,4], color=colors_hex[0],label = label0)
plt.plot(timeFromN0, xkp_all[qoi:,4], label = label1,color=colors_hex[2])
plt.plot(timeFromN0, xdmd_all[qoi:,4], label = label2,color=colors_hex[1])
plt.plot(timeFromN0, xc_all[qoi:,4], label = label3,color=colors_hex[3])
plt.legend(loc="upper right", borderaxespad=0.1,prop={'size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Simulation step',fontsize=14)
plt.axhline(11.99, color="r")
# plt.axhline(5.92, color="k")
# plt.axhline(5.32, color="k")
plt.axhline(5.28, color="r")
plt.ylim([0,16])
plt.ylabel("Depth",fontsize=14)
plt.title("Basin North3")
# plt.show()

plt.subplot(2, 3, 6)
plt.plot(actions_north3, label='North3 Weir', linestyle='-', linewidth=3.0)
plt.plot(actions_north2, label='North2 Weir', linestyle='-', linewidth=2.0)
plt.plot(actions_north1, label='North1 Weir', linestyle='-', linewidth=2.0)
plt.plot(actions_central, label='Central Weir', linestyle='-', linewidth=2.0)
plt.plot(actions_south, label='South Orifice', linestyle='-', linewidth=2.0)
plt.ylim([-0.1,1.1])
plt.legend(loc='upper right',prop={'size': 14})
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Simulation step',fontsize=14)
plt.ylabel('Control asset setting',fontsize=14)
plt.title('Koopman-based MPC')
plt.show()

# new plot on control =============================================================
plt.rcParams['figure.figsize'] = [6, 6]

plt.plot(actions_north3, label='North3 Weir', linestyle='-', linewidth=3.0)
plt.plot(actions_north2, label='North2 Weir', linestyle='--', linewidth=2.0)
plt.plot(actions_north1, label='North1 Weir', linestyle='--', linewidth=2.0)
plt.plot(actions_central, label='Central Weir', linestyle='-', linewidth=2.0)
plt.plot(actions_south, label='South Orifice', linestyle='-.', linewidth=2.0)

plt.ylim([-0.1,1.1])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle('Control settings')
plt.ylabel('Control asset setting')
plt.xlabel('Simulation step')
plt.show()
# '''