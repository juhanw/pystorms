# Import python libraries 
import pystorms
# import sys
# sys.path.append("/controller")
# from denseMPC import MPC
# from MPC_rewrite import MPC
# from MPC_soft import MPC
from MPC_smooth import MPC
# from Koopman import Koopman 
from Koopman_rewrite import Koopman
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
n_basis = 6
n = Xub_extreme.size
m = Uub.size
nk = n + n_basis
n0 = 20    #n0 > nk + m
KPmodel = Koopman(Xub_extreme,Xlb_extreme,Uub,Ulb,nk)
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

            # get control input
            z0 = KPmodel.lift(KPmodel.scale(x0[-1,:])) # x
            ulast_scaled = KPmodel.scale(u[-1,:],state_scale=False)
            actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
            if actions_mpc is None:
                actions_mpc = u[-1,:]
            actions = KPmodel.scale(actions_mpc,scale_down=False,state_scale=False)
            actions = actions.T
            actions_north3.append(actions[4])
            actions_north2.append(actions[3])
            actions_north1.append(actions[2])
            actions_central.append(actions[0])
            actions_south.append(actions[1])
            
            # initialize containers
            xtrue = x0[-1,:].reshape(1,5)
            xkp = xtrue
            umpc = actions.reshape(1,5)
            xkp = np.vstack((xkp,KPmodel.predict(xtrue,umpc) ))

    else:
        done = env_equalfilling.step(umpc[-1,:])
        state = env_equalfilling.state() # y
        # states = [state[2], state[1], state[0], state[4], state[5]]
        xtrue = np.vstack((xtrue,state[:-1].reshape(1,5) ))

        # update Koopman model
        A,B,C = KPmodel.update(xtrue[-1,:],umpc[-1,:])
        xkp_new = KPmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xkp,1))
        xkp = np.vstack((xkp, xkp_new))
        
        # get control input
        z0 = KPmodel.lift(KPmodel.scale(xtrue[-1,:]))
        ulast_scaled = KPmodel.scale(umpc[-1,:],state_scale=False)
        actions_mpc = KMPC.getMPC(z0,ulast_scaled,A,B,C)
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

    print(t, "is time")
    t = t + 1
    # if t > 14000:
    #     break
    
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

# plot on Koopman performance ===================================================
# Evaluate regression accuracy:
xkp_all = xkp[:-1,:]
qoi = 20
error = xtrue - xkp_all
NRMSE_predict = 100*np.sqrt(sum(np.linalg.norm(error[qoi:,:],axis=0)**2)) / np.sqrt(sum(np.linalg.norm(xtrue[qoi:,:],axis=0)**2))
print(NRMSE_predict,"%")
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.plot(xtrue[qoi:,i], '-', label = "Ground Truth")
    plt.plot(xkp_all[qoi:,i], '--', label = "Koopman model")
    plt.title("state"+str(i+1))
plt.legend(loc="upper right", borderaxespad=0.1)
plt.tight_layout()
plt.show()
# print("done")

# '''
# plot on states =============================================================
plotenvironment = env_equalfilling

plt.subplot(2, 3, 1)
plt.plot(plotenvironment.data_log["flow"]["conduit_Eout"], color=colors_hex[0])
plt.axhline(12.0, color="r")
plt.ylim([0,20])
plt.title("Conduit East Out")
plt.ylabel("Flow")

plt.subplot(2, 3, 2)
plt.plot(plotenvironment.data_log["depthN"]["basin_C"], color=colors_hex[0])
plt.axhline(5.7, color="r")
plt.axhline(3.8, color="k")
plt.axhline(3.28, color="k")
plt.axhline(2.21, color="r")
plt.ylim([0,11.5])
plt.ylabel("Depth")
plt.title("Central Basin")


plt.subplot(2, 3, 3)
plt.plot(plotenvironment.data_log["depthN"]["basin_S"], color=colors_hex[0])
plt.axhline(9.5, color="r")
plt.axhline(6.55, color="k")
plt.ylim([0,12])
plt.ylabel("Depth")
plt.title("South Basin")

plt.subplot(2, 3, 4)
plt.plot(plotenvironment.data_log["depthN"]["basin_N1"], color=colors_hex[0])
plt.axhline(5.92, color="r")
plt.axhline(2.11, color="r")
plt.axhline(5.8, color="k")
plt.axhline(5.2, color="k")
plt.ylim([0,12])
plt.ylabel("Depth")
plt.title("Basin North1")

plt.subplot(2, 3, 5)
plt.plot(plotenvironment.data_log["depthN"]["basin_N2"], color=colors_hex[0])
plt.axhline(6.59, color="r")
plt.axhline(5.04, color="k")
plt.axhline(4.44, color="k")
plt.axhline(4.04, color="r")
plt.ylim([0,12])
plt.ylabel("Depth")
plt.title("Basin North2")

plt.subplot(2, 3, 6)
plt.plot(plotenvironment.data_log["depthN"]["basin_N3"], color=colors_hex[0])
plt.axhline(11.99, color="r")
plt.axhline(5.92, color="k")
plt.axhline(5.32, color="k")
plt.axhline(5.28, color="r")
plt.ylim([0,16])
plt.ylabel("Depth")
plt.title("Basin North3")
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