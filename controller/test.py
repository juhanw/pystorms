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
# x0 = env_equalfilling.state()[1:]
# x0 = x0.reshape(1,5)
actions_north3 = []
actions_north2 = []
actions_north1 = []
actions_central = []
actions_south = []
settings = np.ones(5)
Xub = np.asarray([5.7, 9.5, 5.92, 6.59, 11.99])
Xlb = np.asarray([2.21, 0, 2.11, 4.04, 5.28])
# Xub = 12*np.ones((1,5))
# Xlb = 0*np.ones((1,5))
Uub = np.ones((1,5))
Ulb = np.zeros((1,5))

t = 0
nk = 4 + 5
n0 = 100
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
        # states = [state[2], state[1], state[0], state[4], state[5]]
        # x0 = np.vstack((x0,state[1:].reshape(1,5)))
        x0.append(state[:-1])

        if t == n0:
            u = np.asarray(u)
            x0 = np.asarray(x0)
            u.reshape(n0+1,5)
            x0.reshape(n0+1,5)
            
            # initialize Koopman model:
            # initData = np.vstack((x0,u))
            # operator = KPmodel.initOperator(initData)
            operator = KPmodel.initOperator(x0,u,Xub,Xlb,Uub,Ulb)
            A = operator[:nk,:nk]
            B = operator[:nk,nk:]
            C = operator[nk:,:nk]
            
            # initialize controller
            Zub = KPmodel.basis(Xub)
            Zlb = KPmodel.basis(Xlb)
            # KMPC = denseMPC(A,B,C,Uub=Uub,Ulb=Ulb,Zub=Zub,Zlb=Zlb)
            KMPC = denseMPC(A,B,C,Uub=Uub,Ulb=Ulb,Xub=Xub,Xlb=Xlb)
    
            # get control input
            z0 = KPmodel.basis(x0[-1,:]) # x
            actions = KMPC.getMPC(z0,A,B,C)
            actions_north3.append(actions[0])
            actions_north2.append(actions[1])
            actions_north1.append(actions[2])
            actions_central.append(actions[3])
            actions_south.append(actions[4])
            
            # initialize containers
            xtrue = x0[-1,:].reshape(1,5)
            xkp = xtrue
            umpc = actions.reshape(1,5)
            xkp = np.vstack((xkp,KPmodel.predict(xtrue,umpc) ))
            # done = env_equalfilling.step(umpc[-1,:])
            # state = env_equalfilling.state() # y
            # xtrue = np.hstack((xtrue,state[1:].reshape(1,5)))

    else:
        done = env_equalfilling.step(umpc[-1,:])
        state = env_equalfilling.state() # y
        # states = [state[2], state[1], state[0], state[4], state[5]]
        xtrue = np.vstack((xtrue,state[:-1].reshape(1,5) ))

        # update Koopman model
        operator = KPmodel.updateOperator(xtrue[-1,:],umpc[-1,:])
        A = operator[:nk,:nk]
        B = operator[:nk,nk:]
        C = operator[nk:,:nk]
        xkp = np.vstack((xkp,KPmodel.predict(xtrue[-1,:],umpc[-1,:]).reshape(1,np.size(xkp,1)) ))
        
        # get control input
        z0 = KPmodel.basis(xtrue[-1,:])
        actions = KMPC.getMPC(z0,A,B,C)
        actions_north3.append(actions[0])
        actions_north2.append(actions[1])
        actions_north1.append(actions[2])
        actions_central.append(actions[3])
        actions_south.append(actions[4])
        umpc = np.vstack((umpc,actions.reshape(1,np.size(umpc,1)) ))

        # # record
        # umpc = np.vstack((umpc,actions.reshape(1,np.size(umpc,1)) ))
        # done = env_equalfilling.step(actions)
        # state = env_equalfilling.state() # y
        # xtrue = np.vstack((xtrue,state[1:].reshape(1,5) ))

    print(t, "is time")
    t = t + 1
    if t > 300:
        break
    
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
# t_plot = range(len(xkp))
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.plot(xtrue[:,i], '-')
    plt.plot(xkp[:,i], '--')

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