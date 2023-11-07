import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts
import geopandas as gpd
import xarray as xr
import rasterio as rio

# This script loads and analyses different datasets in Budyko space.

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# define functions
def Budyko_curve(aridity, **kwargs):
    # Budyko, M.I., Miller, D.H. and Miller, D.H., 1974. Climate and life (Vol. 508). New York: Academic press.
    return np.sqrt(aridity * np.tanh(1 / aridity) * (1 - np.exp(-aridity)))

def Berghuijs_recharge_curve(aridity):
    alpha = 0.72
    beta = 15.11
    RR = alpha*(1-(np.log(aridity**beta+1)/(1+np.log(aridity**beta+1))))
    return RR

def fast_flow_curve(P, Q, Smax, **kwargs):
    # Smax = max storage
    Qf = P - Smax
    Qf[Qf<0] = 0
    Qf[Qf>Q] = Q[Qf>Q]
    return Qf

stat = "median"

BFI = 0.3# let's assume that BFI = 0.5, i.e. Qb = 0.5*Q
AI = np.linspace(0.1,10,100)
Q_P = 1-Budyko_curve(AI)
E_P = Budyko_curve(AI)
R_P = Berghuijs_recharge_curve(AI)
Qb_P = BFI*Q_P
Qf_P = (1-BFI)*Q_P
Eb_P = R_P - Qb_P
Ef_P = E_P - Eb_P

print("Budyko recharge all fluxes")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.plot(AI, R_P, "-", c="#b6d7a8", alpha=0.9, label="Berghuijs R")
im = axes.plot(AI, Q_P, "-", c="#69868a", alpha=0.9, label="Budyko Q")
im = axes.plot(AI, Qb_P, "-", c="#8cb3b8", alpha=0.9, label="Qb = BFI*Q")
im = axes.plot(AI, Qf_P, "-", c="#b0e0e6", alpha=0.9, label="Qf = (1-BFI)*Q")
im = axes.plot(AI, E_P, "-", c="#996941", alpha=0.9, label="Budyko ET")
im = axes.plot(AI, Eb_P, "-", c="#cc8c57", alpha=0.9, label="ETb = R - Qb")
im = axes.plot(AI, Ef_P, "-", c="#ffb06d", alpha=0.9, label="ETf = ET - ETb")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
axes.set_title('BFI = '+ str(BFI))
fig.savefig(figures_path + "toy_model_all_fluxes_BFI_" + str(BFI) + ".png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")

# todo: play with functions etc.

#AI = np.linspace(0.1,10,100)
P = np.round(np.random.rand(1000)*3000)
PET = np.round(np.random.rand(1000)*2000)
AI = PET/P
#Smax = np.round(np.random.rand(100)*1000)
Q_P = 1-Budyko_curve(AI)
E_P = Budyko_curve(AI)
Qf_P = fast_flow_curve(P, Q_P*P, 1000)/P
#R_P = 1-Budyko_curve(PET/(P-Qf_P*P))
R_P = Berghuijs_recharge_curve(PET/(P-Qf_P*P))
Qb_P = Q_P - Qf_P
#Qb_P = BFI*Q_P
#Qf_P = (1-BFI)*Q_P
Eb_P = R_P - Qb_P
Eb_P[Eb_P > E_P] = E_P[Eb_P > E_P]
Eb_P[Eb_P < 0] = 0
Ef_P = E_P - Eb_P

print("Budyko recharge all fluxes")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.plot(AI, R_P, ".", c="#b6d7a8", alpha=0.75, label="R")
im = axes.plot(AI, Q_P, ".", c="#69868a", alpha=0.75, label="Q")
im = axes.plot(AI, Qb_P, ".", c="#8cb3b8", alpha=0.75, label="Qb")
im = axes.plot(AI, Qf_P, ".", c="#b0e0e6", alpha=0.75, label="Qf")
im = axes.plot(AI, E_P, ".", c="#996941", alpha=0.75, label="ET")
im = axes.plot(AI, Eb_P, ".", c="#cc8c57", alpha=0.75, label="ETb")
im = axes.plot(AI, Ef_P, ".", c="#ffb06d", alpha=0.75, label="ETf")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
#axes.set_title('x')
fig.savefig(figures_path + "toy_model_all_fluxes_Smax.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")

from scipy.stats import lognorm
# todo: play with functions etc.
n=10000
#AI = np.linspace(0.1,10,100)
P = np.round(np.random.rand(n)*3000)
PET = np.round(np.random.rand(n)*2000)
AI = PET/P
#R_P = np.random.rand(n) # random fraction
R_P = Berghuijs_recharge_curve(AI)
#R_P = R_P*np.random.rand(n)
#R = np.copy(P)
#Smax = np.round(np.random.rand(n)*1000) # max recharge capacity
#Smax = 500
#R = R_P*P
#R[R>Smax] = Smax
#R_P = R/P
#R[R>Smax] = Smax[R>Smax]
#R_P = R/P
#dist=lognorm([0.75],loc=0.1)
#R_P = dist.pdf(AI)
Q_P = 1-Budyko_curve(AI)
E_P = Budyko_curve(AI)
BFI = np.random.rand(n) # random fraction
Qb_P = BFI*Q_P
Qf_P = (1-BFI)*Q_P
Eb_P = R_P - Qb_P
Ef_P = E_P - Eb_P

wb_ok = np.full((n), True)
wb_ok[Ef_P<0] = False
wb_ok[Ef_P>E_P] = False
wb_ok[Eb_P<0] = False
wb_ok[Eb_P>E_P] = False
wb_ok[Qf_P<0] = False
wb_ok[Qf_P>Q_P] = False
wb_ok[Qb_P<0] = False
wb_ok[Qb_P>Q_P] = False

a = 0.5
print("Budyko recharge all fluxes")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.plot(AI[wb_ok], Q_P[wb_ok], ".", c="#69868a", alpha=a, label="Q")
im = axes.plot(AI[wb_ok], Qb_P[wb_ok], ".", c="#8cb3b8", alpha=a, label="Qb")
im = axes.plot(AI[wb_ok], Qf_P[wb_ok], ".", c="#b0e0e6", alpha=a, label="Qf")
im = axes.plot(AI[wb_ok], E_P[wb_ok], ".", c="#996941", alpha=a, label="ET")
im = axes.plot(AI[wb_ok], Eb_P[wb_ok], ".", c="#cc8c57", alpha=a, label="ETb")
im = axes.plot(AI[wb_ok], Ef_P[wb_ok], ".", c="#ffb06d", alpha=a, label="ETf")
im = axes.plot(AI[wb_ok], R_P[wb_ok], ".", c="#b6d7a8", alpha=a, label="R")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.2, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
#axes.set_title('x')
fig.savefig(figures_path + "toy_model_all_fluxes.png", dpi=600, bbox_inches='tight')
plt.close()

print(str(BFI[np.logical_and(wb_ok,AI<1)].mean()))
print(str(BFI[np.logical_and(wb_ok,AI>1)].mean()))
print(str(BFI[np.logical_and(wb_ok,AI<0.5)].mean()))

print("Finished plotting data.")

