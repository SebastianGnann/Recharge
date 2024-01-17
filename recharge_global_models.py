import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts
import geopandas as gpd
import xarray as xr
import rasterio as rio
from scipy.stats import lognorm
from functions.plotting_fcts import get_binned_range
from scipy.optimize import curve_fit
from scipy.stats import gamma
from scipy.stats import beta
from scipy.stats import johnsonsu
from functions.easyit.easyit import load_data_all
from functions import plotting_fcts
#import seaborn as sns

# This script loads and analyses different models in Budyko space.

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


### load data ###
data_path = "D:/Python/GHM_Comparison/model_outputs/2b/aggregated/"
ghms = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]
outputs = ["evap", "qr", "qs", "qsb", "qtot", "netrad", "potevap"]
forcings = ["pr", "rlds", "rsds", "tas", "tasmax", "tasmin", "domains"] # domains contains pr, potevap, netrad

df = load_data_all(data_path, forcings, outputs, rmv_outliers=True)

df["netrad"] = df["netrad"] * 12.87 # transform radiation into mm/y
df["totrad"] = df["rlds"] + df["rsds"]
df["totrad"] = df["totrad"] / 2257 * 0.001 * (60 * 60 * 24 * 365) # transform radiation into mm/y

#domains = ["wet warm", "wet cold", "dry cold", "dry warm"]

print("Finished loading data.")

### define functions ###

models = ["clm45", "cwatm", "h08", "jules-w1", "lpjml", "matsiro", "pcr-globwb", "watergap2"]
for model in models:
    df_tmp = df[df["ghm"]==model]

    stat = "median"
    a = 0.05

    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    axes = plt.axes()
    axes.plot(df_tmp["aridity_netrad"], df_tmp["qsb"]/df_tmp["pr"], ".", markersize=2, c="#073763", alpha=a)
    axes.plot(df_tmp["aridity_netrad"], df_tmp["qtot"]/df_tmp["pr"], ".", markersize=2, c="#0b5394", alpha=a)
    axes.plot(df_tmp["aridity_netrad"], df_tmp["evap"]/df_tmp["pr"], ".", markersize=2, c="#38761D", alpha=a)
    axes.plot(df_tmp["aridity_netrad"], df_tmp["qr"]/df_tmp["pr"], ".", markersize=2, c="grey", alpha=a)
    plotting_fcts.plot_lines_group(df_tmp["aridity_netrad"], df_tmp["qsb"]/df_tmp["pr"], "#073763", n=11, label='Qb', statistic=stat, uncertainty=False)
    plotting_fcts.plot_lines_group(df_tmp["aridity_netrad"], df_tmp["qtot"]/df_tmp["pr"], "#0b5394", n=11, label='Q', statistic=stat, uncertainty=False)
    plotting_fcts.plot_lines_group(df_tmp["aridity_netrad"], df_tmp["evap"]/df_tmp["pr"], "#38761D", n=11, label='E', statistic=stat, uncertainty=False)
    plotting_fcts.plot_lines_group(df_tmp["aridity_netrad"], df_tmp["qr"]/df_tmp["pr"], "grey", n=11, label='Recharge', statistic=stat, uncertainty=False, linestyle='--')
    axes.set_xlabel("PET / P [-]")
    axes.set_ylabel("Flux / P [-]")
    axes.set_xlim([0.2, 5])
    axes.set_ylim([-0.1, 1.1])
    axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
    axes.set_xscale('log')
    plotting_fcts.plot_grid(axes)
    fig.savefig(figures_path + "models/Budyko_recharge_scatter_"+model+".png", dpi=600, bbox_inches='tight')
    plt.close()

#latitude_plots(df)
