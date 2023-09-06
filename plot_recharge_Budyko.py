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
    return np.sqrt(aridity * np.tanh(1 / aridity) * (1 - np.exp(-aridity)));

def Berghuijs_recharge_curve(aridity):
    alpha = 0.72
    beta = 15.11
    RR = alpha*(1-(np.log(aridity**beta+1)/(1+np.log(aridity**beta+1))))
    return RR

### load data ###

# Caravan
df = pd.read_csv("./results/caravan_processed.csv")
df_Caravan = df
#np.logical_and.reduce((df_Caravan["flow_perc_complete"]>80, df_Caravan["hft_ix_s09"]<100, df_Caravan["urb_pc_sse"]<5, df_Caravan["ire_pc_sse"]<1))]
df_Caravan = df_Caravan.loc[df_Caravan["flow_perc_complete"]>80]
print("Finished Caravan.")

"""
# CAMELS
data_path = "C:/Users/gnann/Documents/MATLAB/CAMELS_Matlab/"
df = pd.read_csv(data_path + "CAMELS_table.csv")
df_CAMELS = df
print("Finished CAMELS.")
"""

# ERA5
ds_ERA5 = xr.open_dataset(results_path + "ERA5_aggregated.nc4") # used to extract aridity for recharge datasets
df_ERA5 = pd.read_csv(results_path + "ERA5_aggregated.csv")
df_ERA5 = df_ERA5.sample(100000) # to reduce size
print("Finished ERA5.")

# Moeck
data_path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"
df = pd.read_csv(data_path + "data/global_groundwater_recharge_moeck-et-al.csv", sep=',')
selected_data = []
for lat, lon in zip(df['Latitude'], df['Longitude']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Latitude"] == lat, df["Longitude"]==lon)]["Groundwater recharge [mm/y]"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Moeck = ds_combined.to_dataframe()
df_Moeck["recharge_ratio"] = df_Moeck["recharge"]/df_Moeck["tp"]
print("Finished Moeck.")

# MacDonald
data_path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"
df = pd.read_csv(data_path + "data/Recharge_data_Africa_BGS.csv", sep=';')
selected_data = []
for lat, lon in zip(df['Lat'], df['Long']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Lat"] == lat, df["Long"]==lon)]["Recharge_mmpa"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_MacDonald = ds_combined.to_dataframe()
df_MacDonald["recharge_ratio"] = df_MacDonald["recharge"]/df_MacDonald["tp"]
print("Finished MacDonald.")

# Cuthbert
data_path = "C:/Users/gnann/Documents/PYTHON/Recharge/results/"
df = pd.read_csv(data_path + "green-roofs_deep_drainage.csv")
df_Cuthbert = df
print("Finished Cuthbert.")

### plot data ###

print("Budy recharge all fluxes")
stat = "median"
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], s=2.5, c="#a6cee3", alpha=0.25, lw=0)
im = axes.scatter(df_Caravan["aridity_netrad"], df_Caravan["BFI"]*df_Caravan["TotalRR"], s=2.5, c="#1f78b4", alpha=0.25, lw=0)
#im = axes.scatter(df_Caravan["aridity_netrad"], 1-(1-df_Caravan["BFI"])*df_Caravan["TotalRR"], s=2.5, c="#947351", alpha=0.25, lw=0)
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=2.5, c="#497a21", alpha=0.25, lw=0)
im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=2.5, c="#7bcb3a", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_ERA5["aridity_netrad"], df_ERA5["e"]/df_ERA5["tp"], "#CEA97C", n=11, label='ET ERA5', statistic=stat)
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], "#a6cee3", n=11, label='Q Caravan', statistic=stat)
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["BFI"]*df_Caravan["TotalRR"], "#1f78b4", n=11, label='Qb Caravan')
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], 1-(1-df_Caravan["BFI"])*df_Caravan["TotalRR"], "#947351", n=11, label='P-Qf Caravan')
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#497a21", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#7bcb3a", n=6, label='MacDonald', statistic=stat)
plotting_fcts.plot_lines_group(df_Cuthbert["PET"]/df_Cuthbert["P"], df_Cuthbert["D(=P-AET)"]/df_Cuthbert["P"], "#A496CF", n=11, label='D/P Cuthbert')
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "--", c="#a6cee3", alpha=0.75, label="Budyko Q")
im = axes.plot(np.linspace(0.1,10,100), Budyko_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Budyko_curve(np.linspace(0.1,10,100)), "--", c="#CEA97C", alpha=0.75, label="Budyko ET")
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs R")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_all_fluxes.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge all fluxes lines only")
stat = "median"
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
plotting_fcts.plot_lines_group(df_ERA5["aridity_netrad"], df_ERA5["e"]/df_ERA5["tp"], "#CEA97C", n=11, label='ET ERA5', statistic=stat)
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], "#a6cee3", n=11, label='Q Caravan', statistic=stat)
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["BFI"]*df_Caravan["TotalRR"], "#1f78b4", n=11, label='Qb Caravan')
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], 1-(1-df_Caravan["BFI"])*df_Caravan["TotalRR"], "#947351", n=11, label='P-Qf Caravan')
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#497a21", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#7bcb3a", n=6, label='MacDonald', statistic=stat)
plotting_fcts.plot_lines_group(df_Cuthbert["PET"]/df_Cuthbert["P"], df_Cuthbert["D(=P-AET)"]/df_Cuthbert["P"], "#A496CF", n=11, label='D/P Cuthbert')
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "--", c="#a6cee3", alpha=0.75, label="Budyko Q")
im = axes.plot(np.linspace(0.1,10,100), Budyko_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Budyko_curve(np.linspace(0.1,10,100)), "--", c="#CEA97C", alpha=0.75, label="Budyko ET")
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs R")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_all_fluxes_lines_only.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=2.5, c="#497a21", alpha=0.25, lw=0)
im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=2.5, c="#7bcb3a", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#497a21", n=11, label='Moeck')
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#7bcb3a", n=6, label='MacDonald')
plotting_fcts.plot_lines_group(df_Cuthbert["PET"]/df_Cuthbert["P"], df_Cuthbert["D(=P-AET)"]/df_Cuthbert["P"], "#A496CF", n=11, label='D/P Cuthbert')
m = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge with shaded areas")
stat = "median"
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1,10,1000), 0*np.linspace(0.1,10,1000), 1-Budyko_curve(np.linspace(0.1,10,1000)), color="tab:blue", alpha=0.1)
axes.fill_between(np.linspace(0.1,10,1000),1-Budyko_curve(np.linspace(0.1,10,1000)), 1+0*np.linspace(0.1,10,1000), color="tab:green", alpha=0.1)
im = axes.plot(np.linspace(0.1,10,1000), Berghuijs_recharge_curve(np.linspace(0.1,10,1000)), "-", c="grey", alpha=0.75, label='Berghuijs')
#plotting_fcts.plot_lines_group(df_ERA5["aridity_netrad"], df_ERA5["e"]/df_ERA5["tp"], "#CEA97C", n=11, label='ET ERA5', statistic=stat)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], "#a6cee3", n=11, label='Qtot', statistic=stat)
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["BFI"]*df_Caravan["TotalRR"], "#1f78b4", n=11, label='Qb')
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#497a21", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#7bcb3a", n=6, label='MacDonald', statistic=stat)
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_shaded_areas.png", dpi=600, bbox_inches='tight')
plt.close()
