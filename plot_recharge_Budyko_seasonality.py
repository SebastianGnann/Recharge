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

# ERA5
ds_ERA5 = xr.open_dataset(results_path + "ERA5_aggregated.nc4") # used to extract aridity for recharge datasets
df_ERA5 = pd.read_csv(results_path + "ERA5_aggregated.csv")
df_ERA5 = df_ERA5.sample(100000) # to reduce size
print("Finished ERA5.")

# BIOCLIM ERA5
BIO_path = "D:/Data/BIO_era5/"
ds_BIO15 = xr.open_dataset(BIO_path + "BIO15_era5_1979-2018-mean_v1.0.nc")
ds_BIO18 = xr.open_dataset(BIO_path + "BIO18_era5_1979-2018-mean_v1.0.nc")
ds_BIO19 = xr.open_dataset(BIO_path + "BIO19_era5_1979-2018-mean_v1.0.nc")
ds_BIO12 = xr.open_dataset(BIO_path + "BIO12_era5_1979-2018-mean_v1.0.nc")
conversion_factor = 3600*24*91.3*1000 #

# Moeck
df = pd.read_csv("./results/global_groundwater_recharge_moeck-et-al.csv", sep=',')
selected_data = []
for lat, lon in zip(df['Latitude'], df['Longitude']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Latitude"]==lat, df["Longitude"]==lon)]["Groundwater recharge [mm/y]"].values[0]
    data_point["BIO15"] = ds_BIO15.sel(latitude=lat, longitude=lon, method='nearest')["BIO15"].values[0]
    data_point["BIO18"] = ds_BIO18.sel(latitude=lat, longitude=lon, method='nearest')["BIO18"].values[0]
    data_point["BIO19"] = ds_BIO19.sel(latitude=lat, longitude=lon, method='nearest')["BIO19"].values[0]
    data_point["BIO12"] = ds_BIO12.sel(latitude=lat, longitude=lon, method='nearest')["BIO12"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Moeck = ds_combined.to_dataframe()
df_Moeck["recharge_ratio"] = df_Moeck["recharge"]/df_Moeck["tp"]
print("Finished Moeck.")

# MacDonald
df = pd.read_csv("./results/Recharge_data_Africa_BGS.csv", sep=';')
selected_data = []
for lat, lon in zip(df['Lat'], df['Long']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Lat"]==lat, df["Long"]==lon)]["Recharge_mmpa"].values[0]
    data_point["BIO15"] = ds_BIO15.sel(latitude=lat, longitude=lon, method='nearest')["BIO15"].values[0]
    data_point["BIO18"] = ds_BIO18.sel(latitude=lat, longitude=lon, method='nearest')["BIO18"].values[0]
    data_point["BIO19"] = ds_BIO19.sel(latitude=lat, longitude=lon, method='nearest')["BIO19"].values[0]
    data_point["BIO12"] = ds_BIO12.sel(latitude=lat, longitude=lon, method='nearest')["BIO12"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_MacDonald = ds_combined.to_dataframe()
df_MacDonald["recharge_ratio"] = df_MacDonald["recharge"]/df_MacDonald["tp"]
print("Finished MacDonald.")

"""
# Hartmann
# copied from PNAS SI
df = pd.read_csv("./results/recharge_Hartmann_PNAS.csv", sep=',')
selected_data = []
for lat, lon in zip(df['Lat'], df['Lon']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Lat"]==lat, df["Lon"]==lon)]["Recharge"].values[0]
    data_point["BIO15"] = ds_BIO15.sel(latitude=lat, longitude=lon, method='nearest')["BIO15"].values[0]
    data_point["BIO18"] = ds_BIO18.sel(latitude=lat, longitude=lon, method='nearest')["BIO18"].values[0]
    data_point["BIO19"] = ds_BIO19.sel(latitude=lat, longitude=lon, method='nearest')["BIO19"].values[0]
    data_point["BIO12"] = ds_BIO12.sel(latitude=lat, longitude=lon, method='nearest')["BIO12"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Hartmann = ds_combined.to_dataframe()
df_Hartmann["recharge_ratio"] = df_Hartmann["recharge"]/df_Hartmann["tp"]
print("Finished Hartmann.")
"""

### plot data ###

stat = "median"

print("Budyko recharge seasonality")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
cm = plt.cm.get_cmap('RdYlBu')
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=5, c=(df_Moeck["BIO19"]*91.3)/(df_Moeck["BIO12"]*365), alpha=0.75, lw=0, cmap=cm, vmin=0, vmax=0.5)
im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=10, c=(df_MacDonald["BIO19"]*91.3)/(df_MacDonald["BIO12"]*365), alpha=0.75, lw=0, cmap=cm, vmin=0, vmax=0.5)
#plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#497a21", n=11, label='Moeck', statistic=stat)
#plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#7bcb3a", n=6, label='MacDonald', statistic=stat)
#m = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
#im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
#axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plt.colorbar(im)
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_seasonality2.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge seasonality")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
cm = plt.cm.get_cmap('RdYlBu')
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=5, c=(df_Moeck["BIO19"]-df_Moeck["BIO18"])*conversion_factor, alpha=0.75, lw=0, cmap=cm, vmin=-500, vmax=500)
im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=10, c=(df_MacDonald["BIO19"]-df_MacDonald["BIO18"])*conversion_factor, alpha=0.75, lw=0, cmap=cm, vmin=-500, vmax=500)
#plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#497a21", n=11, label='Moeck', statistic=stat)
#plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#7bcb3a", n=6, label='MacDonald', statistic=stat)
#m = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
#im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
#axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plt.colorbar(im)
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_seasonality.png", dpi=600, bbox_inches='tight')
plt.close()

print("Precipitation and recharge")
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
cm = plt.cm.get_cmap('RdYlBu')
im = axes.scatter(df_Moeck["tp"], df_Moeck["recharge"], s=5, c=(df_Moeck["BIO19"]-df_Moeck["BIO18"])*conversion_factor, alpha=0.75, lw=0, cmap=cm, vmin=-500, vmax=500)
im = axes.scatter(df_MacDonald["tp"], df_MacDonald["recharge"], s=10, c=(df_MacDonald["BIO19"]-df_MacDonald["BIO18"])*conversion_factor, alpha=0.75, lw=0, cmap=cm, vmin=-500, vmax=500)
axes.set_xlabel("P [mm/yr]")
axes.set_ylabel("Recharge [mm/yr]")
axes.set_xlim([0, 2200])
axes.set_ylim([0, 1200])
#axes.legend()
#axes.axline((0, 0), slope=1, c='silver', label='1:1 line', linestyle='--')
#axes.grid()
#axes.set_yscale('log')
plt.colorbar(im)
#plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "precipitation_recharge_seasonality.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=2.5, c="#497a21", alpha=0.25, lw=0)
im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=2.5, c="#7bcb3a", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#497a21", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#7bcb3a", n=6, label='MacDonald', statistic=stat)
m = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")
