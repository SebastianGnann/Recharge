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

# Moeck
df = pd.read_csv("./results/global_groundwater_recharge_moeck-et-al.csv", sep=',')
selected_data = []
for lat, lon in zip(df['Latitude'], df['Longitude']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Latitude"]==lat, df["Longitude"]==lon)]["Groundwater recharge [mm/y]"].values[0]
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
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_MacDonald = ds_combined.to_dataframe()
df_MacDonald["recharge_ratio"] = df_MacDonald["recharge"]/df_MacDonald["tp"]
print("Finished MacDonald.")

# Hartmann
# copied from PNAS SI
df = pd.read_csv("./results/recharge_Hartmann_PNAS.csv", sep=',')
selected_data = []
for lat, lon in zip(df['Lat'], df['Lon']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["Lat"]==lat, df["Lon"]==lon)]["Recharge"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Hartmann = ds_combined.to_dataframe()
df_Hartmann["recharge_ratio"] = df_Hartmann["recharge"]/df_Hartmann["tp"]
print("Finished Hartmann.")

### plot data ###

stat = "median"

print("Precipitation and recharge")
fig = plt.figure(figsize=(3, 2.75), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Moeck["tp"], df_Moeck["recharge"], s=10, c="skyblue", alpha=0.1, lw=0)
im = axes.scatter(df_MacDonald["tp"], df_MacDonald["recharge"], s=10, c="tab:blue", alpha=0.9, lw=0)
im = axes.scatter(df_Hartmann["tp"], df_Hartmann["recharge"], s=10, c="tab:orange", alpha=0.9, lw=0)
plotting_fcts.plot_lines_group(df_Moeck["tp"], df_Moeck["recharge"], "skyblue", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["tp"], df_MacDonald["recharge"], "tab:blue", n=6, label='MacDonald', statistic=stat)
plotting_fcts.plot_lines_group(df_Hartmann["tp"], df_Hartmann["recharge"], "tab:orange", n=6, label='Hartmann', statistic=stat)
axes.set_xlabel("P [mm/yr]")
axes.set_ylabel("Recharge [mm/yr]")
axes.set_xlim([0, 2200])
axes.set_ylim([0.1, 2000])
axes.legend()
#axes.axline((0, 0), slope=1, c='silver', label='1:1 line', linestyle='--')
#axes.grid()
axes.set_yscale('log')
#plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "precipitation_recharge_karst.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")
