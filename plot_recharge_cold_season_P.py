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

### plot data ###

stat = "median"

print("Budyko recharge seasonality")
fig = plt.figure(figsize=(5, 3), constrained_layout=True)
axes = plt.axes()
cm = plt.cm.get_cmap('plasma')
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=5, c=(df_Moeck["BIO19"]*91.3)/(df_Moeck["BIO12"]*365), alpha=0.75, lw=0, cmap=cm, vmin=0, vmax=0.5)
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.25, 20])
axes.set_ylim([0, 1.2])
#axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plt.colorbar(im)
plt.axhline(y=1.0, color='lightgrey', linestyle='--', linewidth=1)
#plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "recharge_cold_season_P.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")
