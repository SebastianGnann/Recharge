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

# Konikow
df = pd.read_csv("./results/konikow.csv", sep=',')
#df = df.iloc[:-3]
selected_data = []
for lat, lon in zip(df['Lat'], df['Lon']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["Capture Fraction"] = \
        df.loc[np.logical_and(df["Lat"]==lat, df["Lon"]==lon)]["CAPTURE FRACTION"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_konikow = ds_combined.to_dataframe()
print("Finished Konikow.")

### plot data ###

stat = "median"

x = df_konikow["aridity_netrad"]
y = df_konikow["Capture Fraction"]
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

print("Konikow")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_konikow["aridity_netrad"], df_konikow["Capture Fraction"], s=10, c="tab:orange", alpha=0.9, lw=0)
#axes.plot(x, p(x), "tab:orange", lw=2)
plotting_fcts.plot_lines_group(df_konikow["aridity_netrad"], df_konikow["Capture Fraction"], "tab:orange", n=6, label='Konikow', statistic=stat)
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Capture Fraction [-]")
axes.set_xlim([0.2, 10])
axes.set_ylim([-0.1, 1.1])
#axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "aridity_capture.png", dpi=600, bbox_inches='tight')
plt.close()

x = df_konikow["tp"]
y = 1 - df_konikow["Capture Fraction"]
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

print("Konikow")
fig = plt.figure(figsize=(4, 3), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_konikow["tp"], 1-df_konikow["Capture Fraction"], s=10, c="tab:orange", alpha=0.9, lw=0)
axes.plot(x, p(x), "tab:orange", lw=2)
#plotting_fcts.plot_lines_group(df_konikow["tp"], 1-df_konikow["Capture Fraction"], "tab:orange", n=3, label='Konikow', statistic=stat)
axes.set_xlabel("P [mm/yr]")
axes.set_ylabel("Depletion Fraction [-]")
axes.set_xlim([0, 1500])
axes.set_ylim([0, 1])
#axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
#axes.set_xscale('log')
#plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "precipitation_depletion.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")
