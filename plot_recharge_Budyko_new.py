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
df_Caravan = pd.read_csv("./results/caravan_processed.csv")
#df_Caravan = df_Caravan.loc[np.logical_and.reduce((df_Caravan["flow_perc_complete"]>80, df_Caravan["hft_ix_s09"]<100, df_Caravan["urb_pc_sse"]<5, df_Caravan["ire_pc_sse"]<1))]
df_Caravan = df_Caravan.loc[df_Caravan["flow_perc_complete"]>80]
df_Caravan2 = pd.read_csv("./results/caravan_processed_camels.csv")
print("Finished Caravan.")

# CAMELS
df_CAMELS = pd.read_csv("./results/CAMELS_table.csv")
print("Finished CAMELS.")

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

# Lee
# HESS preprint
#df_Lee2 = pd.read_csv("./results/dat07_u.csv", sep=',')
df = pd.read_csv("./results/dat07_u.csv", sep=',')
selected_data = []
for lat, lon in zip(df['lat'], df['lon']):
    data_point = ds_ERA5.sel(latitude=lat, longitude=lon, method='nearest')#['tp']#.values()
    data_point["recharge"] = \
        df.loc[np.logical_and(df["lat"]==lat, df["lon"]==lon)]["Recharge mean mm/y"].values[0]
    selected_data.append(data_point)
ds_combined = xr.concat(selected_data, dim='time')
df_Lee = ds_combined.to_dataframe()
df_Lee["recharge_ratio"] = df_Lee["recharge"]/df_Lee["tp"]
print("Finished Lee.")

# Cuthbert
df_Cuthbert = pd.read_csv("./results/green-roofs_deep_drainage.csv")
print("Finished Cuthbert.")

### plot data ###

# todo: remove snowy catchments ...

stat = "median"

print("Budyko recharge all fluxes lines only")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
plotting_fcts.plot_lines_group(df_ERA5["aridity_netrad"], df_ERA5["e"]/df_ERA5["tp"], "tab:green", n=11, label='ET ERA5', statistic=stat)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], "#a6cee3", n=11, label='Q Caravan', statistic=stat)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["BFI"]*df_Caravan["TotalRR"], "#1f78b4", n=11, label='Qb Caravan', statistic=stat)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], 1-(1-df_Caravan["BFI"])*df_Caravan["TotalRR"], "#93c47d", n=11, label='P-Qf Caravan', statistic=stat)
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], df_CAMELS["runoff_ratio"], "#a6cee3", n=11, label='Q CAMELS US', statistic=stat)
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], df_CAMELS["BFI"]*df_CAMELS["runoff_ratio"], "#1f78b4", n=11, label='Qb CAMELS US', statistic=stat)
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], 1-(1-df_CAMELS["BFI"])*df_CAMELS["runoff_ratio"], "#93c47d", n=11, label='P-Qf CAMELS US', statistic=stat)
plotting_fcts.plot_lines_group(df_Lee["aridity_netrad"], df_Lee["recharge_ratio"], "#977173", n=11, label='Lee', statistic=stat)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#a86487", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#704b5b", n=6, label='MacDonald', statistic=stat)
#plotting_fcts.plot_lines_group(df_Cuthbert["PET"]/df_Cuthbert["P"], df_Cuthbert["D(=P-AET)"]/df_Cuthbert["P"], "#A496CF", n=11, label='D/P Cuthbert', statistic=stat)
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "-", c="black", alpha=0.75, label="Budyko Q")
im = axes.plot(np.linspace(0.1,10,1000), Berghuijs_recharge_curve(np.linspace(0.1,10,1000)), "-", c="grey", alpha=0.75, label='Berghuijs')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_all_fluxes_lines_only.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Lee["aridity_netrad"], df_Lee["recharge_ratio"], s=2.5, c="#977173", alpha=0.25, lw=0)
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=2.5, c="#a86487", alpha=0.25, lw=0)
im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=2.5, c="#704b5b", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_Lee["aridity_netrad"], df_Lee["recharge_ratio"], "#977173", n=11, label='Lee', statistic=stat)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#a86487", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#704b5b", n=6, label='MacDonald', statistic=stat)
plotting_fcts.plot_lines_group(df_Cuthbert["PET"]/df_Cuthbert["P"], df_Cuthbert["D(=P-AET)"]/df_Cuthbert["P"], "#A496CF", n=11, label='D/P Cuthbert', statistic=stat)
im = axes.plot(np.linspace(0.1,10,1000), Berghuijs_recharge_curve(np.linspace(0.1,10,1000)), "-", c="grey", alpha=0.75, label='Berghuijs')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko standard")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
#im = axes.scatter(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], s=2.5, c="#a6cee3", alpha=0.25, lw=0)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], "#a6cee3", n=11, label='Q Caravan', statistic=stat)
im = axes.scatter(df_Caravan2["aridity_netrad"], df_Caravan2["TotalRR"], s=2.5, c="tab:blue", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_Caravan2["aridity_netrad"], df_Caravan2["TotalRR"], "tab:blue", n=11, label='Q Caravan', statistic=stat)
im = axes.scatter(df_CAMELS["aridity"], df_CAMELS["runoff_ratio"], s=2.5, c="tab:purple", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], df_CAMELS["runoff_ratio"], "tab:purple", n=11, label='Q CAMELS', statistic=stat)
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "-", c="black", alpha=0.75, label="Budyko Q")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_standard.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge with shaded areas")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1,10,1000), 0*np.linspace(0.1,10,1000), 1-Budyko_curve(np.linspace(0.1,10,1000)), color="tab:blue", alpha=0.1)
axes.fill_between(np.linspace(0.1,10,1000),1-Budyko_curve(np.linspace(0.1,10,1000)), 1+0*np.linspace(0.1,10,1000), color="tab:green", alpha=0.1)
im = axes.plot(np.linspace(0.1,10,1000), Berghuijs_recharge_curve(np.linspace(0.1,10,1000)), "-", c="grey", alpha=0.75, label='Berghuijs')
#plotting_fcts.plot_lines_group(df_ERA5["aridity_netrad"], df_ERA5["e"]/df_ERA5["tp"], "#CEA97C", n=11, label='ET ERA5', statistic=stat)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], "#a6cee3", n=11, label='Qtot', statistic=stat)
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], df_CAMELS["BFI"]*df_CAMELS["runoff_ratio"], "#1f78b4", n=11, label='Qb CAMELS US', statistic=stat)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#a86487", n=11, label='Moeck', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#704b5b", n=6, label='MacDonald', statistic=stat)
plotting_fcts.plot_lines_group(df_Lee["aridity_netrad"], df_Lee["recharge_ratio"], "#977173", n=11, label='Lee', statistic=stat)
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_shaded_areas.png", dpi=600, bbox_inches='tight')
plt.close()

print("Budyko recharge with shaded areas and uncertainty range")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
axes.fill_between(np.linspace(0.1,10,1000), 0*np.linspace(0.1,10,1000), 1-Budyko_curve(np.linspace(0.1,10,1000)), color="tab:blue", alpha=0.1)
axes.fill_between(np.linspace(0.1,10,1000),1-Budyko_curve(np.linspace(0.1,10,1000)), 1+0*np.linspace(0.1,10,1000), color="tab:green", alpha=0.1)
im = axes.plot(np.linspace(0.1,10,1000), Berghuijs_recharge_curve(np.linspace(0.1,10,1000)), "-", c="grey", alpha=0.75, label='Berghuijs')
#plotting_fcts.plot_lines_group(df_ERA5["aridity_netrad"], df_ERA5["e"]/df_ERA5["tp"], "#CEA97C", n=11, label='ET ERA5', statistic=stat)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], df_Caravan["TotalRR"], "#a6cee3", n=11, label='Qtot', statistic=stat)
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], df_CAMELS["BFI"]*df_CAMELS["runoff_ratio"], "#1f78b4", n=11, label='Qb CAMELS US', statistic=stat, uncertainty=True)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#a86487", n=11, label='Moeck', statistic=stat, uncertainty=True)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#704b5b", n=6, label='MacDonald', statistic=stat, uncertainty=True)
plotting_fcts.plot_lines_group(df_Lee["aridity_netrad"], df_Lee["recharge_ratio"], "#977173", n=6, label='Lee', statistic=stat, uncertainty=True)
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_shaded_areas_uncertainty.png", dpi=600, bbox_inches='tight')
plt.close()


print("Finished plotting data.")
