import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts
import geopandas as gpd
import xarray as xr
import rasterio as rio

# This script loads and analyses different datasets in Budyko space.

def Budyko_curve(aridity, **kwargs):
    # Budyko, M.I., Miller, D.H. and Miller, D.H., 1974. Climate and life (Vol. 508). New York: Academic press.
    return np.sqrt(aridity * np.tanh(1 / aridity) * (1 - np.exp(-aridity)));

def Berghuijs_recharge_curve(aridity):
    alpha = 0.72
    beta = 15.11
    RR = alpha*(1-(np.log(aridity**beta+1)/(1+np.log(aridity**beta+1))))
    return RR

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# todo: plot Miguez Macho in absolute terms and for different continents

# load data

# todo: load netrad data
# todo: load caravan again and use online signatures
# todo:only use certain catchments, e.g. no human influences, long enough complete time series, etc.
# Caravan
df = pd.read_csv("./results/caravan_processed.csv")
df_Caravan = df
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

"""
# FLUXCOM
data_path = "D:/Data/FLUXCOM/RS/ensemble/720_360/monthly/"
name_list = ["H", "LE", "Rn"]
var_list = ["H.RS.EBC-ALL.MLM-ALL.METEO-NONE.720_360.monthly.",
            "LE.RS.EBC-ALL.MLM-ALL.METEO-NONE.720_360.monthly.",
            "Rn.RS.EBC-NONE.MLM-ALL.METEO-NONE.720_360.monthly."]
years = ["2001", "2002", "2003", "2004", "2005",
         "2006", "2007", "2008", "2009", "2010",
         "2011", "2012", "2013", "2014", "2015"]

# get multi annual averages
def re(path,name):
    data = xr.open_dataset(path)
    d = weighted_temporal_mean(data,name)
    d.name = name
    return d

df_tot = pd.DataFrame(columns = ["lat", "lon"])
for name, var in zip(name_list, var_list):

    # get annual averages
    data = []
    for y in years:
        path = data_path + var + y + ".nc"
        data.append(re(path,name))

    # get average of all years
    data_all_years = xr.concat(data,"time")
    data_avg = data_all_years.mean("time")

    # transform into dataframe
    df = data_avg.to_dataframe().reset_index()
    df[name] = df[name] * (10**6/86400)*12.87 # MJ m^-2 d^-1 into W m^-2 into mm/y
    df_tot = pd.merge(df_tot, df, on=['lat', 'lon'], how='outer')

data_path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"
df_domains = pd.read_csv(data_path + "model_outputs/2b/aggregated/domains.csv", sep=',')
df = pd.merge(df_tot, df_domains, on=['lat', 'lon'], how='outer')
df = df.dropna()
df_FLUXCOM = df
print("Finished FLUXCOM.")
"""

# Cuthbert
data_path = "C:/Users/gnann/Documents/PYTHON/Recharge/results/"
df = pd.read_csv(data_path + "green-roofs_deep_drainage.csv")
df_Cuthbert = df
print("Finished Cuthbert.")

# plot standard Budyko plot
print("Several fluxes Budyko")
stat = "median"
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
#im = axes.scatter(df_FLUXCOM["aridity_netrad_gswp3"], 1-df_FLUXCOM["LE"]/df_FLUXCOM["pr_gswp3"], s=1, c="#F7C188", alpha=0.25, lw=0)
#im = axes.scatter(df_Caravan["aridity"], (df_Caravan["Q_mean"]/df_Caravan["p_mean"]), s=2.5, c="#a6cee3", alpha=0.25, lw=0)
#im = axes.scatter(df_Caravan["aridity"], ((df_Caravan["BFI"]*df_Caravan["Q_mean"])/df_Caravan["p_mean"]), s=2.5, c="#1f78b4", alpha=0.25, lw=0)
#im = axes.scatter(df_CAMELS["aridity"], (1-(1-df_Caravan["BFI"])*(df_Caravan["Q_mean"]/df_Caravan["p_mean"])), s=2.5, c="#947351", alpha=0.25, lw=0)
#im = axes.scatter(df_Moeck["aridity_hpet"], (df_Moeck["Groundwater recharge [mm/y]"]/df_Moeck["tp"]), s=2.5, c="#b2df8a", alpha=0.25, lw=0)
#im = axes.scatter(df_MacDonald["aridity_hpet"], (df_MacDonald["Recharge_mmpa"]/df_MacDonald["tp"]), s=2.5, c="#33a02c", alpha=0.25, lw=0)
#plotting_fcts.plot_lines_group(df_FLUXCOM["aridity_netrad_gswp3"], df_FLUXCOM["LE"]/df_FLUXCOM["pr_gswp3"], "#F7C188", n=11, label='ET Fluxcom')
plotting_fcts.plot_lines_group(df_ERA5["aridity_netrad"], df_ERA5["e"]/df_ERA5["tp"], "#CEA97C", n=11, label='ET ERA5', statistic=stat)
#plotting_fcts.plot_lines_group(df_ERA5["aridity_hpet"], df_ERA5["e"]/df_ERA5["tp"], "#CEA97C", n=11, label='ET ERA5', statistic=stat)
plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], (df_Caravan["runoff_ratio"]), "#a6cee3", n=11, label='Qtot', statistic=stat)
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], ((df_Caravan["BFI"]*df_Caravan["Q_mean"])/df_Caravan["p_mean"]), "#1f78b4", n=11, label='Qb')
#plotting_fcts.plot_lines_group(df_Caravan["aridity_netrad"], (1-(1-df_Caravan["BFI"])*(df_Caravan["Q_mean"]/df_Caravan["p_mean"])), "#947351", n=11, label='P-Qf')
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#b2df8a", n=11, label='GWR1', statistic=stat)
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#33a02c", n=6, label='GWR2', statistic=stat)
#plotting_fcts.plot_lines_group(df_Cuthbert["PET"]/df_Cuthbert["P"], df_Cuthbert["D(=P-AET)"]/df_Cuthbert["P"], "#A496CF", n=11, label='D/P')
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



# plot standard Budyko plot
print("Recharge Budyko")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=2.5, c="#b2df8a", alpha=0.25, lw=0)
im = axes.scatter(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], s=2.5, c="#33a02c", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], "#b2df8a", n=11, label='GWR1')
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad"], df_MacDonald["recharge_ratio"], "#33a02c", n=6, label='GWR2')
plotting_fcts.plot_lines_group(df_Cuthbert["PET"]/df_Cuthbert["P"], df_Cuthbert["D(=P-AET)"]/df_Cuthbert["P"], "#A496CF", n=11, label='D/P')
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
