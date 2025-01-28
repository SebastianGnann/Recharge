import scipy.io
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

# Caravan

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

stat = "median"

df_Moeck = df_Moeck.dropna()

from scipy.optimize import curve_fit
from scipy import stats
def func(x, a, b):
    return a * x ** b

df_Moeck["tp_rand"] = df_Moeck["tp"] + np.random.normal(0, 0.0001, len(df_Moeck["tp"]))

params, params_covariance = curve_fit(func, df_Moeck["tp_rand"], df_Moeck["recharge"], p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
recharge_fit = func(np.arange(0,2000,1), a, b)

x = df_Moeck["tp_rand"]
y = df_Moeck["recharge"]

bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, 20))
mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
bin_median = stats.mstats.mquantiles(x, np.linspace(0.05, 0.95, len(bin_edges)-1))
params, params_covariance = curve_fit(func, bin_median, mean_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_20_mean = func(np.arange(0,2000,1), a, b)

params, params_covariance = curve_fit(func, bin_median, median_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_20_median = func(np.arange(0,2000,1), a, b)

bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, 50))
mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
bin_median = stats.mstats.mquantiles(x, np.linspace(0.05, 0.95, len(bin_edges)-1))
params, params_covariance = curve_fit(func, bin_median, mean_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_50_mean = func(np.arange(0,2000,1), a, b)

bin_edges = np.linspace(0, 2000, 21)
mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
bin_median = np.linspace(50, 1950, 20)
params, params_covariance = curve_fit(func, bin_median, mean_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_20_mean_equal = func(np.arange(0,2000,1), a, b)

print("Recharge binning")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Moeck["tp_rand"], df_Moeck["recharge"], s=10, c="dimgrey", alpha=0.25, lw=0)
#im = axes.scatter(bin_median, mean_stat.statistic, s=25, c="tab:blue", alpha=0.75, lw=0)
plt.plot(np.arange(0,2000,1), recharge_fit, label="full data", color='dimgrey')
plt.plot(np.arange(0,2000,1), y_fit_20_mean, label="20 bins mean", color='tab:blue')
plt.plot(np.arange(0,2000,1), y_fit_20_median, label="20 bins median", color='tab:orange')
#plt.plot(np.arange(0,2000,1), y_fit_50_mean, label="50 bins mean", color='tab:red')
plt.plot(np.arange(0,2000,1), y_fit_20_mean_equal, label="20 bins equally spaced", color='tab:purple')
#plotting_fcts.plot_lines_group(df_Moeck["tp"], df_Moeck["recharge"], "dimgrey", n=11, label='R', statistic=stat)
axes.set_xlabel("P [mm/y]")
axes.set_ylabel("R [mm/y]")
axes.set_xlim([0, 2000])
axes.set_ylim([0, 2000])
axes.legend(loc='upper left')#, bbox_to_anchor=(1.4, 0.5))
#axes.set_xscale('log')
#plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "recharge_precipitation_binning.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")



params, params_covariance = curve_fit(func, df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
recharge_fit = func(np.arange(0.5,10,0.01), a, b)

x = df_Moeck["aridity_netrad"]
y = df_Moeck["recharge_ratio"]

bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, 20))
mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
bin_median = stats.mstats.mquantiles(x, np.linspace(0.05, 0.95, len(bin_edges)-1))
params, params_covariance = curve_fit(func, bin_median, mean_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_20_mean = func(np.arange(0.5,10,0.01), a, b)

params, params_covariance = curve_fit(func, bin_median, median_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_20_median = func(np.arange(0.5,10,0.01), a, b)

bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, 50))
bin_edges = np.unique(bin_edges)
mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
bin_median = stats.mstats.mquantiles(x, np.linspace(0.05, 0.95, len(bin_edges)-1))
params, params_covariance = curve_fit(func, bin_median, mean_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_50_mean = func(np.arange(0.5,10,0.01), a, b)

bin_edges = np.linspace(0.5, 5, 19)
mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
bin_median = np.linspace(0.625, 4.875, 18)
params, params_covariance = curve_fit(func, bin_median, mean_stat.statistic, p0=[1, 2], maxfev=2000)
a, b = params
print(f"Fitted parameters: a = {a}, b = {b}")
y_fit_20_mean_equal = func(np.arange(0.5,10,0.01), a, b)

print("Recharge binning")
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Moeck["aridity_netrad"], df_Moeck["recharge_ratio"], s=10, c="dimgrey", alpha=0.25, lw=0)
#im = axes.scatter(bin_median, mean_stat.statistic, s=25, c="tab:blue", alpha=0.75, lw=0)
plt.plot(np.arange(0.5,10,0.01), recharge_fit, label="full data", color='dimgrey')
plt.plot(np.arange(0.5,10,0.01), y_fit_20_mean, label="20 bins mean", color='tab:blue')
plt.plot(np.arange(0.5,10,0.01), y_fit_20_median, label="20 bins median", color='tab:orange')
#plt.plot(np.arange(0.5,10,0.01), y_fit_50_mean, label="50 bins mean", color='tab:red')
plt.plot(np.arange(0.5,10,0.01), y_fit_20_mean_equal, label="20 bins equally spaced", color='tab:purple')
#plotting_fcts.plot_lines_group(df_Moeck["tp"], df_Moeck["recharge"], "dimgrey", n=11, label='R', statistic=stat)
axes.set_xlabel("PET/P [-]")
axes.set_ylabel("R/P [-]")
axes.set_xlim([0, 5])
axes.set_ylim([0, 1.5])
axes.legend(loc='upper right')#, bbox_to_anchor=(1.4, 0.5))
#axes.set_xscale('log')
#plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "recharge_ratio_aridity_binning.png", dpi=600, bbox_inches='tight')
plt.close()

print("Finished plotting data.")
