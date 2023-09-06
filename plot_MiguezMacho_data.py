import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts
import geopandas as gpd
from shapely.geometry import Point
from functions.weighted_mean import weighted_temporal_mean
import xarray as xr

# This script loads and analyses Caravan data.

# check if folders exist
results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# load data

# load data
data_path = "D:/Data/MiguezMacho2021/"
path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"

file_paths = [data_path + "EURASIA_sources_annualmean.nc",
              data_path + "AFRICA_sources_annualmean.nc",
              data_path + "AUSTRALIA_sources_annualmean.nc",
              data_path + "NAMERICA_sources_annualmean.nc",
              data_path + "SAMERICA_sources_annualmean.nc"]

name = 'ALL'
#file_paths = [data_path + "AUSTRALIA_sources_annualmean.nc"]

df_list = []
for f in file_paths:
    ds_tmp = xr.open_dataset(f)

    df_tmp = ds_tmp.to_dataframe()
    df_tmp = df_tmp.loc[df_tmp["mask"] != 0]
    df_tmp = df_tmp.sample(n=int(len(df_tmp)/1000)).reset_index()

    df_list.append(df_tmp)

df = pd.concat(df_list)

print("Finished loading data.")

gdf = gpd.GeoDataFrame(df)
geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

df_domains = pd.read_csv(path + "model_outputs/2b/aggregated/domains.csv", sep=',')
geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
closest = closest.rename(columns={'geometry': 'closest_geom'})
df = gdf.join(closest, rsuffix='domains') # merge the datasets by index (for this, it is good to use '.join()' -function)

print("Finished MiguezMacho.")

# plot standard Budyko plot
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
# ticks
im = axes.scatter(df["aridity_netrad_gswp3"], (df["ET"]*365/df["pr_gswp3"]), s=2.5, c="grey", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], (df["ET"]*365/df["pr_gswp3"]), "grey", n=11, statistic='mean', label='ET')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE1"]*(df["ET"]*365/df["pr_gswp3"]), s=2.5, c="tab:orange", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE1"]*(df["ET"]*365/df["pr_gswp3"]), "tab:orange", n=11, statistic='mean', label='Source 1')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE2"]*(df["ET"]*365/df["pr_gswp3"]), s=2.5, c="tab:blue", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE2"]*(df["ET"]*365/df["pr_gswp3"]), "tab:blue", n=11, statistic='mean', label='Source 2')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE3"]*(df["ET"]*365/df["pr_gswp3"]), s=2.5, c="tab:purple", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE3"]*(df["ET"]*365/df["pr_gswp3"]), "tab:purple", n=11, statistic='mean', label='Source 3')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE4"]*(df["ET"]*365/df["pr_gswp3"]), s=2.5, c="tab:brown", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE4"]*(df["ET"]*365/df["pr_gswp3"]), "tab:brown", n=11, statistic='mean', label='Source 4')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 10])
#axes.set_xlim([0, 5])
axes.set_ylim([-0.1, 1.5])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "MiguezMacho_" + name + ".png", dpi=600, bbox_inches='tight')
plt.close()


# plot ET as function of aridity
fig = plt.figure(figsize=(7, 4), constrained_layout=True)
axes = plt.axes()
# ticks
im = axes.scatter(df["aridity_netrad_gswp3"], (df["ET"]*365), s=2.5, c="grey", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], (df["ET"]*365), "grey", n=11, statistic='mean', label='ET')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE1"]*(df["ET"]*365), s=2.5, c="tab:orange", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE1"]*(df["ET"]*365), "tab:orange", n=11, statistic='mean', label='Source 1')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE2"]*(df["ET"]*365), s=2.5, c="tab:blue", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE2"]*(df["ET"]*365), "tab:blue", n=11, statistic='mean', label='Source 2')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE3"]*(df["ET"]*365), s=2.5, c="tab:purple", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE3"]*(df["ET"]*365), "tab:purple", n=11, statistic='mean', label='Source 3')
im = axes.scatter(df["aridity_netrad_gswp3"], df["SOURCE4"]*(df["ET"]*365), s=2.5, c="tab:brown", alpha=0.1, lw=0)
plotting_fcts.plot_lines_group(df["aridity_netrad_gswp3"], df["SOURCE4"]*(df["ET"]*365), "tab:brown", n=11, statistic='mean', label='Source 4')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux [-]")
axes.set_xlim([0.2, 10])
#axes.set_xlim([0, 5])
axes.set_ylim([-0.1, 1500])
axes.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "MiguezMacho_ET_" + name + ".png", dpi=600, bbox_inches='tight')
plt.close()
