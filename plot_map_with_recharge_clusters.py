import os
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as dt
import matplotlib.colors as ml_colors
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from brewer2mpl import brewer2mpl
import matplotlib.ticker as mticker

figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

# load data
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

# merge datasets and calculate categories
df_Moeck["source"] = "Moeck"
df_MacDonald["source"] = "MacDonald"
df_Recharge = pd.concat([df_Moeck[["latitude", "longitude", "tp", "netrad", "aridity_netrad", "recharge_ratio", "source"]],
                        df_MacDonald[["latitude", "longitude", "tp", "netrad", "aridity_netrad", "recharge_ratio", "source"]]])
df_Recharge["category"] = 0
df_Recharge = df_Recharge.reset_index()
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] <= 0.8,
                               df_Recharge["recharge_ratio"] <= 1.0)),
                "category"] = 1
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] > 0.8, df_Recharge["aridity_netrad"] <= 1.25,
                                         df_Recharge["recharge_ratio"] >= 0.0, df_Recharge["recharge_ratio"] < 0.2)),
                "category"] = 2
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] > 0.8, df_Recharge["aridity_netrad"] <= 1.25,
                                         df_Recharge["recharge_ratio"] > 0.2, df_Recharge["recharge_ratio"] <= 0.4)),
                "category"] = 3
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] > 0.8, df_Recharge["aridity_netrad"] <= 1.25,
                                         df_Recharge["recharge_ratio"] > 0.4, df_Recharge["recharge_ratio"] <= 1.0)),
                "category"] = 4
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] > 1.25, df_Recharge["aridity_netrad"] <= 2.0,
                                         df_Recharge["recharge_ratio"] >= 0.0, df_Recharge["recharge_ratio"] <= 0.2)),
                "category"] = 5
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] > 1.25, df_Recharge["aridity_netrad"] <= 2.0,
                                         df_Recharge["recharge_ratio"] > 0.2, df_Recharge["recharge_ratio"] <= 1.0)),
                "category"] = 6
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] > 2.0,
                                         df_Recharge["recharge_ratio"] >= 0.0, df_Recharge["recharge_ratio"] < 0.2)),
                "category"] = 7
df_Recharge.loc[np.logical_and.reduce((df_Recharge["aridity_netrad"] > 2.0,
                                         df_Recharge["recharge_ratio"] >= 0.2, df_Recharge["recharge_ratio"] <= 1.0)),
                "category"] = 8
df_Recharge.loc[df_Recharge["recharge_ratio"] > 1.0,
                "category"] = 9

df_Recharge = df_Recharge.dropna().reset_index()
# todo: a few points that fall outside the land area have nan values for tp and netrad

# plot data
o = brewer2mpl.get_map("Spectral", "Diverging", 9, reverse=True) # prepare colour map
c = o.mpl_colormap
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
customnorm = ml_colors.BoundaryNorm(boundaries=np.linspace(0.5,9.5,10), ncolors=256)
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "Moeck", "longitude"], df_Recharge.loc[df_Recharge["source"] == "Moeck", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "Moeck", "category"], cmap=c, s=2.5, marker="o", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "MacDonald", "longitude"], df_Recharge.loc[df_Recharge["source"] == "MacDonald", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "MacDonald", "category"], cmap=c, s=2.5, marker="s", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
x0, y0, x1, y1 = box.bounds
ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5)
cbar.set_label("Recharge category")
# cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
cbar.ax.tick_params(labelsize=12)
plt.gca().outline_patch.set_visible(False)
#gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
#gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
#gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
cbar.set_ticks(np.linspace(1, 9, 9))
fig.savefig(figures_path + "recharge_clusters.png", dpi=600, bbox_inches='tight')
plt.close()

o = brewer2mpl.get_map("Spectral", "Diverging", 9, reverse=True) # prepare colour map
c = o.mpl_colormap
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
customnorm = ml_colors.BoundaryNorm(boundaries=np.linspace(0.5,9.5,10), ncolors=256)
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "Moeck", "longitude"], df_Recharge.loc[df_Recharge["source"] == "Moeck", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "Moeck", "category"], cmap=c, s=2.5, marker="o", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "MacDonald", "longitude"], df_Recharge.loc[df_Recharge["source"] == "MacDonald", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "MacDonald", "category"], cmap=c, s=2.5, marker="s", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
box = sgeom.box(minx=110, maxx=150, miny=-10, maxy=-40)
x0, y0, x1, y1 = box.bounds
ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5)
cbar.set_label("Recharge category")
# cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
cbar.ax.tick_params(labelsize=12)
plt.gca().outline_patch.set_visible(False)
#gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
#gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
#gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
cbar.set_ticks(np.linspace(1, 9, 9))
fig.savefig(figures_path + "recharge_clusters_AUS.png", dpi=600, bbox_inches='tight')
plt.close()

# plot data
o = brewer2mpl.get_map("Spectral", "Diverging", 8, reverse=True) # prepare colour map
c = o.mpl_colormap
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
customnorm = ml_colors.BoundaryNorm(boundaries=[0, 0.5, 0.8, 1.25, 2.0, 100.0], ncolors=256)
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "Moeck", "longitude"], df_Recharge.loc[df_Recharge["source"] == "Moeck", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "Moeck", "aridity_netrad"], cmap=c, s=2.5, marker="o", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "MacDonald", "longitude"], df_Recharge.loc[df_Recharge["source"] == "MacDonald", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "MacDonald", "aridity_netrad"], cmap=c, s=2.5, marker="s", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
x0, y0, x1, y1 = box.bounds
ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5)
cbar.set_label("Aridity")
# cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
cbar.ax.tick_params(labelsize=12)
plt.gca().outline_patch.set_visible(False)
#gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
#gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
#gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
#cbar.set_ticks(np.linspace(1, 8, 8))
fig.savefig(figures_path + "aridity_clusters.png", dpi=600, bbox_inches='tight')
plt.close()

# plot data
o = brewer2mpl.get_map("Spectral", "Diverging", 8, reverse=True) # prepare colour map
c = o.mpl_colormap
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
customnorm = ml_colors.BoundaryNorm(boundaries=[0, 0.5, 0.8, 1.25, 2.0, 100.0], ncolors=256)
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "Moeck", "longitude"], df_Recharge.loc[df_Recharge["source"] == "Moeck", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "Moeck", "aridity_netrad"], cmap=c, s=2.5, marker="o", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
sc = ax.scatter(df_Recharge.loc[df_Recharge["source"] == "MacDonald", "longitude"], df_Recharge.loc[df_Recharge["source"] == "MacDonald", "latitude"],
                c=df_Recharge.loc[df_Recharge["source"] == "MacDonald", "aridity_netrad"], cmap=c, s=2.5, marker="s", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
box = sgeom.box(minx=110, maxx=150, miny=-10, maxy=-40)
x0, y0, x1, y1 = box.bounds
ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5)
cbar.set_label("Aridity")
# cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
cbar.ax.tick_params(labelsize=12)
plt.gca().outline_patch.set_visible(False)
#gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
#gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
#gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
#cbar.set_ticks(np.linspace(1, 8, 8))
fig.savefig(figures_path + "aridity_clusters_AUS.png", dpi=600, bbox_inches='tight')
plt.close()


# plot data
cat = 4
o = brewer2mpl.get_map("Spectral", "Diverging", 9, reverse=True) # prepare colour map
c = o.mpl_colormap
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
customnorm = ml_colors.BoundaryNorm(boundaries=np.linspace(0.5,9.5,10), ncolors=256)
sc = ax.scatter(df_Recharge["longitude"], df_Recharge["latitude"],
                c="grey", s=2.5, marker="o", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
sc = ax.scatter(df_Recharge.loc[df_Recharge["category"] == cat, "longitude"], df_Recharge.loc[df_Recharge["category"] == cat, "latitude"],
                c="tab:orange", s=2.5, marker="o", alpha=0.75, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
ax.coastlines(linewidth=0.5)
box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
x0, y0, x1, y1 = box.bounds
ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())
#cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5)
#cbar.set_label("Recharge category")
# cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
#cbar.ax.tick_params(labelsize=12)
plt.gca().outline_patch.set_visible(False)
#gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
#gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
#gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])
#cbar.set_ticks(np.linspace(1, 9, 9))
fig.savefig(figures_path + "recharge_clusters_" + str(cat) + ".png", dpi=600, bbox_inches='tight')
plt.close()
