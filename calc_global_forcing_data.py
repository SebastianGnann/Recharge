from osgeo import gdal
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import xarray as xr
import rasterio

results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

file_path = "D:/Data/ERA5/data/"
ds_pr = xr.open_dataset(file_path + "data.nc")
#ds_pr = ds_pr.resample(time="1Y").sum(min_count=1)
ds_pr = ds_pr.mean(dim='time')
#ds_pr = ds_pr.assign_coords(longitude=(((ds_pr.longitude + 180) % 360) - 180))
#ds_pr["longitude"] = np.round(ds_pr["longitude"], 1)
#ds_pr["latitude"] = np.round(ds_pr["latitude"], 1)
ds_pr["tp"] = ds_pr["tp"]*1000*365 # convert to mm/yr by multiplying by 1000

df_pr = ds_pr.to_dataframe().reset_index().dropna()
ds_pr.to_netcdf(results_path + "ERA5_pr.nc4")
df_pr.to_csv(results_path + "ERA5_pr.csv", index=False)
#todo: currently until 2021, change to 2020

print("finished pr")

file_path = "D:/Data/hPET/"
# get long-term average forcing fields and align them
ds_list = []
for file_id in range(1981,2021):
    print(str(file_id)+"_daily_pet")
    ds_tmp = xr.open_dataset(file_path + "1981_daily_pet.nc")
    ds_tmp = ds_tmp.sum(dim='time')
    ds_list.append(ds_tmp)

ds_pet = xr.concat(ds_list, dim='year')
ds_pet = ds_pet.mean(dim='year')
ds_pet = ds_pet.where(np.isfinite(ds_pet), np.nan)
df_pet = ds_pet.to_dataframe().reset_index().dropna()
ds_pet.to_netcdf(results_path + "hPET.nc4")
df_pet.to_csv(results_path + "hPET.csv", index=False)

print("finished pet")

# calculate aridity
#df_aridity

df_pr = pd.read_csv(results_path + "ERA5_pr.csv")
df_pet = pd.read_csv(results_path + "hPET.csv")
df_pr["longitude"] = np.round(df_pr["longitude"], 1)
df_pr["latitude"] = np.round(df_pr["latitude"], 1)
df_pet["longitude"] = np.round(df_pet["longitude"], 1)
df_pet["latitude"] = np.round(df_pet["latitude"], 1)

df_combined = pd.merge(df_pr, df_pet, on=['latitude', 'longitude'], how='outer')
df_combined["aridity"] = df_combined["pet"]/df_combined["tp"]
df_combined = df_combined.dropna()
df_combined.to_csv(results_path + "aridity.csv", index=False)

from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as ml_colors
import cartopy.crs as ccrs
import shapely.geometry as sgeom
from brewer2mpl import brewer2mpl
import os
from functools import reduce
import random
import matplotlib.colors
from pingouin import partial_corr
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter, LatitudeLocator)

# prepare colour map
o = brewer2mpl.get_map("RdBu", "Diverging", 9, reverse=True)
c = o.mpl_colormap

# create figure
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()

customnorm = ml_colors.BoundaryNorm(boundaries=np.linspace(0,2,11), ncolors=256)
sc = ax.scatter(df_combined["longitude"], df_combined["latitude"], c=df_combined["aridity"], cmap=c, marker='s', s=.35, edgecolors='none',
                norm=customnorm, transform=ccrs.PlateCarree())
# ax.coastlines(linewidth=0.5)

box = sgeom.box(minx=180, maxx=-180, miny=90, maxy=-60)
x0, y0, x1, y1 = box.bounds
ax.set_extent([x0, x1, y0, y1], ccrs.PlateCarree())

cbar = plt.colorbar(sc, orientation='horizontal', pad=0.01, shrink=.5, extend='max')
cbar.set_label("Aridity")
# cbar.set_ticks([-100,-50,-10,-1,0,1,10,50,100])
cbar.ax.tick_params(labelsize=12)
plt.gca().outline_patch.set_visible(False)

gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='grey', alpha=0.75, linestyle='-')
gl.xlocator = mticker.FixedLocator([-120, -60, 0, 60, 120])
gl.ylocator = mticker.FixedLocator([-60, -30, 0, 30, 60])

plt.show()


"""
# align rasters
data_path = "/home/hydrosys/data/" #data_path = r"D:/Data/"
results_path = "/home/hydrosys/data/resampling/"

if not os.path.isdir(results_path):
    os.makedirs(results_path)

bounds = [-180, -90, 180, 90]

# 5 minute resolution
res = 5/60

path_list = ["CHELSA/CHELSA_bio12_1981-2010_V.2.1.tif"]
name_list = ["P_CHELSA"]

for path, name in zip(path_list, name_list):
    print(name)
    ds_path = data_path + path
    ds = gdal.Open(ds_path)
    dsRes = gdal.Warp(results_path + name + "_5min.tif", ds,
                      outputBounds=bounds, xRes=res, yRes=res, resampleAlg="med", dstSRS="EPSG:4326")
"""
