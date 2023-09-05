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

results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

def aggregate_era5(data_path, var):
    file_path = data_path + var + "/data.nc"
    ds = xr.open_dataset(file_path)
    ds = ds.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))
    #ds = ds.resample(time="1Y").sum(min_count=1)
    ds = ds.mean(dim='time')
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)) # shift coordinates
    ds = ds.sortby(ds.longitude)
    ds["longitude"] = np.round(ds["longitude"], 1)
    ds["latitude"] = np.round(ds["latitude"], 1)
    df = ds.to_dataframe().reset_index().dropna()
    ds.to_netcdf(results_path + "ERA5_" + var + ".nc4")
    df.to_csv(results_path + "ERA5_" + var + ".csv", index=False)
    del ds
    del df
    print("finished aggregation", var)

"""
file_path = "D:/Data/ERA5/p/data.nc"
ds = xr.open_dataset(file_path)
#ds = ds.sel(time=slice(dt(1975, 1, 1), dt(2004, 12, 31)))
ds = ds.mean(dim='time')
pr = ds['tp']
#(Optional) convert longitude from (0-360) to (-180 to 180) (if required)
pr.coords['longitude'] = (pr.coords['longitude'] + 180) % 360 - 180
pr = pr.sortby(pr.longitude)
#Define lat/long
pr = pr.rio.set_spatial_dims('longitude', 'latitude')
#Check for the CRS
pr.rio.crs
pr.rio.set_crs("epsg:4326")
pr.rio.to_raster(results_path + r"pr_GeoTIFF.tif")
"""

# aggregate datasets
data_path = "D:/Data/ERA5/"
var_list = ["p", "pet", "aet", "rs", "rl", "lh"]
for var in var_list:
    aggregate_era5(data_path,var)

# todo: load netcdf and mege them
# load and merge aggregated datasets
df = pd.read_csv(results_path + "hPET.csv")
for var in var_list:
    df_tmp = pd.read_csv(results_path + "ERA5_" + var + ".csv")
    df = pd.merge(df, df_tmp, on=['latitude', 'longitude'], how='outer')

df["tp"] = df["tp"]*1000*365 # convert to mm/yr
df["e"] = df["e"]*-1000*365 # convert to mm/yr
df["pev"] = df["pev"]*-1000*365 # convert to mm/yr
df["ssr"] = df["ssr"]/86400 # convert to W/m2
df["str"] = df["str"]/86400 # convert to W/m2
df["slhf"] = df["slhf"]/-86400 # convert to W/m2
df["aridity_era5"] = df["pev"]/df["tp"]
df["aridity_hpet"] = df["pet"]/df["tp"]
df["netrad"] = (df["ssr"] + df ["str"])*12.87 # convert to mm/yr
df["aridity_netrad"] = df["netrad"]/df["tp"]
df["aet_lh"] = df["slhf"]*12.87 # convert to mm/yr
#df.to_csv(results_path + "aridity.csv", index=False)

ds = xr.Dataset.from_dataframe(df)
ds = ds.set_coords(['latitude', 'longitude'])
ds.to_netcdf(results_path + "aridity.nc4")

# aggregate hPET dataset
data_path = "D:/Data/hPET/"
def aggregate_hpet(data_path):
    ds_list = []
    for file_id in range(1981,2021):
        print(str(file_id)+"_daily_pet")
        ds_tmp = xr.open_dataset(data_path + "1981_daily_pet.nc")
        ds_tmp = ds_tmp.sum(dim='time')
        ds_list.append(ds_tmp)
    ds_pet = xr.concat(ds_list, dim='year')
    ds_pet = ds_pet.mean(dim='year')
    ds_pet = ds_pet.where(np.isfinite(ds_pet), np.nan)
    ds_pet["longitude"] = np.round(ds_pet["longitude"], 1)
    ds_pet["latitude"] = np.round(ds_pet["latitude"], 1)
    df_pet = ds_pet.to_dataframe().reset_index().dropna()
    ds_pet.to_netcdf(results_path + "hPET.nc4")
    df_pet.to_csv(results_path + "hPET.csv", index=False)
    print("finished hpet")

#aggregate_hpet(data_path)

# plot results
o = brewer2mpl.get_map("RdBu", "Diverging", 9, reverse=True) # prepare colour map
c = o.mpl_colormap
plt.rcParams['axes.linewidth'] = 0.1
fig = plt.figure()
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
customnorm = ml_colors.BoundaryNorm(boundaries=np.linspace(0,2,11), ncolors=256)
sc = ax.scatter(df["longitude"], df["latitude"], c=df["aridity"], cmap=c, marker='s', s=.35, edgecolors='none',
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
