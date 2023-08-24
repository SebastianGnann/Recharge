import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy import stats
import seaborn as sns
from functions import plotting_fcts
import geopandas as gpd
from shapely.geometry import Point
import rasterio as rio
#from pingouin import partial_corr
import rioxarray
import geopandas
import xarray as xr
from shapely.geometry import mapping
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# This script loads and analyses Caravan data.

# prepare data
data_path = "D:/Data/Caravan/"

# check if folders exist
results_path = "D:/Data/Caravan_Fluxcom/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)
figures_path = "figures/"
if not os.path.isdir(figures_path):
    os.makedirs(figures_path)

# load data
df = pd.read_csv(results_path + "caravan_processed.csv")

fluxcom_path = r"D:/Data/FLUXCOM/RS/ensemble/4320_2160/8daily/"

var_list = ["H", "LE", "Rn"]
name_list = ["H.RS.EBC-ALL.MLM-ALL.METEO-NONE.4320_2160.8daily.",
            "LE.RS.EBC-ALL.MLM-ALL.METEO-NONE.4320_2160.8daily.",
            "Rn.RS.EBC-NONE.MLM-ALL.METEO-NONE.4320_2160.8daily."]
years = ["2001", "2002", "2003", "2004", "2005",
         "2006", "2007", "2008", "2009", "2010",
         "2011", "2012", "2013", "2014", "2015"]

"""
df_tot = pd.DataFrame(columns = ["lat", "lon"])
for name, var in zip(name_list, var_list):

    # get annual averages
    data = []
    for y in years:
        path = fluxcom_path + var + y + ".nc"
        data.append(re(path,name))

    # get average of all years
    data_all_years = xr.concat(data,"time")
    data_avg = data_all_years.mean("time")

    # transform into dataframe
    df = data_avg.to_dataframe().reset_index()
    df[name] = df[name] * (10**6/86400)*12.87 # MJ m^-2 d^-1 into W m^-2 into mm/y
    df_tot = pd.merge(df_tot, df, on=['lat', 'lon'], how='outer')

df_tot = df_tot.dropna()
"""

# D:\Data\Caravan\shapefiles\camels
shp_path = "D:/Data/Caravan/shapefiles/camels/camels_basin_shapes.shp" # todo: automatic path

geodf = geopandas.read_file(shp_path, crs="epsg:4326")

# todo: think about best order of looping and how to speed up code
for i in range(400, 403): # len(df)
    print(str(i))

    for var, name in zip(var_list, name_list):
        print(var)
        print(name)

        df_fluxcom = pd.DataFrame(columns=['gauge_id','date',var])

        for year in years: #todo: merge all years
            #print(year)

            path = "D:/Data/FLUXCOM/RS/ensemble/4320_2160/8daily/" + name + year + ".nc"

            fluxcom = xr.open_dataset(path)
            fluxcom = fluxcom[var]
            fluxcom.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            fluxcom.rio.write_crs("epsg:4326", inplace=True)
            #todo: to get more precise results, resample to 10 times resolution - then perhaps swap loops again

            geodf_single = gpd.GeoDataFrame(geodf.iloc[i,:]).transpose()
            clipped = fluxcom.rio.clip(geodf_single.geometry.apply(mapping), geodf.crs, drop=False) #.geometry.apply(mapping)

            """
            clipped[0].plot.pcolormesh(vmin=0.5, vmax=2.0)
            #fluxcom[0].plot.pcolormesh(vmin=0.5, vmax=2.5)
            plt.xlim(-125,-65)
            plt.ylim(30,55)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            #plt.show()
            plt.savefig(figures_path + geodf_single.loc[i, "gauge_id"] + '_map' + ".png", dpi=600, bbox_inches='tight')
            plt.close()
            """

            clipped_mean = clipped.mean(axis=(1,2)).values
            df_tmp = pd.DataFrame(columns=['gauge_id','date',var])
            df_tmp['gauge_id'] = [geodf_single.loc[i, "gauge_id"]] * len(clipped_mean)
            df_tmp['date'] = clipped.time
            df_tmp[var] = clipped_mean

            if len(df_fluxcom) == 0:
                df_fluxcom = df_tmp
            else:
                df_fluxcom = pd.concat([df_fluxcom, df_tmp])

        fig = plt.figure(figsize=(5, 2), constrained_layout=True)
        plt.plot(df_fluxcom["date"], df_fluxcom[var])
        plt.ylabel(var)
        fig.savefig(figures_path + geodf_single.loc[i, "gauge_id"] + '_timeseries_' + var + ".png", dpi=600, bbox_inches='tight')
        plt.close()

        #print(df_fluxcom.groupby('gauge_id').mean() * 0.408 * 365) #https://www.fao.org/3/x0490e/x0490e04.htm
        #df_mean = df_fluxcom.groupby('gauge_id').mean()

        # save results
        df_fluxcom.to_csv(results_path + geodf_single.loc[i, "gauge_id"] + '_timeseries_' + var + '_fluxcom_processed.csv', index=False)
