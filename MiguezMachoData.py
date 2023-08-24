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
df = ds = xr.open_dataset(data_path + "AUSTRALIA_sources_annualmean.nc")
df = ds.to_dataframe()
print("Finished loading data.")


