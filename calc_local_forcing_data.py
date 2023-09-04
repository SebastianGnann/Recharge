import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from functions import get_nearest_neighbour, plotting_fcts
import geopandas as gpd
from shapely.geometry import Point
from functions.weighted_mean import weighted_temporal_mean
import xarray as xr

results_path = "results/"
if not os.path.isdir(results_path):
    os.makedirs(results_path)

data_path = "D:/Data/hPET/"

# extract values for recharge datasets


# extract values for caravan (start with CAMELS US)
