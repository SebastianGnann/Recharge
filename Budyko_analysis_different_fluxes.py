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

# Caravan
data_path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"
df = pd.read_csv(data_path + "data/complete_table.csv")
df["aridity"] = 1/(df["ari_ix_sav"]/100)
df_Caravan = df
print("Finished Caravan.")

# CAMELS
data_path = "C:/Users/gnann/Documents/MATLAB/CAMELS_Matlab/"
df = pd.read_csv(data_path + "CAMELS_table.csv")
df_CAMELS = df
print("Finished CAMELS.")

# Moeck
data_path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"
df = pd.read_csv(data_path + "data/global_groundwater_recharge_moeck-et-al.csv", sep=',')

gdf = gpd.GeoDataFrame(df)
geometry = [Point(xy) for xy in zip(df.Longitude, df.Latitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

df_domains = pd.read_csv(data_path + "model_outputs/2b/aggregated/domains.csv", sep=',')
geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
closest = closest.rename(columns={'geometry': 'closest_geom'})
df = gdf.join(closest) # merge the datasets by index (for this, it is good to use '.join()' -function)

df_Moeck = df
print("Finished Moeck.")

# MacDonald
data_path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"
df = pd.read_csv(data_path + "data/Recharge_data_Africa_BGS.csv", sep=';')

gdf = gpd.GeoDataFrame(df)
geometry = [Point(xy) for xy in zip(df.Long, df.Lat)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

df_domains = pd.read_csv(data_path + "model_outputs/2b/aggregated/domains.csv", sep=',')
geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
closest = closest.rename(columns={'geometry': 'closest_geom'})
df = gdf.join(closest) # merge the datasets by index (for this, it is good to use '.join()' -function)

df_MacDonald = df
print("Finished MacDonald.")

# FLUXNET
data_path = "C:/Users/gnann/Documents/PYTHON/GHM_Comparison/"
df_tmp = pd.read_csv(data_path + "data/FLUXNET_SITE_ID_REDUCED-YY.csv", sep=',')
df_tmp = df_tmp.replace([-9999.0],np.nan)

# average over years
df = df_tmp.groupby("SITE_ID").mean()
df = df.reset_index()

df["LATENT HEAT FLUX"] = df["LATENT HEAT FLUX"]*12.87 # transform latent heat flux into ET using latent heat of vaporisation
df["NET RADIATION"] = df["NET RADIATION"]*12.87
df["Aridity"] = df["NET RADIATION"]/df["PRECIPITATION"]

gdf = gpd.GeoDataFrame(df)
geometry = [Point(xy) for xy in zip(df.LONGITUDE, df.LATITUDE)]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

df_domains = pd.read_csv(data_path + "model_outputs/2b/aggregated/domains.csv", sep=',')
geometry = [Point(xy) for xy in zip(df_domains.lon, df_domains.lat)]
gdf_domains = gpd.GeoDataFrame(df_domains, geometry=geometry)

closest = get_nearest_neighbour.nearest_neighbor(gdf, gdf_domains, return_dist=True)
closest = closest.rename(columns={'geometry': 'closest_geom'})
df = gdf.join(closest) # merge the datasets by index (for this, it is good to use '.join()' -function)

df_FLUXNET = df
print("Finished FLUXNET.")

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

def Budyko_curve(aridity, **kwargs):
    # Budyko, M.I., Miller, D.H. and Miller, D.H., 1974. Climate and life (Vol. 508). New York: Academic press.
    return np.sqrt(aridity * np.tanh(1 / aridity) * (1 - np.exp(-aridity)));

def Berghuijs_recharge_curve(aridity):
    alpha = 0.72
    beta = 15.11
    RR = alpha*(1-(np.log(aridity**beta+1)/(1+np.log(aridity**beta+1))))
    return RR

# plot standard Budyko plot
print("Several fluxes Budyko")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
#im = axes.scatter(df_FLUXCOM["aridity_netrad_gswp3"], 1-df_FLUXCOM["LE"]/df_FLUXCOM["pr_gswp3"], s=1, c="#fb9a99", alpha=0.25, lw=0)
#im = axes.scatter(df_Caravan["aridity"], (df_Caravan["Q_mean"]/df_Caravan["p_mean"]), s=2.5, c="#a6cee3", alpha=0.25, lw=0)
#im = axes.scatter(df_Caravan["aridity"], ((df_Caravan["BFI"]*df_Caravan["Q_mean"])/df_Caravan["p_mean"]), s=2.5, c="#1f78b4", alpha=0.25, lw=0)
#im = axes.scatter(df_CAMELS["aridity"], df_CAMELS["runoff_ratio"], s=2.5, c="#a6cee3", alpha=0.25, lw=0)
#im = axes.scatter(df_CAMELS["aridity"], (df_CAMELS["BFI"]*df_CAMELS["runoff_ratio"]), s=2.5, c="#1f78b4", alpha=0.25, lw=0)
#im = axes.scatter(df_CAMELS["aridity"], (1-(1-df_CAMELS["BFI"])*df_CAMELS["runoff_ratio"]), s=2.5, c="#a35a4e", alpha=0.25, lw=0)
#im = axes.scatter(df_Moeck["aridity_netrad_gswp3"], (df_Moeck["Groundwater recharge [mm/y]"]/df_Moeck["pr_gswp3"]), s=2.5, c="#b2df8a", alpha=0.25, lw=0)
#im = axes.scatter(df_MacDonald["aridity_netrad_gswp3"], (df_MacDonald["Recharge_mmpa"]/df_MacDonald["pr_gswp3"]), s=2.5, c="#33a02c", alpha=0.25, lw=0)
#im = axes.scatter(df_FLUXNET["Aridity"], 1-df_FLUXNET["LATENT HEAT FLUX"]/df_FLUXNET["PRECIPITATION"], s=2.5, c="#f85656", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_FLUXCOM["aridity_netrad_gswp3"], df_FLUXCOM["LE"]/df_FLUXCOM["pr_gswp3"], "#fb9a99", n=11, label='ET')
#plotting_fcts.plot_lines_group(df_Caravan["aridity"], (df_Caravan["Q_mean"]/df_Caravan["p_mean"]), "#a6cee3", n=11, label='Qtot')
#plotting_fcts.plot_lines_group(df_Caravan["aridity"], ((df_Caravan["BFI"]*df_Caravan["Q_mean"])/df_Caravan["p_mean"]), "#1f78b4", n=11, label='Qb')
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], df_CAMELS["runoff_ratio"], "#a6cee3", n=11, label='Qtot')
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], (df_CAMELS["BFI"]*df_CAMELS["runoff_ratio"]), "#1f78b4", n=11, label='Qb')
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], (1-(1-df_CAMELS["BFI"])*df_CAMELS["runoff_ratio"]), "#a35a4e", n=11, label='P-Qf')
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad_gswp3"], df_Moeck["Groundwater recharge [mm/y]"]/df_Moeck["pr_gswp3"], "#b2df8a", n=11, label='GWR1')
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad_gswp3"], df_MacDonald["Recharge_mmpa"]/df_MacDonald["pr_gswp3"], "#33a02c", n=6, label='GWR2')
#plotting_fcts.plot_lines_group(df_FLUXNET["Aridity"], 1-df_FLUXNET["LATENT HEAT FLUX"]/df_FLUXNET["PRECIPITATION"], "#f85656", n=6, label='1-ET2')
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), 1-Budyko_curve(np.linspace(0.1,10,100)), "--", c="#a6cee3", alpha=0.75, label="Budyko")
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
#axes.set_xlim([0, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge_etc.png", dpi=600, bbox_inches='tight')
plt.close()

# plot standard Budyko plot
print("Recharge Budyko")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
# ticks
im = axes.scatter(df_Moeck["aridity_netrad_gswp3"], (df_Moeck["Groundwater recharge [mm/y]"]/df_Moeck["pr_gswp3"]), s=2.5, c="#b2df8a", alpha=0.25, lw=0)
im = axes.scatter(df_MacDonald["aridity_netrad_gswp3"], (df_MacDonald["Recharge_mmpa"]/df_MacDonald["pr_gswp3"]), s=2.5, c="#33a02c", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_Moeck["aridity_netrad_gswp3"], df_Moeck["Groundwater recharge [mm/y]"]/df_Moeck["pr_gswp3"], "#b2df8a", n=11, label='GWR1')
plotting_fcts.plot_lines_group(df_MacDonald["aridity_netrad_gswp3"], df_MacDonald["Recharge_mmpa"]/df_MacDonald["pr_gswp3"], "#33a02c", n=6, label='GWR2')
m = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="black", alpha=0.75)
im = axes.plot(np.linspace(0.1,10,100), Berghuijs_recharge_curve(np.linspace(0.1,10,100)), "--", c="#b2df8a", alpha=0.75, label="Berghuijs")
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.2, 5])
#axes.set_xlim([0, 5])
axes.set_ylim([-0.1, 1.1])
axes.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
axes.set_xscale('log')
plotting_fcts.plot_grid(axes)
fig.savefig(figures_path + "Budyko_recharge.png", dpi=600, bbox_inches='tight')
plt.close()


# plot P-R plot
print("Precipitation Recharge")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Moeck["pr_gswp3"], df_Moeck["Groundwater recharge [mm/y]"], s=2.5, c="#b2df8a", alpha=0.25, lw=0)
plotting_fcts.plot_lines_group(df_Moeck["pr_gswp3"], df_Moeck["Groundwater recharge [mm/y]"], "#b2df8a", n=11, label='GWR1')
axes.set_xlabel("P [mm/y]")
axes.set_ylabel("R [mm/y]")
#axes.set_xlim([0.25, 10])
axes.set_xlim([-100, 3000])
axes.set_ylim([-100, 1500])
axes.legend(loc='center right', bbox_to_anchor=(1.35, 0.5))
axes.grid()
fig.savefig(figures_path + "precipitation_recharge.png", dpi=600, bbox_inches='tight')
plt.close()


# plot standard Budyko plot
print("Baseflow Budyko")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_CAMELS["aridity"], (df_CAMELS["BFI"]*df_CAMELS["runoff_ratio"]), s=2.5, c="#1f78b4", alpha=0.5, lw=0)
plotting_fcts.plot_lines_group(df_Caravan["aridity"], ((df_Caravan["BFI"]*df_Caravan["Q_mean"])/df_Caravan["p_mean"]), "#1f78b4", n=11, label='Qb')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.25, 10])
#axes.set_xlim([0, 5])
axes.set_ylim([-0.1, 1.2])
axes.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
axes.set_xscale('log')
axes.grid()
fig.savefig(figures_path + "Budyko_baseflow.png", dpi=600, bbox_inches='tight')
plt.close()


# plot standard Budyko plot
print("Baseflow Budyko")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Caravan["aridity"], (df_Caravan["Q_mean"]/df_Caravan["p_mean"]), s=2.5, c="tab:purple", alpha=0.5, lw=0)
plotting_fcts.plot_lines_group(df_Caravan["aridity"], (df_Caravan["Q_mean"]/df_Caravan["p_mean"]), "tab:purple", n=11, label='Q Caravan')
im = axes.scatter(df_CAMELS["aridity"], df_CAMELS["runoff_ratio"], s=2.5, c="tab:blue", alpha=0.5, lw=0)
plotting_fcts.plot_lines_group(df_CAMELS["aridity"], df_CAMELS["runoff_ratio"], "tab:blue", n=11, label='Q CAMELS')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("Flux / P [-]")
axes.set_xlim([0.25, 10])
axes.set_xlim([0, 5])
axes.set_ylim([-0.1, 1.2])
axes.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
axes.grid()
axes.set_xscale('log')
fig.savefig(figures_path + "Budyko_comparison.png", dpi=600, bbox_inches='tight')
plt.close()


#plotting_fcts.plot_Budyko_limits(df_Caravan["aridity_netrad"], df_Caravan["streamflow"])
#axes.plot(np.linspace(0, 1, 100), np.linspace(1, 0, 100), '--', c='gray')
#axes.plot(np.linspace(0, 100, 100), np.linspace(1, 1, 100), '--', c='gray')
#axes.plot(np.linspace(1, 100, 100), np.linspace(0, 0, 100), '--', c='gray')
#axes.grid()

# plot data
df_Fan = pd.read_csv('C:/Users/gnann/Documents/PYTHON/Topography/data/' + 'wtd_data.csv')
df_Fan = df_Fan.dropna()
df_Fan["dummy"] = ""

# slope and wtd
print("WTD Budyko")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
axes = plt.axes()
im = axes.scatter(df_Fan["aridity_30s"], df_Fan["wtd"], s=2, c="silver", alpha=0.01, lw=0)
plotting_fcts.plot_lines_group(df_Fan["aridity_30s"], df_Fan["wtd"], "tab:blue", n=11, label='Fan')
axes.set_xlabel("PET / P [-]")
axes.set_ylabel("WTD [m]")
axes.set_xlim([0.25, 5])
axes.set_ylim([0, 15])
axes.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
axes.grid()
axes.set_xscale('log')
#axes.set_yscale('log')
plotting_fcts.plot_grid_alt(axes)
fig.savefig(figures_path + "WTD_aridity.png", dpi=600, bbox_inches='tight')
plt.close()

import seaborn as sns
# slope and wtd
x_name = "aridity_30s"
y_name = "wtd"
sns.set(rc={'figure.figsize': (4, 4)})
sns.set_style("ticks")
g = sns.FacetGrid(df_Fan, col="dummy", col_wrap=4)
g.map_dataframe(plt.scatter, x_name, y_name, color="silver", marker='o', lw=0, alpha=0.01, s=1, label=None)
g.set(xlim=[0.25, 5], ylim=[1, 100])
g.map_dataframe(plotting_fcts.plot_bins_group, x_name, y_name, color="tab:blue", group_type="dummy", group="")
#g.add_legend(loc=(.2, .75), handletextpad=0.0)
g.set(xlabel = "Aridity [-]", ylabel = "Water Table Depth [m]")
g.set_titles(col_template='{col_name}')
g.set(xscale='log', yscale='log')
plt.savefig(figures_path + x_name + '_' + y_name + ".png", dpi=600, bbox_inches='tight')
plt.close()
