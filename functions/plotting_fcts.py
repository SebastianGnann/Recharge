import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# This script contains a collection of function that make or help with making plots.

def plot_origin_line(x, y, **kwargs):

    ax = plt.gca()
    lower_lim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    upper_lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(lower_lim, upper_lim, 1000), np.linspace(lower_lim,  upper_lim, 1000), '--', color='black', alpha=0.5, zorder=1)


def plot_Budyko_limits(x, y, **kwargs):

    ax = plt.gca()
    lim = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--', c='gray')
    ax.plot(np.linspace(1, lim, 100), np.linspace(1, 1, 100), '--', c='gray')


def plot_lines_group(x, y, color, n=11, label='', statistic='mean', uncertainty=False, linestyle='-', **kwargs):

    import matplotlib.patheffects as mpe
    outline = mpe.withStroke(linewidth=4, foreground='white', alpha=0.75)

    # extract data
    df = kwargs.get('data')

    # get correlations

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, min_stat, max_stat, \
    asymmetric_error, bin_median = get_binned_stats(x, y, n)

    ax = plt.gca()
    corr_str = ''
    r_sp, _ = stats.spearmanr(x, y, nan_policy='omit')
    #corr_str = corr_str + r' $\rho_s$ ' + str(domain) + " = " + str(np.round(r_sp,2))
    corr_str = corr_str + str(np.round(r_sp,2))
    print(corr_str)
    r_sp_tot, _ = stats.spearmanr(x, y, nan_policy='omit')
    print("(" + str(np.round(r_sp_tot,2)) + ")")

    # plot bins
    ax = plt.gca()
    #ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
    #            fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc=color, alpha=0.9, label=corr_str)
    if statistic == 'median':
        ax.plot(bin_median, median_stat.statistic, color=color, path_effects=[outline], label=label, linestyle=linestyle)
        if uncertainty == True:
            ax.fill_between(bin_median, p_25_stat.statistic, p_75_stat.statistic, facecolor=color, alpha=0.2)
    elif statistic == 'mean':
        ax.plot(bin_median, mean_stat.statistic, color=color, path_effects=[outline], label=label, linestyle=linestyle)
        if uncertainty == True:
            ax.fill_between(bin_median, mean_stat.statistic-std_stat.statistic, mean_stat.statistic+std_stat.statistic, facecolor=color, alpha=0.2)
    else:
        print('incorrect statistic - using median')
        ax.plot(bin_median, median_stat.statistic, color=color, path_effects=[outline], label=label, linestyle=linestyle)
    #ax.fill_between(bin_median, p_25_stat.statistic, p_75_stat.statistic, facecolor=color, alpha=0.1)
    #ax.fill_between(bin_median, p_05_stat.statistic, p_95_stat.statistic, facecolor=color, alpha=0.1)


def get_binned_stats(x, y, n=11):

    # calculate binned statistics
    bin_edges = stats.mstats.mquantiles(x[~np.isnan(x)], np.linspace(0, 1, n))
    #bin_edges = np.linspace(0, 2500, 11)
    bin_edges = np.unique(bin_edges)
    mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
    std_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanstd(y), bins=bin_edges)
    median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
    p_05_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .05), bins=bin_edges)
    p_25_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .25), bins=bin_edges)
    p_75_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .75), bins=bin_edges)
    p_95_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanquantile(y, .95), bins=bin_edges)
    min_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmin(y), bins=bin_edges)
    max_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmax(y), bins=bin_edges)
    asymmetric_error = [median_stat.statistic - p_25_stat.statistic, p_75_stat.statistic - median_stat.statistic]
    bin_median = stats.mstats.mquantiles(x, np.linspace(0.05, 0.95, len(bin_edges)-1))
    #bin_median = np.linspace(125, 2375, 10)

    return bin_edges, \
           mean_stat, std_stat, median_stat, \
           p_05_stat, p_25_stat, p_75_stat, p_95_stat, min_stat, max_stat, \
           asymmetric_error, bin_median


def get_binned_range(x, y, bin_edges):

    # calculate binned statistics
    min_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmin(y), bins=bin_edges)
    max_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmax(y), bins=bin_edges)
    median_stat = stats.binned_statistic(x, y, statistic=np.nanmedian, bins=bin_edges)
    mean_stat = stats.binned_statistic(x, y, statistic=lambda y: np.nanmean(y), bins=bin_edges)
    bin_median = bin_edges[1:] - (bin_edges [0]+bin_edges[1])/2

    return min_stat, max_stat, median_stat, mean_stat, bin_median


def plot_grid(axes):
    [axes.axvline(x=i, c="lightgrey", linewidth=1.0, zorder=0) for i in [0.5, 0.8, 1.25, 2.0]]
    [axes.axhline(y=i, c="lightgrey", linewidth=1.0, zorder=0) for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    axes.axvline(1.0, linestyle='--', c="grey", linewidth=1.0, zorder=0, alpha=0.5)
    axes.set_xticks([0.2, 0.5, 0.8, 1.25, 2.0, 5.0])
    axes.set_xticklabels(['0.2', '0.5', '0.8', '1.25', '2.0', '5.0'])
    axes.fill_between(np.linspace(0.1,10,10), np.ones(10), 2*np.ones(10), color='lightgrey', alpha=0.25)
    axes.fill_between(np.linspace(0.1,10,10), -1*np.ones(10), 0*np.ones(10), color='lightgrey', alpha=0.25)


def plot_grid_alt(axes):
    [axes.axvline(x=i, c="lightgrey", linewidth=1.0, zorder=0) for i in [0.5, 0.8, 1.25, 2.0]]
    [axes.axhline(y=i, c="lightgrey", linewidth=1.0, zorder=0) for i in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    #axes.axvline(1.0, linestyle='--', c="grey", linewidth=1.0, zorder=0, alpha=0.5)
    axes.set_xticks([0.2, 0.5, 0.8, 1.25, 2.0, 5.0])
    axes.set_xticklabels(['0.2', '0.5', '0.8', '1.25', '2.0', '5.0'])
    #axes.fill_between(np.linspace(0.1,10,10), np.ones(10), 2*np.ones(10), color='lightgrey', alpha=0.25)
    #axes.fill_between(np.linspace(0.1,10,10), -1*np.ones(10), 0*np.ones(10), color='lightgrey', alpha=0.25)


def plot_bins_group(x, y, color="tab:blue", group_type="aridity_class", group="energy-limited", **kwargs):

    # extract data
    df = kwargs.get('data')

    # get correlations
    #df = df.dropna()
    df_group = df.loc[df[group_type]==group]

    # calculate binned statistics
    bin_edges, \
    mean_stat, std_stat, median_stat, \
    p_05_stat, p_25_stat, p_75_stat, p_95_stat, min_stat, max_stat, \
    asymmetric_error, bin_median = get_binned_stats(df_group[x], df_group[y])

    ax = plt.gca()
    corr_str = ''
    r_sp, _ = stats.spearmanr(df.loc[df[group_type] == group, x], df.loc[df[group_type] == group, y], nan_policy='omit')
    #corr_str = corr_str + r' $\rho_s$ ' + str(group) + " = " + str(np.round(r_sp,2))
    corr_str = corr_str + str(np.round(r_sp,2))
    print(corr_str)
    r_sp_tot, _ = stats.spearmanr(df[x], df[y], nan_policy='omit')
    print("(" + str(np.round(r_sp_tot,2)) + ")")

    # plot bins
    ax = plt.gca()
    ax.errorbar(bin_median, median_stat.statistic, xerr=None, yerr=asymmetric_error, capsize=2,
                fmt='o', ms=4, elinewidth=1, c='black', ecolor='black', mec='black', mfc=color, alpha=0.9, label=corr_str)
