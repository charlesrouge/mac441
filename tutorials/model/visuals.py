from .func_FDC import compute_fdc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


'''
List of plotting methods within:
    - storage_timeseries: plots 1 storage timeseries
    - compare_storage_timeseries: plots 2 or 3 storage timeseries
    - flow_timeseries: plots 1 flow timeseries
    - compare_flow_timeseries: plots 2 or 3 flows timeseries
    - monthly_averages: compute average annual cycle of given variable, monthly time step
    - compare_monthly_averages: same for same given variable in 2 or 3 different cases
    - annual_average: annual average of quantity over time
    - plot_regret: bar plots of regrets for a discrete sets of uncertain scenarios and levers
'''


def storage_timeseries(reservoir, balance, **kwargs):
    """
    Plots daily storage over time. Arguments:
        reservoir: an instance of the Reservoir class whose storage is being plotted
        balance: a Pandas DataFrame containing the time series of storage
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
    Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", balance.index[0])
    last_date = kwargs.pop('last_date', balance.index[-1])

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    s, = ax.plot(balance.index, balance['Storage (m3)'], c='b', linewidth=2, label='Storage')
    s_min, = ax.plot(balance.index, reservoir.dead_storage * np.ones(len(balance)), c='black', linestyle='--',
                     linewidth=2, label='Dead storage')
    ax.legend(handles=[s, s_min], loc=4, prop={'size': 14})
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel('Storage (m3)', size=16)
    ax.set_xlim(first_date, last_date)
    ax.tick_params(axis='both', which='major', labelsize=14)

    return fig


def compare_storage_timeseries(reservoir, storage_1, storage_2, labels, **kwargs):
    """
    Plots daily storage over time. Arguments:
        reservoir: an instance of the Reservoir class whose storage is being plotted
        storage_1: a Pandas Series containing a first time series of storage
        storage_2: a Pandas Series containing a second time series of storage
        labels: a list of Strings for legend labels, one for each of the time series above
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
        optional argument `storage_3`: a Pandas Series containing a third time series of storage
    Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", storage_1.index[0])
    last_date = kwargs.pop('last_date', storage_1.index[-1])
    # If there are only two time series to compare
    dummy_array = np.empty(5)
    dummy_array[:] = np.nan
    storage_3 = kwargs.pop('storage_3', pd.Series(dummy_array))

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(storage_1.index, storage_1, c='b', linewidth=2, label=labels[0])
    ax.plot(storage_2.index, storage_2, c='r', linewidth=2, label=labels[1])
    if storage_3.hasnans is False:  # There is a third time series
        ax.plot(storage_3.index, storage_3, c='k', linewidth=2, label=labels[2])
    ax.plot(storage_1.index, reservoir.dead_storage * np.ones(len(storage_1)), c='black', linestyle='--',
            linewidth=2, label='Dead storage')
    legend = ax.legend(loc=4, prop={'size': 14})
    ax.set_xlabel('Date', size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Storage (m3)', size=16)
    ax.set_xlim(first_date, last_date)

    return fig


def flow_timeseries(time_series, **kwargs):
    """
    Plots daily timeseries of a water balance flow component over time. Arguments:
        balance: a Pandas Series containing the time series to plot
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
    Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", time_series.index[0])
    last_date = kwargs.pop('last_date', time_series.index[-1])

    # Define Figure object
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Plot
    ax.plot(pd.date_range(start=first_date, end=last_date, freq='D'),
            time_series.loc[first_date:last_date], c='b', linewidth=2)

    # Axes specifications
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel(time_series.name, size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(first_date, last_date)
    ax.set_ylim(0, time_series.loc[first_date:last_date].max() * 1.1)

    return fig


def compare_flow_timeseries(reference, alternative, labels, **kwargs):
    """
    Plots daily timeseries of a water balance flow component over time. Arguments:
        reference: a Pandas Series containing the first time series to plot
        alternative: a Pandas Series containing the second time series to plot
        labels: a list of two Strings for legend labels for the two time series above.
        optional argument `first_date`: a datetime date to specify when to start plotting
        optional argument `last_date`: a datetime date to specify when to stop plotting
        optional argument `y_axis_label`: a String object if different from `alternative`
        optional argument `alternative_2`: a Pandas Series containing a third time series to plot
        Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Optional arguments
    first_date = kwargs.pop("first_date", reference.index[0])
    last_date = kwargs.pop('last_date', reference.index[-1])
    y_axis_label = kwargs.pop('y_axis_label', alternative.name)
    # If there are only two time series to compare
    dummy_array = np.empty(5)
    dummy_array[:] = np.nan
    alternative_2 = kwargs.pop('alternative_2', pd.Series(dummy_array))

    # Plot figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    # Finding maximal flow to set y-axis appropriately
    y_max = max(reference.loc[first_date:last_date].max(), alternative.loc[first_date:last_date].max())
    if alternative_2.hasnans is False:  # There is a third time series
        y_max = max(y_max, alternative_2.loc[first_date:last_date].max())

    # Adding key plots
    ax.plot(reference.index, reference, c='b', linewidth=2, label=labels[0])
    ax.plot(alternative.index, alternative, c='r', linewidth=2, label=labels[1])
    if alternative_2.hasnans is False:  # There is a third time series
        ax.plot(alternative_2.index, alternative_2, c='k', linewidth=2, label=labels[2])

    # Axes and legend specifications
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel(y_axis_label, size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(first_date, last_date)
    ax.set_ylim(0, y_max*1.1)
    ax.legend(prop={'size': 14})

    return fig


def compute_monthly_averages(flows):
    """
    Computes monthly average inflows from a `flows` pandas Series.
    Output:
    averages: a Numpy array of size ((nb_years, 12)) for the 12 average monthly values for each year.
    """

    # Parameter: number of years
    nb_years = flows.index[-1].year - flows.index[0].year + 1

    # Initialise output
    averages = np.zeros((nb_years, 12))

    # Get the time series of monthly flow averages
    monthly_av = flows.resample('ME').mean()

    # Main loop to compute all 12 monthly averages
    for month in range(12):
        averages[:, month] = monthly_av[monthly_av.index.month == month + 1]

    return averages


def monthly_averages(flows, **kwargs):
    """
    Plot monthly average inflows from `flows` pandas Series.
    """

    # Optional argument
    yaxis_label = kwargs.pop('yaxis_label', 'Average inflows (m3/s)')  # Label for the y-axis
    plot_all_years = kwargs.pop('plot_all_years', False)  # Whether we plot individual years or only the average

    # Get monthly average inflows
    monthly_means = compute_monthly_averages(flows)
    annual_cycle = np.mean(monthly_means, axis=0)

    # Plot figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    if plot_all_years is True:
        ax.plot(np.arange(1, 13, 1), annual_cycle, c='black', linewidth=2, label='Average')
        ax.legend()
        for yr in range(monthly_means.shape[0]):
            ax.plot(np.arange(1, 13, 1), monthly_means[yr, :], c='b')
        ax.set_ylim(0, np.max(np.max(monthly_means)))
    ax.plot(np.arange(1, 13, 1), annual_cycle, c='black', linewidth=2)
    plt.xticks(ticks=np.arange(1, 13, 1), labels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Month', size=16)
    ax.set_ylabel(yaxis_label, size=16)
    ax.set_xlim(1, 12)

    return fig


def compare_monthly_averages(reference, alternative, labels, **kwargs):
    """
    Plot a comparison of monthly average inflows from two time series. Arguments:
        reference: pandas DataFrame containing a column with reference inflows
        alternative: pandas DataFrame containing a column with alternative inflows
        labels: list of two Strings, the labels to insert in the figure's legend
    """

    # Optional arguments: third time series to compare
    dummy_array = np.empty(5)
    dummy_array[:] = np.nan
    alternative_2 = kwargs.pop('alternative_2', pd.Series(dummy_array))

    # First, compute monthly averages for both DataFrames
    average_1 = np.sum(compute_monthly_averages(reference), axis=0)
    average_2 = np.sum(compute_monthly_averages(alternative), axis=0)
    if alternative_2.hasnans is False:  # There is a third time series
        average_3 = np.sum(compute_monthly_averages(alternative_2), axis=0)

    # Plot figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(1, 13, 1), average_1, c='b', label=labels[0])
    ax.plot(np.arange(1, 13, 1), average_2, c='r', label=labels[1])
    if alternative_2.hasnans is False:  # There is a third time series
        ax.plot(np.arange(1, 13, 1), average_3, c='k', label=labels[2])
    plt.xticks(ticks=np.arange(1, 13, 1), labels=['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Month', size=16)
    ax.set_ylabel('Average inflows (m3/s)', size=16)
    ax.set_xlim(1, 12)
    ax.legend(prop={'size': 14})

    return fig


def compare_fdc(reference, alternative, labels, **kwargs):

    # Optional arguments: third time series to compare
    dummy_array = np.empty(5)
    dummy_array[:] = np.nan
    alternative_2 = kwargs.pop('alternative_2', pd.Series(dummy_array))

    # Derive Flow Duration Curves (FDC) for the different datasets
    fdc_reference, fdc_probability = compute_fdc(reference)
    fdc_alternative, fdc_probability = compute_fdc(alternative)
    if alternative_2.hasnans is False:  # There is a third time series
        fdc_alternative_2, fdc_probability = compute_fdc(alternative_2)

    # Plot figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale('log')
    ax.plot(fdc_probability * 100, fdc_reference, 'r', linewidth=3, linestyle='dashed', label=labels[0])
    ax.plot(fdc_probability * 100, fdc_alternative, 'b', linewidth=3, label=labels[1])
    if alternative_2.hasnans is False:  # There is a third time series
        ax.plot(fdc_probability * 100, fdc_alternative_2, 'k', linewidth=3, label=labels[2])
    ax.legend(loc='upper right', prop={'size': 14})
    ax.set_xlabel('Exceedance Probability [%]', size=16)
    ax.set_ylabel('Flow rate [$m^3/s$]', size=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.grid()

    return fig


def annual_average(daily_data, data_label):
    """
    Plots the annual average of the chosen time series over time. Arguments:
        daily_data: pandas Series of the daily data to average
        data_label: string of quantity + unit (note first letter should be lowercase).
    Returns the matplotlib figure created, for plotting / saving, etc.
    """

    # Convert daily data to annual
    annual_data = daily_data.resample('YE').sum()

    # PLot this
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(annual_data.index.year, annual_data, c='b')
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel('Annual ' + data_label, size=16)
    ax.set_xlim(annual_data.index.year[0], annual_data.index.year[-1])
    ax.tick_params(axis='both', which='major', labelsize=14)

    return fig


def plot_regret(array, metric_name, ref_dim, lever_list, scenario_list, scenario_label):

    # Compute regret (assuming more of the quantity in `array` is desirable. Otherwise pass `-array` as argument)
    regret = np.zeros(array.shape)
    for i in range(array.shape[ref_dim]):
        if ref_dim == 0:  # Regret as deviation from most favourable lever, for each scenario
            ref_value = np.max(array[i, :])
            regret[i, :] = (ref_value - array[i, :]) / ref_value
        else:  # Regret as deviation from most favourable scenario, for each lever
            ref_value = np.max(array[:, i])
            regret[:, i] = (ref_value - array[:, i]) / ref_value

    # Plot definition
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)

    # Heatmap
    if ref_dim == 0:
        im = ax.imshow(regret, cmap='inferno')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Deviation from most favourable lever', size=14, rotation=-90, va="bottom")
        ax.set_xticks(np.arange(regret.shape[1]))
        ax.set_xticklabels(lever_list)
        ax.set_xlabel('Levers', size=16)
        ax.set_yticks(np.arange(regret.shape[0]))
        ax.set_yticklabels(scenario_list)
        ax.set_ylabel(scenario_label, size=16)

    else:
        im = ax.imshow(np.transpose(regret), cmap='inferno')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Deviation from most favourable scenario', size=14, rotation=-90, va="bottom")
        ax.set_xticks(np.arange(regret.shape[0]))
        ax.set_xticklabels(scenario_list)
        ax.set_xlabel(scenario_label, size=16)
        ax.set_yticks(np.arange(regret.shape[1]))
        ax.set_yticklabels(lever_list)
        ax.set_ylabel('Levers', size=16)


    ax.set_title(metric_name, size=18)

    return fig
