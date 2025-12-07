import pandas as pd
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from io import StringIO
import sys
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def process_glaciers_list(initial_list_df):
    """
    Process the raw glaciers list to get a clean DataFrame.
    """
    # Drop the first 3 rows
    glaciers_list_df = initial_list_df.drop(index=[0, 1]).reset_index(drop=True)

    # Rename the first column to 'SWISS GLACIER LIST (AVAILABLE DATA)'
    glaciers_list_df.columns = ['SWISS GLACIER LIST (AVAILABLE DATA)']

    # Split data into several columns
    glaciers_list_df = glaciers_list_df[
        'SWISS GLACIER LIST (AVAILABLE DATA)'
    ].str.split(',', expand=True)

    # Set first row as column headers
    # Rename columns to include measurement units
    glaciers_list_df.rename(
        columns={
            glaciers_list_df.columns[0]: 'glacier name',
            glaciers_list_df.columns[1]: 'glacier ID',
            glaciers_list_df.columns[2]: 'coordx (X_LV95)',
            glaciers_list_df.columns[3]: 'coordy (Y_LV95)',
            glaciers_list_df.columns[4]: 'glacier area (km2)',
            glaciers_list_df.columns[5]: 'survey year for glacier area (yyyy)',
            glaciers_list_df.columns[6]: 'length change',
            glaciers_list_df.columns[7]: 'mass balance',
            glaciers_list_df.columns[8]: 'volume change'
        },
        inplace=True
    )

    return glaciers_list_df

def process_length_change(initial_length_df):
    """
    Process the raw length change DataFrame.
    """
    # Delete rows 1 & 2
    length_change_df = initial_length_df.drop(index=[0, 1])


    # Rename the first column to 'SWISS GLACIER LENGTH CHANGE'
    length_change_df.columns = ['SWISS GLACIER LENGTH CHANGE']
    
    # Split data into several columns
    length_change_df = length_change_df[
        'SWISS GLACIER LENGTH CHANGE'
    ].str.split(',', expand=True)

    # Reset index
    length_change_df = length_change_df.reset_index(drop=True)
    
    # Rename columns to include measurement units
    length_change_df.rename(
        columns={
            length_change_df.columns[0]: 'glacier name',
            length_change_df.columns[1]: 'glacier ID',
            length_change_df.columns[2]: 'start date of observation (yyyy-mm-dd)',
            length_change_df.columns[3]: 'quality of start date',
            length_change_df.columns[4]: 'end date of observation (yyyy-mm-dd)',
            length_change_df.columns[5]: 'quality of end date',
            length_change_df.columns[6]: 'length change (m)',
            length_change_df.columns[7]: 'elevation of glacier tongue',
            length_change_df.columns[8]: 'observer'
            },
        inplace=True
    )

    return length_change_df


def process_mass_balance_hy(initial_mass_balance_hy_df):
    """
    Process the raw mass balance (hydrological year) DataFrame.
    """
    # Drop unnecessary rows
    mass_balance_hy_df = initial_mass_balance_hy_df.drop(index=[0, 1])
    
    # Rename the first column to 'SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR)'
    mass_balance_hy_df.columns = ['SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR)']

    # Split data into several columns
    mass_balance_hy_df = mass_balance_hy_df[
        'SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR)'
    ].str.split(',', expand=True)

    # Reset index
    mass_balance_hy_df = mass_balance_hy_df.reset_index(drop=True)

    # Drop the original columns 13-16
    mass_balance_hy_df = mass_balance_hy_df.drop(
        columns=[
            mass_balance_hy_df.columns[13],
            mass_balance_hy_df.columns[14],
            mass_balance_hy_df.columns[15],
            mass_balance_hy_df.columns[16]
        ]
    )

    # Rename columns to include units
    mass_balance_hy_df.rename(
        columns={
            mass_balance_hy_df.columns[0]: 'glacier name',
            mass_balance_hy_df.columns[1]: 'glacier ID',
            mass_balance_hy_df.columns[2]: 'start date of observation (yyyy-mm-dd)',
            mass_balance_hy_df.columns[3]: 'end date of winter observation (yyyy-mm-dd)',
            mass_balance_hy_df.columns[4]: 'end date of observation (yyyy-mm-dd)',
            mass_balance_hy_df.columns[5]: 'winter mass balance (mm w.e.)',
            mass_balance_hy_df.columns[6]: 'summer mass balance (mm w.e.)',
            mass_balance_hy_df.columns[7]: 'annual mass balance (mm w.e.)',
            mass_balance_hy_df.columns[8]: 'equilibrium line altitude (m asl.)',
            mass_balance_hy_df.columns[9]: 'accumulation area ratio (%)',
            mass_balance_hy_df.columns[10]: 'glacier area (km2)',
            mass_balance_hy_df.columns[11]: 'minimum elevation of glacier (m asl.)',
            mass_balance_hy_df.columns[12]: 'maximum elevation of glacier (m asl.)'
        },
        inplace=True
    )

    return mass_balance_hy_df

def process_mass_balance_hy_eb(initial_mass_balance_hy_eb_df):
    """
    Process the raw mass balance (hydrological year with elevation bins) DataFrame.
    """
    # Drop unnecessary rows
    mass_balance_hy_eb_df = initial_mass_balance_hy_eb_df.drop(index=[0, 1])

    # Rename the first column to 'SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR) ELEVATION BINS'
    mass_balance_hy_eb_df.columns = ['SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR) ELEVATION BINS']

    # Split data into several columns
    mass_balance_hy_eb_df = mass_balance_hy_eb_df[
        'SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR) ELEVATION BINS'
    ].str.split(',', expand=True)

    # Reset index
    mass_balance_hy_eb_df = mass_balance_hy_eb_df.reset_index(drop=True)

    # Drop the original columns 11-14
    mass_balance_hy_eb_df = mass_balance_hy_eb_df.drop(
        columns=[
            mass_balance_hy_eb_df.columns[11],
            mass_balance_hy_eb_df.columns[12],
            mass_balance_hy_eb_df.columns[13],
            mass_balance_hy_eb_df.columns[14]
        ]
    )

    # Rename columns to include units
    mass_balance_hy_eb_df.rename(
        columns={
            mass_balance_hy_eb_df.columns[0]: 'glacier name',
            mass_balance_hy_eb_df.columns[1]: 'glacier ID',
            mass_balance_hy_eb_df.columns[2]: 'start date of observation (yyyy-mm-dd)',
            mass_balance_hy_eb_df.columns[3]: 'end date of winter observation (yyyy-mm-dd)',
            mass_balance_hy_eb_df.columns[4]: 'end date of observation (yyyy-mm-dd)',
            mass_balance_hy_eb_df.columns[5]: 'winter mass balance (mm w.e.)',
            mass_balance_hy_eb_df.columns[6]: 'summer mass balance (mm w.e.)',
            mass_balance_hy_eb_df.columns[7]: 'annual mass balance (mm w.e.)',
            mass_balance_hy_eb_df.columns[8]: 'area of elevation bin (km2)',
            mass_balance_hy_eb_df.columns[9]: 'lower elevation of bin (m asl.)',
            mass_balance_hy_eb_df.columns[10]: 'upper elevation of bin (m asl.)'
        },
        inplace=True
    )

    return mass_balance_hy_eb_df


def clean_weather_metadata(metadata_raw):
    """
    Clean weather metadata by selecting relevant columns.
    """
    if metadata_raw is not None:
        metadata = metadata_raw[['parameter_shortname', 'parameter_description_en', 'parameter_unit']]
        return metadata
    else:
        return None

def process_city_weather(city_weather_raw):
    """
    Process weather data for a city DataFrame:
    - Convert 'reference_timestamp' to datetime and set as index.
    - Filter data between 1914-10-01 and 2025-10-01.
    - Drop NA columns and 'reference_timestamp'.
    - Add 'year-month' column and reorder columns.
    - Select and rename 'rhs150m0' and 'ths200m0' columns.
    """
    # Convert 'reference_timestamp' to datetime and set as index
    city_weather_raw['date'] = pd.to_datetime(city_weather_raw['reference_timestamp'], format='%d.%m.%Y %H:%M')
    city_weather = city_weather_raw.drop(columns=['station_abbr'])
    city_weather = city_weather.set_index('date')

    # Filter data between 1914-10-01 and 2025-10-01
    city_1914 = city_weather[
        (city_weather.index >= '1914-10-01') &
        (city_weather.index < '2025-10-01')
    ]
    city_1914 = city_1914.dropna(axis=1)
    city_1914 = city_1914.drop('reference_timestamp', axis=1, errors='ignore')

    # Add 'year-month' column and reorder
    city_1914['year-month'] = city_1914.index.to_series().dt.to_period('M').astype(str)
    cols = city_1914.columns.tolist()
    cols.insert(0, cols.pop(cols.index('year-month')))
    city_1914 = city_1914[cols]

    # Select and rename columns
    city_processed = city_1914[['year-month', 'rhs150m0', 'ths200m0']].copy()
    city_processed.rename(
        columns={
            'rhs150m0': 'total precipitation (mm)',
            'ths200m0': 'mean temperature (°C)'
        },
        inplace=True
    )

    return city_processed

def calculate_hydrological_year_aggregates(city_data):
    """
    Calculate hydrological year aggregates for precipitation and temperature from city weather data.
    - Groups data into hydrological years (October to September).
    - Calculates total precipitation and mean temperature for each hydrological year.
    """
    # Calculate total precipitation for hydrological years
    city_p = city_data[['total precipitation (mm)']].copy()
    hy_data_p = city_p.groupby(np.arange(len(city_p)) // 12).sum()

    # Generate the date range for hydrological years (October to September)
    start_date = '1914-10-01'
    end_date = '2024-10-01'
    hy_dates = pd.date_range(start=start_date, end=end_date, freq='YS-OCT')
    hy_data_p.index = hy_dates

    # Calculate mean temperature for hydrological years
    city_temp = city_data[['mean temperature (°C)']].copy()
    city_temp["days_in_month"] = city_temp.index.days_in_month
    city_temp['temp_times_days'] = city_temp['mean temperature (°C)'] * city_temp['days_in_month']

    n = 12  # Number of rows per group (12 months)
    hy_mean_temp = pd.DataFrame({
        'hy mean temperature (°C)': [
            city_temp['temp_times_days'].iloc[i:i + n].sum() /
            city_temp['days_in_month'].iloc[i:i + n].sum()
            for i in range(0, len(city_temp), n)
        ]
    })
    hy_mean_temp = hy_mean_temp.round(1)
    hy_mean_temp.index = hy_dates

    # Combine precipitation and temperature data
    hy_data = pd.concat([hy_data_p, hy_mean_temp], axis=1)
    hy_data = hy_data.round(1)
    hy_data["date"] = hy_data.index
    hy_data.rename(columns={'total precipitation (mm)': 'hy total precipitation (mm)'}, inplace=True)

    return hy_data

def calculate_seasonal_aggregates(city_data):
    """
    Calculate seasonal aggregates for precipitation and temperature from city weather data.
    Returns two DataFrames: winter and summer seasonal aggregates.
    """
    # Define winter and summer months
    winter_months = [10, 11, 12, 1, 2, 3, 4]  # Oct–Apr
    summer_months = [5, 6, 7, 8, 9]           # May–Sep

    def classify_season(date):
        if date.month in winter_months:
            return 'winter'
        else:
            return 'summer'

    def season_year(date):
        # Winter belongs to the year of January (e.g., winter 1914-15 -> 1915)
        if date.month >= 10:
            return date.year + 1  # Oct–Dec -> next year
        else:
            return date.year

    def compute_season_date(row):
        season_year, season = row.name  # MultiIndex: (season_year, season)
        if season == "winter":
            return pd.Timestamp(season_year - 1, 10, 1)
        else:  # summer
            return pd.Timestamp(season_year, 5, 1)

    # Create a copy of the city data
    city_seasonal = city_data.copy()

    # Classify seasons and season years
    city_seasonal['season'] = city_seasonal.index.to_series().apply(classify_season)
    city_seasonal['season_year'] = city_seasonal.index.to_series().apply(season_year)

    # Calculate seasonal precipitation
    seasonal_sum = (
        city_seasonal[['total precipitation (mm)', 'season', 'season_year']]
        .groupby(['season_year', 'season'])
        .sum()
    )

    # Calculate seasonal temperature
    city_seasonal['days_in_month'] = city_seasonal.index.days_in_month
    city_seasonal['temp_times_days'] = city_seasonal['mean temperature (°C)'] * city_seasonal['days_in_month']
    seasonal_temp = (
        city_seasonal.groupby(['season_year', 'season'])
        .apply(lambda df: df['temp_times_days'].sum() / df['days_in_month'].sum())
        .to_frame(name='mean seasonal temperature (°C)')
    )

    # Combine seasonal data
    seasonal_data = pd.concat([seasonal_sum, seasonal_temp], axis=1)
    seasonal_data = seasonal_data.round(1)
    seasonal_data['date'] = seasonal_data.apply(compute_season_date, axis=1)
    seasonal_data = seasonal_data.set_index('date').sort_index().reset_index()

    # Separate winter and summer data
    weather_winter = seasonal_data.iloc[::2, :].reset_index(drop=True)
    weather_summer = seasonal_data.iloc[1::2, :].reset_index(drop=True)

    # Rename columns
    weather_summer.rename(
        columns={
            'total precipitation (mm)': 'summer total precipitation (mm)',
            'mean seasonal temperature (°C)': 'summer mean temperature (°C)'
        },
        inplace=True
    )

    weather_winter.rename(
        columns={
            'total precipitation (mm)': 'winter total precipitation (mm)',
            'mean seasonal temperature (°C)': 'winter mean temperature (°C)'
        },
        inplace=True
    )

    return weather_winter, weather_summer

def calculate_weather_deviations_1961_1990(city_data, city_name):
    """
    Calculate monthly temperature and precipitation deviations from 1961-1990 norms.
    Returns two DataFrames: temp_dev_1961_1990 and precip_dev_1961_1990.
    """
    # Define norms for each city (temperature and precipitation)
    norms = {
        'sion': {
            'temp': [9.5, 3.4, -0.4, -0.8, -1.6, -5.3, -9.4, -13.7, -17.0, -19.1, -17.9, -14.6],
            'precip': [50, 60, 61, 53, 57, 48, 36, 41, 52, 48, 55, 38]
        },
        'davos': {
            'temp': [4.7, -1.0, -4.4, -5.3, -4.7, -2.2, 1.3, 5.9, 9.0, 11.3, 10.8, 8.3],
            'precip': [58, 66, 65, 68, 59, 60, 55, 91, 120, 132, 135, 90]
        },
        'altdorf': {
            'temp': [9.8, 4.6, 1.0, 0.5, 1.9, 4.6, 8.2, 12.5, 15.5, 17.6, 16.8, 14.0],
            'precip': [73, 81, 75, 71, 68, 71, 86, 99, 130, 132, 133, 86]
        }
    }

    # Extract temperature and precipitation data
    city_temp = city_data[['ths200m0']].copy().reset_index(drop=True)
    city_precip = city_data[['rhs150m0']].copy().reset_index(drop=True)

    # Pivot temperature data
    city_temp['year'] = city_temp.index // 12
    city_temp['month'] = city_temp.index % 12
    temp_dev_1961_1990 = city_temp.pivot(index='year', columns='month', values='ths200m0')

    # Rename columns for temperature deviations
    monthly_observations_t = [
        'october_td', 'november_td', 'december_td',
        'january_td', 'february_td', 'march_td',
        'april_td', 'may_td', 'june_td',
        'july_td', 'august_td', 'september_td'
    ]
    temp_dev_1961_1990.columns = monthly_observations_t

    # Add hydrological year column
    temp_dev_1961_1990['hydrological year'] = [f"{1914 + i}-{1915 + i}" for i in range(len(temp_dev_1961_1990))]

    # Calculate temperature deviations
    for i, month in enumerate(monthly_observations_t):
        temp_dev_1961_1990[month] -= norms[city_name]['temp'][i]

    # Pivot precipitation data
    city_precip['year'] = city_precip.index // 12
    city_precip['month'] = city_precip.index % 12
    precip_dev_1961_1990 = city_precip.pivot(index='year', columns='month', values='rhs150m0')

    # Rename columns for precipitation deviations
    monthly_observations_p = [
        'october_pd', 'november_pd', 'december_pd',
        'january_pd', 'february_pd', 'march_pd',
        'april_pd', 'may_pd', 'june_pd',
        'july_pd', 'august_pd', 'september_pd'
    ]
    precip_dev_1961_1990.columns = monthly_observations_p

    # Add hydrological year column
    precip_dev_1961_1990['hydrological year'] = [f"{1914 + i}-{1915 + i}" for i in range(len(precip_dev_1961_1990))]

    # Calculate precipitation deviations
    for i, month in enumerate(monthly_observations_p):
        precip_dev_1961_1990[month] -= norms[city_name]['precip'][i]

    # Add seasonal aggregates for precipitation
    precip_dev_1961_1990['opt_season_pd'] = precip_dev_1961_1990[
        ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd']
    ].sum(axis=1)
    precip_dev_1961_1990['opt_season+march_pd'] = precip_dev_1961_1990[
        ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd', 'march_pd']
    ].sum(axis=1)
    precip_dev_1961_1990['winter_pd'] = precip_dev_1961_1990[
        ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd', 'march_pd', 'april_pd']
    ].sum(axis=1)

    # Add seasonal aggregates for temperature
    temp_dev_1961_1990['opt_season_td'] = (
        temp_dev_1961_1990['may_td'] * 31 +
        temp_dev_1961_1990['june_td'] * 30 +
        temp_dev_1961_1990['july_td'] * 31 +
        temp_dev_1961_1990['august_td'] * 31
    ) / 123
    temp_dev_1961_1990['summer_td'] = (
        temp_dev_1961_1990['may_td'] * 31 +
        temp_dev_1961_1990['june_td'] * 30 +
        temp_dev_1961_1990['july_td'] * 31 +
        temp_dev_1961_1990['august_td'] * 31 +
        temp_dev_1961_1990['september_td'] * 30
    ) / 153

    # Round values
    temp_dev_1961_1990 = temp_dev_1961_1990.round(1)
    precip_dev_1961_1990 = precip_dev_1961_1990.round(1)

    return temp_dev_1961_1990, precip_dev_1961_1990


def calculate_weather_deviations_1991_2020(city_data, city_name):
    """
    Calculate monthly temperature deviations (from th9120mv) and precipitation deviations from 1991-2020 norms.
    Returns two DataFrames: temp_dev_1991_2020 and precip_dev_1991_2020.
    """
    # Define precipitation norms for each city (1991-2020)
    precip_norms = {
        'sion': [43, 50, 68, 52, 40, 37, 34, 52, 62, 60, 38],
        'davos': [77, 71, 68, 70, 52, 57, 54, 89, 133, 150, 96],
        'altdorf': [84, 81, 83, 70, 59, 72, 81, 117, 141, 154, 105]
    }

    # Extract temperature deviation and precipitation data
    city_temp_dev = city_data[['th9120mv']].copy().reset_index(drop=True)
    city_precip = city_data[['rhs150m0']].copy().reset_index(drop=True)

    # Pivot temperature deviation data
    city_temp_dev['year'] = city_temp_dev.index // 12
    city_temp_dev['month'] = city_temp_dev.index % 12
    temp_dev_1991_2020 = city_temp_dev.pivot(index='year', columns='month', values='th9120mv')

    # Rename columns for temperature deviations
    monthly_observations_t = [
        'october_td', 'november_td', 'december_td',
        'january_td', 'february_td', 'march_td',
        'april_td', 'may_td', 'june_td',
        'july_td', 'august_td', 'september_td'
    ]
    temp_dev_1991_2020.columns = monthly_observations_t

    # Add hydrological year column
    temp_dev_1991_2020['hydrological year'] = [f"{1914 + i}-{1915 + i}" for i in range(len(temp_dev_1991_2020))]
    temp_dev_1991_2020 = temp_dev_1991_2020.reset_index(drop=True)

    # Pivot precipitation data
    city_precip['year'] = city_precip.index // 12
    city_precip['month'] = city_precip.index % 12
    precip_dev_1991_2020 = city_precip.pivot(index='year', columns='month', values='rhs150m0')

    # Rename columns for precipitation deviations
    monthly_observations_p = [
        'october_pd', 'november_pd', 'december_pd',
        'january_pd', 'february_pd', 'march_pd',
        'april_pd', 'may_pd', 'june_pd',
        'july_pd', 'august_pd', 'september_pd'
    ]
    precip_dev_1991_2020.columns = monthly_observations_p

    # Add hydrological year column
    precip_dev_1991_2020['hydrological year'] = [f"{1914 + i}-{1915 + i}" for i in range(len(precip_dev_1991_2020))]
    precip_dev_1991_2020 = precip_dev_1991_2020.reset_index(drop=True)

    # Calculate precipitation deviations
    for i, month in enumerate(monthly_observations_p[:-1]):  # Exclude September for now
        precip_dev_1991_2020[month] -= precip_norms[city_name][i]

    # Add September norm (last element in the list)
    precip_dev_1991_2020['september_pd'] -= precip_norms[city_name][-1]

    # Add seasonal aggregates for precipitation
    precip_dev_1991_2020['opt_season_pd'] = precip_dev_1991_2020[
        ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd']
    ].sum(axis=1)
    precip_dev_1991_2020['opt_season+march_pd'] = precip_dev_1991_2020[
        ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd', 'march_pd']
    ].sum(axis=1)
    precip_dev_1991_2020['winter_pd'] = precip_dev_1991_2020[
        ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd', 'march_pd', 'april_pd']
    ].sum(axis=1)

    # Add seasonal aggregates for temperature
    temp_dev_1991_2020['opt_season_td'] = (
        temp_dev_1991_2020['may_td'] * 31 +
        temp_dev_1991_2020['june_td'] * 30 +
        temp_dev_1991_2020['july_td'] * 31 +
        temp_dev_1991_2020['august_td'] * 31
    ) / 123
    temp_dev_1991_2020['summer_td'] = (
        temp_dev_1991_2020['may_td'] * 31 +
        temp_dev_1991_2020['june_td'] * 30 +
        temp_dev_1991_2020['july_td'] * 31 +
        temp_dev_1991_2020['august_td'] * 31 +
        temp_dev_1991_2020['september_td'] * 30
    ) / 153

    # Round values
    temp_dev_1991_2020 = temp_dev_1991_2020.round(1)
    precip_dev_1991_2020 = precip_dev_1991_2020.round(1)

    return temp_dev_1991_2020, precip_dev_1991_2020

