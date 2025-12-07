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

# --- Data Preparation Functions ---
def prepare_glacier_data(glacier_name, length_change_df):
    glacier_df = length_change_df[length_change_df['glacier name'] == glacier_name].copy()
    if glacier_df.empty:
        print(f"No data found for {glacier_name}.")
        return None
    glacier_df = glacier_df[[
        'start date of observation (yyyy-mm-dd)',
        'end date of observation (yyyy-mm-dd)',
        'length change (m)'
    ]]
    glacier_df['date'] = pd.to_datetime(glacier_df['end date of observation (yyyy-mm-dd)'])
    glacier_df = glacier_df[['length change (m)', 'date']]
    glacier_df['cumulative length change (m)'] = glacier_df['length change (m)'].cumsum()
    glacier_df = glacier_df.set_index('date')
    return glacier_df

def prepare_mass_balance_data(glacier_name, mass_balance_hy_df, balance_type="annual"):
    mb_df = mass_balance_hy_df[mass_balance_hy_df["glacier name"] == glacier_name].copy()
    if mb_df.empty:
        print(f"No data found for {glacier_name}.")
        return None
    mb_df = mb_df[
        [
            "start date of observation (yyyy-mm-dd)",
            "end date of observation (yyyy-mm-dd)",
            f"{balance_type} mass balance (mm w.e.)",
        ]
    ]
    mb_df["end date"] = pd.to_datetime(mb_df["end date of observation (yyyy-mm-dd)"])
    mb_df = mb_df[[f"{balance_type} mass balance (mm w.e.)", "end date"]]
    if balance_type == "annual":
        mb_df[f"cumulative {balance_type} mass balance (mm w.e.)"] = mb_df[
            f"{balance_type} mass balance (mm w.e.)"
        ].cumsum()
    mb_df = mb_df.set_index("end date")
    return mb_df

# --- Glacier Plotting Functions ---
def plot_glacier_cumulative_length_change(glacier_df, glacier_name, figsize=(12, 6)):
    if glacier_df is None:
        return None
    annual_df = glacier_df.resample('YS').last()
    fig_cl, ax = plt.subplots(figsize=figsize)
    ax.plot(
        annual_df.index,
        annual_df['cumulative length change (m)'],
        linestyle='-',
        color='skyblue',
        marker='o',
        markersize=4,
        markerfacecolor='skyblue',
        markeredgecolor='skyblue'
    )
    ax.set_ylabel('Cumulative Length Change [m]')
    ax.set_title(f'{glacier_name} Cumulative Length Change Over Time')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig_cl

def plot_glacier_length_change_bar(glacier_df, glacier_name, figsize=(12, 6)):
    if glacier_df is None:
        return None
    fig_l, ax = plt.subplots(figsize=figsize)
    dates = glacier_df.index
    bar_width = (dates.max() - dates.min()) / len(dates) / 2
    ax.bar(
        dates,
        glacier_df['length change (m)'],
        color='skyblue',
        width=bar_width
    )
    ax.set_ylabel('Length Change [m]')
    ax.set_title(f'{glacier_name} Length Change Over Time')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig_l

def plot_mass_balance_bar(mb_df, glacier_name, balance_type, figsize=(12, 6)):
    if mb_df is None:
        return None
    fig_mb, ax = plt.subplots(figsize=figsize)
    dates = mb_df.index
    bar_width = (dates.max() - dates.min()) / len(dates) / 2
    ax.bar(
        dates,
        mb_df[f"{balance_type} mass balance (mm w.e.)"],
        color="skyblue",
        width=bar_width,
    )
    ax.set_ylabel(f"{balance_type.capitalize()} Mass Balance [mm w.e.]")
    ax.set_title(f"{glacier_name} {balance_type.capitalize()} Mass Balance Over Time")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig_mb

def plot_cumulative_mass_balance(mb_df, glacier_name, figsize=(12, 6)):
    if mb_df is None:
        return None
    annual_mb_df = mb_df.resample("YS").last()
    fig_cmb, ax = plt.subplots(figsize=figsize)
    ax.plot(
        annual_mb_df.index,
        annual_mb_df["cumulative annual mass balance (mm w.e.)"],
        linestyle="-",
        color="skyblue",
        marker="o",
        markersize=4,
        markerfacecolor="skyblue",
        markeredgecolor="skyblue",
    )
    ax.set_ylabel("Cumulative Annual Mass Balance [mm w.e.]")
    ax.set_title(f"{glacier_name} Cumulative Annual Mass Balance Over Time")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    return fig_cmb

def plot_mass_balance_for_glaciers_eb(mass_balance_hy_eb_df, glacier_name, figsize=(12, 6)):
    glacier_mb_eb_df = mass_balance_hy_eb_df[
        mass_balance_hy_eb_df['glacier name'] == glacier_name
    ].copy()
    glacier_mb_eb_df = glacier_mb_eb_df[
        [
            'start date of observation (yyyy-mm-dd)',
            'end date of observation (yyyy-mm-dd)',
            'upper elevation of bin (m asl.)',
            'annual mass balance (mm w.e.)'
        ]
    ]
    glacier_mb_eb_df = glacier_mb_eb_df.reset_index(drop=True)
    glacier_mb_eb_df['end date'] = pd.to_datetime(
        glacier_mb_eb_df['end date of observation (yyyy-mm-dd)']
    )
    glacier_mb_eb_df = glacier_mb_eb_df[
        [
            'annual mass balance (mm w.e.)',
            'upper elevation of bin (m asl.)',
            'end date'
        ]
    ]
    elevations = sorted(glacier_mb_eb_df['upper elevation of bin (m asl.)'].unique())
    dark_blue_to_dark_gray = LinearSegmentedColormap.from_list(
        'dark_blue_to_dark_gray', ['#003366', '#555555']
    )
    colors = dark_blue_to_dark_gray(np.linspace(0, 1, len(elevations)))
    fig_mb_eb, ax = plt.subplots(figsize=figsize)
    for i, (elev, group) in enumerate(glacier_mb_eb_df.groupby('upper elevation of bin (m asl.)')):
        group = group.sort_values('end date')
        for j in range(len(group) - 1):
            current_year = group['end date'].iloc[j]
            next_year = group['end date'].iloc[j + 1]
            if (next_year - current_year).days <= 366:
                ax.plot(
                    group['end date'].iloc[j:j+2],
                    group['annual mass balance (mm w.e.)'].iloc[j:j+2],
                    alpha=0.6,
                    color=colors[i]
                )
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual Mass Balance (mm w.e.)')
    ax.set_title(f'Annual Mass Balance for each Elevation Bin over Time - {glacier_name}')
    ax.grid(True, alpha=0.8)
    plt.tight_layout()
    return fig_mb_eb

# --- Weather Plotting Functions ---
def plot_summer_temperature(summer_df, city_name, norm6190, norm9120, figsize=(12, 6)):
    fig_t, ax = plt.subplots(figsize=figsize)
    ax.plot(summer_df['date'],
            summer_df['summer mean temperature (°C)'],
            label='Summer Mean Temperature (°C)',
            color='red')
    ax.axhline(y=norm6190,
               color='black',
               linestyle='--',
               label=f'1961-1990 Norm ({norm6190:.1f}°C)')
    ax.axhline(y=norm9120,
               color='black',
               label=f'1991-2020 Norm ({norm9120:.1f}°C)')
    ax.set_title(f'{city_name} Summer Mean Temperature')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Temperature (°C)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig_t

def plot_winter_precipitation(winter_df, city_name, norm6190, norm9120, figsize=(12, 6)):
    fig_p, ax = plt.subplots(figsize=figsize)
    ax.plot(winter_df['date'],
            winter_df['winter total precipitation (mm)'],
            label='Winter Total Precipitation (mm)',
            color='blue')
    ax.axhline(y=norm6190,
               color='black',
               linestyle='--',
               label=f'1961-1990 Norm ({norm6190:.1f} mm)')
    ax.axhline(y=norm9120,
               color='black',
               label=f'1991-2020 Norm ({norm9120:.1f} mm)')
    ax.set_title(f'{city_name} Winter Total Precipitation')
    ax.set_xlabel('Year')
    ax.set_ylabel('Precipitation (mm)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig_p

# --- Mass Balance vs. Weather Plotting Function ---
def plot_mass_balance_weather(
    glacier_name,
    temp_data,
    temp_column,
    mass_balance_summer,
    mass_balance_winter=None,
    precip_data=None,
    precip_column=None,
    figsize=(12, 6)
):
    fig_smb_st, ax1 = plt.subplots(figsize=figsize)
    ax1.scatter(temp_data[temp_column],
                mass_balance_summer['summer mass balance (mm w.e.)'])
    ax1.set_xlabel('Summer Mean Temperature (°C)')
    ax1.set_ylabel('Summer Mass Balance (mm w.e.)')
    ax1.set_title(f"{glacier_name} Summer Mass Balance with relation to Temperature")
    ax1.grid(True)
    plt.tight_layout()

    fig_wmb_wp = None
    if mass_balance_winter is not None and precip_data is not None:
        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.scatter(precip_data[precip_column],
                    mass_balance_winter['winter mass balance (mm w.e.)'])
        ax2.set_xlabel('Winter Total Precipitation (mm)')
        ax2.set_ylabel('Winter Mass Balance (mm w.e.)')
        ax2.set_title(f"{glacier_name} Winter Mass Balance with relation to Precipitation")
        ax2.grid(True)
        plt.tight_layout()

    return fig_smb_st, fig_wmb_wp