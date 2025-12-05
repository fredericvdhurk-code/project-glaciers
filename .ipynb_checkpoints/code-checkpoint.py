import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

length_change_df = pd.read_csv('project-glaciers/data/length_change.csv')
volume_change_df = pd.read_csv('project-glaciers/data/volume_change.csv')
mass_balance_hy_df = pd.read_csv('project-glaciers/data/mass_balance_hy.csv')
mass_balance_hy_eb_df = pd.read_csv('project-glaciers/data/mass_balance_hy_eb.csv')
davos_summer =  pd.read_csv('project-glaciers/data/weather_data_davos_summer.csv')
davos_winter = pd.read_csv('project-glaciers/data/weather_data_davos_winter.csv')
altdorf_summer =  pd.read_csv('project-glaciers/data/weather_data_altdorf_summer.csv')
altdorf_winter = pd.read_csv('project-glaciers/data/weather_data_altdorf_winter.csv')
sion_summer =  pd.read_csv('project-glaciers/data/weather_data_sion_summer.csv')
sion_winter = pd.read_csv('project-glaciers/data/weather_data_sion_winter.csv')

# Convert the date column to datetime for all weather dataframes
for df in [altdorf_summer, altdorf_winter,
           davos_summer, davos_winter,
           sion_summer, sion_winter]:
    # Assuming the date column is named 'date' or similar; adjust if needed
    date_col = 'date'  # Replace with the actual column name if different
    df[date_col] = pd.to_datetime(df[date_col])


# 1961-1990 climate norms

altdorf_summer_norm_6190_t = 15.3
altdorf_winter_norm_6190_p = 525

davos_summer_norm_6190_t = 9.1
davos_winter_norm_6190_p = 431

sion_summer_norm_6190_t = 16.5
sion_winter_norm_6190_p = 365

# 1991-2020 climate norms

altdorf_summer_norm_9120_t = 16.7
altdorf_winter_norm_9120_p = 530

davos_summer_norm_9120_t = 10.5
davos_winter_norm_9120_p = 449

sion_summer_norm_9120_t = 18.1
sion_winter_norm_9120_p = 324



def prepare_glacier_data(glacier_name, length_change_df):
    """
    Prepare the length change data for a specific glacier.
    Returns None if no data is found.
    """
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

def plot_glacier_cumulative_length_change(glacier_df, glacier_name, figsize=(12, 6)):
    """Plot the cumulative length change for a glacier as a line chart with points and gaps."""
    if glacier_df is None:
        return

    # Resample to annual frequency, filling missing years with NaN
    annual_df = glacier_df.resample('YS').last()

    plt.figure(figsize=figsize)
    # Plot line with gaps and blue points
    plt.plot(
        annual_df.index,
        annual_df['cumulative length change (m)'],
        linestyle='-',
        color='skyblue',
        marker='o',
        markersize=4,  # Small points
        markerfacecolor='skyblue',  # Blue points
        markeredgecolor='skyblue'
    )
    plt.xlabel('Year')
    plt.ylabel('Cumulative Length Change [m]')
    plt.title(f'{glacier_name} Cumulative Length Change Over Time')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_glacier_length_change_bar(glacier_df, glacier_name, figsize=(12, 6)):
    """Plot the length change for a glacier as a bar chart."""
    if glacier_df is None:
        return

    plt.figure(figsize=figsize)
    dates = glacier_df.index
    bar_width = (dates.max() - dates.min()) / len(dates) / 2
    plt.bar(
        dates,
        glacier_df['length change (m)'],
        color='skyblue',
        width=bar_width
    )
    plt.xlabel('Year')
    plt.ylabel('Length Change [m]')
    plt.title(f'{glacier_name} Length Change Over Time')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# List of glaciers to analyze
glaciers = [
    'Rhonegletscher',
    'Grosser Aletschgletscher',
    'Silvrettagletscher',
    'Claridenfirn',
    'Griesgletscher',
    'Allalingletscher',
    'Schwarzberggletscher',
    'Hohlaubgletscher',
    'Glacier du Giétro'
]

# Generate and display plots for each glacier
#for glacier in glaciers:
#    glacier_df = prepare_glacier_data(glacier, length_change_df)
#    if glacier_df is not None:
#        plot_glacier_length_change_bar(glacier_df, glacier)
#        plot_glacier_cumulative_length_change(glacier_df, glacier)
#    else:
#        print(f"Skipping {glacier} (no data).")



def prepare_mass_balance_data(glacier_name, mass_balance_df, balance_type="annual"):
    """
    Prepare mass balance data for a specific glacier and type.
    Args:
        glacier_name (str): Name of the glacier.
        mass_balance_df (pd.DataFrame): DataFrame containing mass balance data.
        balance_type (str): "summer", "winter", or "annual".
    Returns:
        pd.DataFrame: DataFrame with mass balance and cumulative mass balance.
    """
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

def plot_mass_balance_bar(mb_df, glacier_name, balance_type, figsize=(12, 6)):
    """
    Plot mass balance as a bar chart.
    Args:
        mb_df (pd.DataFrame): DataFrame with mass balance data.
        glacier_name (str): Name of the glacier.
        balance_type (str): "summer", "winter", or "annual".
        figsize (tuple): Figure size.
    """
    if mb_df is None:
        return
    plt.figure(figsize=figsize)
    dates = mb_df.index
    if glacier_name == "Rhonegletscher":
        # Custom bar width for Rhonegletscher (adjust as needed)
        bar_width = 200  # in days
    else:
        # Default bar width calculation for other glaciers
        bar_width = (dates.max() - dates.min()) / len(dates) / 2
    plt.bar(
        dates,
        mb_df[f"{balance_type} mass balance (mm w.e.)"],
        color="skyblue",
        width=bar_width,
    )
    plt.ylabel(f"{balance_type.capitalize()} Mass Balance [mm w.e.]")
    plt.title(f"{glacier_name} {balance_type.capitalize()} Mass Balance Over Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_cumulative_mass_balance(mb_df, glacier_name, figsize=(10, 6)):
    """
    Plot cumulative annual mass balance as a line chart with points and gaps.
    Args:
        mb_df (pd.DataFrame): DataFrame with mass balance data.
        glacier_name (str): Name of the glacier.
        figsize (tuple): Figure size.
    """
    if mb_df is None:
        return
    # Resample to annual frequency to create gaps for missing years
    annual_mb_df = mb_df.resample("YS").last()
    plt.figure(figsize=figsize)
    plt.plot(
        annual_mb_df.index,
        annual_mb_df["cumulative annual mass balance (mm w.e.)"],
        linestyle="-",
        color="skyblue",
        marker="o",
        markersize=4,
        markerfacecolor="skyblue",
        markeredgecolor="skyblue",
    )
    plt.ylabel("Cumulative Annual Mass Balance [mm w.e.]")
    plt.title(f"{glacier_name} Cumulative Annual Mass Balance Over Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    mass_balance_hy_df = pd.read_csv("project-glaciers/data/mass_balance_hy.csv")
    mass_balance_hy_eb_df = pd.read_csv("project-glaciers/data/mass_balance_hy_eb.csv")

    # List of glaciers to analyze
    glaciers = [
        "Rhonegletscher",
        "Grosser Aletschgletscher",
        "Silvrettagletscher",
        "Claridenfirn",
        "Griesgletscher",
        "Allalingletscher",
        "Schwarzberggletscher",
        "Hohlaubgletscher",
        "Glacier du Giétro",
    ]

    # Generate and display plots for each glacier
#    for glacier in glaciers:
        # Summer mass balance
#        summer_mb_df = prepare_mass_balance_data(
#            glacier, mass_balance_hy_eb_df, balance_type="summer"
#        )
#        if summer_mb_df is not None:
#            plot_mass_balance_bar(summer_mb_df, glacier, "summer")

        # Winter mass balance
#        winter_mb_df = prepare_mass_balance_data(
#            glacier, mass_balance_hy_eb_df, balance_type="winter"
#        )
#        if winter_mb_df is not None:
#            plot_mass_balance_bar(winter_mb_df, glacier, "winter")

        # Annual mass balance
#        annual_mb_df = prepare_mass_balance_data(
#            glacier, mass_balance_hy_df, balance_type="annual"
#        )
#        if annual_mb_df is not None:
#            plot_mass_balance_bar(annual_mb_df, glacier, "annual")
#            plot_cumulative_mass_balance(annual_mb_df, glacier)

#if __name__ == "__main__":
#    main()


def plot_mass_balance_for_glaciers(mass_balance_hy_eb_df, glacier_names):
    """
    Plots the annual mass balance for each elevation bin over time for multiple glaciers.
    Colors range from dark blue (low elevation) to dark gray (high elevation).
    Lines are discontinuous for missing years.
    """
    for glacier in glacier_names:
        # Filter data for the current glacier
        glacier_mb_eb_df = mass_balance_hy_eb_df[
            mass_balance_hy_eb_df['glacier name'] == glacier
        ].copy()
        # Select relevant columns
        glacier_mb_eb_df = glacier_mb_eb_df[
            [
                'start date of observation (yyyy-mm-dd)',
                'end date of observation (yyyy-mm-dd)',
                'upper elevation of bin (m asl.)',
                'annual mass balance (mm w.e.)'
            ]
        ]
        # Reset index and convert 'end date' to datetime
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

        # Get unique elevations and sort them
        elevations = sorted(glacier_mb_eb_df['upper elevation of bin (m asl.)'].unique())

        # Create a custom colormap: dark blue to dark gray
        dark_blue_to_dark_gray = LinearSegmentedColormap.from_list(
            'dark_blue_to_dark_gray', ['#003366', '#555555']
        )
        colors = dark_blue_to_dark_gray(np.linspace(0, 1, len(elevations)))

        # Plot
        plt.figure(figsize=(10, 6))
        for i, (elev, group) in enumerate(glacier_mb_eb_df.groupby('upper elevation of bin (m asl.)')):
            # Sort by date
            group = group.sort_values('end date')
            # Plot each segment, breaking for gaps > 1 year
            for j in range(len(group) - 1):
                current_year = group['end date'].iloc[j]
                next_year = group['end date'].iloc[j + 1]
                if (next_year - current_year).days <= 366:  # Allow for leap years
                    plt.plot(
                        group['end date'].iloc[j:j+2],
                        group['annual mass balance (mm w.e.)'].iloc[j:j+2],
                        alpha=0.6,
                        color=colors[i]
                    )
                # If gap > 1 year, do not connect

        plt.xlabel('Year')
        plt.ylabel('Annual Mass Balance (mm w.e.)')
        plt.title(f'Annual Mass Balance for each Elevation Bin over Time - {glacier}')
        plt.grid(True, alpha=0.8)
        plt.show()

# List of glaciers to analyze
glaciers = [
    "Rhonegletscher",
    "Grosser Aletschgletscher",
    "Silvrettagletscher",
    "Claridenfirn",
    "Griesgletscher",
    "Allalingletscher",
    "Schwarzberggletscher",
    "Hohlaubgletscher",
    "Glacier du Giétro",
]

#plot_mass_balance_for_glaciers(mass_balance_hy_eb_df, glaciers)



# Function to plot summer temperature with norms
def plot_summer_temperature(summer_df, city_name, norm6190, norm9120):
    plt.figure(figsize=(12, 6))
    plt.plot(summer_df['date'],
             summer_df['summer mean temperature (°C)'],
             label='Summer Mean Temperature (°C)',
             color='red')
    plt.axhline(y=norm6190,
                color='black',
                linestyle='--',
                label=f'1961-1990 Norm ({norm6190:.1f}°C)')
    plt.axhline(y=norm9120,
                color='black',
                label=f'1991-2020 Norm ({norm9120:.1f}°C)')
    plt.title(f'{city_name} Summer Mean Temperature')
    plt.xlabel('Year')
    plt.ylabel('Mean Temperature (°C)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# Function to plot winter precipitation with norm
def plot_winter_precipitation(winter_df, city_name, norm6190, norm9120):
    plt.figure(figsize=(12, 6))
    plt.plot(winter_df['date'],
             winter_df['winter total precipitation (mm)'],
             label='Winter Total Precipitation (mm)',
             color='blue')
    plt.axhline(y=norm6190,
                color='black',
                linestyle='--',
                label=f'1961-1990 Norm ({norm6190:.1f} mm)')
    plt.axhline(y=norm9120,
                color='black',
                label=f'1991-2020 Norm ({norm9120:.1f} mm)')
    plt.title(f'{city_name} Winter Total Precipitation')
    plt.xlabel('Year')
    plt.ylabel('Precipitation (mm)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()


# Plot for Altdorf
#plot_summer_temperature(altdorf_summer, 'Altdorf', altdorf_summer_norm_6190_t, altdorf_summer_norm_9120_t)
#plot_winter_precipitation(altdorf_winter, 'Altdorf', altdorf_winter_norm_6190_p, altdorf_winter_norm_9120_p)

# Plot for Davos
#plot_summer_temperature(davos_summer, 'Davos', davos_summer_norm_6190_t, davos_summer_norm_9120_t)
#plot_winter_precipitation(davos_winter, 'Davos', davos_winter_norm_6190_p, davos_winter_norm_9120_p)

# Plot for Sion
#plot_summer_temperature(sion_summer, 'Sion', sion_summer_norm_6190_t, sion_summer_norm_9120_t)
#plot_winter_precipitation(sion_winter, 'Sion', sion_winter_norm_6190_p, sion_winter_norm_9120_p)


def plot_mass_balance_weather(
    glacier_name,
    temp_data,
    temp_column,
    mass_balance_summer,
    mass_balance_winter=None,
    precip_data=None,
    precip_column=None,
):
    """
    Plot summer mass balance vs temperature and winter mass balance vs precipitation
    for a glacier.
    """
    plt.figure(figsize=(12, 6))
    plt.scatter(temp_data[temp_column], mass_balance_summer['summer mass balance (mm w.e.)'])
    plt.xlabel('Summer Mean Temperature (°C)')
    plt.ylabel('Summer Mass Balance (mm w.e.)')
    plt.title(f"{glacier_name} Summer Mass Balance with relation to Temperature")
    plt.grid(True)
    plt.show()

    if mass_balance_winter is not None and precip_data is not None:
        plt.figure(figsize=(12, 6))
        plt.scatter(precip_data[precip_column], mass_balance_winter['winter mass balance (mm w.e.)'])
        plt.xlabel('Winter Total Precipitation (mm)')
        plt.ylabel('Winter Mass Balance (mm w.e.)')
        plt.title(f"{glacier_name} Winter Mass Balance with relation to Precipitation")
        plt.grid(True)
        plt.show()

# Convert date strings to datetime objects for comparison
dates_to_exclude_summer = pd.to_datetime(['1994-05-01', '1995-05-01'])
dates_to_exclude_winter = pd.to_datetime(['1993-10-01', '1994-10-01'])

# Filter Claridenfirn summer and winter data
altdorf_summer_filtered = altdorf_summer[~altdorf_summer['date'].isin(dates_to_exclude_summer)]
altdorf_winter_filtered = altdorf_winter[~altdorf_winter['date'].isin(dates_to_exclude_winter)]

# Slice Sion data for specific glaciers
sion_summer_allalin = sion_summer.iloc[41:]  # Allalingletscher, Hohlaubgletscher, Schwarzberggletscher
sion_winter_allalin = sion_winter.iloc[41:]

sion_summer_gries = sion_summer.iloc[47:]  # Griesgletscher
sion_winter_gries = sion_winter.iloc[47:]

sion_summer_gietro = sion_summer.iloc[52:]  # Glacier du Giétro
sion_winter_gietro = sion_winter.iloc[52:]

# Assign weather data to glaciers
glaciers = {
    "Grosser Aletschgletscher": {
        "temp_data": sion_summer,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Grosser Aletschgletscher'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Grosser Aletschgletscher'
        ],
        "precip_data": sion_winter,
        "precip_column": 'winter total precipitation (mm)',
    },
    "Allalingletscher": {
        "temp_data": sion_summer_allalin,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Allalingletscher'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Allalingletscher'
        ],
        "precip_data": sion_winter_allalin,
        "precip_column": 'winter total precipitation (mm)',
    },
    "Hohlaubgletscher": {
        "temp_data": sion_summer_allalin,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Hohlaubgletscher'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Hohlaubgletscher'
        ],
        "precip_data": sion_winter_allalin,
        "precip_column": 'winter total precipitation (mm)',
    },
    "Schwarzberggletscher": {
        "temp_data": sion_summer_allalin,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Schwarzberggletscher'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Schwarzberggletscher'
        ],
        "precip_data": sion_winter_allalin,
        "precip_column": 'winter total precipitation (mm)',
    },
    "Griesgletscher": {
        "temp_data": sion_summer_gries,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Griesgletscher'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Griesgletscher'
        ],
        "precip_data": sion_winter_gries,
        "precip_column": 'winter total precipitation (mm)',
    },
    "Glacier du Giétro": {
        "temp_data": sion_summer_gietro,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Glacier du Giétro'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Glacier du Giétro'
        ],
        "precip_data": sion_winter_gietro,
        "precip_column": 'winter total precipitation (mm)',
    },
    "Silvrettagletscher": {
        "temp_data": davos_summer,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Silvrettagletscher'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Silvrettagletscher'
        ],
        "precip_data": davos_winter,
        "precip_column": 'winter total precipitation (mm)',
    },
    "Claridenfirn": {
        "temp_data": altdorf_summer_filtered,
        "temp_column": 'summer mean temperature (°C)',
        "mass_balance_summer": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Claridenfirn'
        ],
        "mass_balance_winter": mass_balance_hy_df[
            mass_balance_hy_df['glacier name'] == 'Claridenfirn'
        ],
        "precip_data": altdorf_winter_filtered,
        "precip_column": 'winter total precipitation (mm)',
    },
}

# Generate plots for each glacier
#for name, data in glaciers.items():
#    plot_mass_balance_weather(
#        glacier_name=name,
#        temp_data=data["temp_data"],
#        temp_column=data["temp_column"],
#        mass_balance_summer=data["mass_balance_summer"],
#        mass_balance_winter=data["mass_balance_winter"],
#        precip_data=data["precip_data"],
#        precip_column=data["precip_column"],
#    )



