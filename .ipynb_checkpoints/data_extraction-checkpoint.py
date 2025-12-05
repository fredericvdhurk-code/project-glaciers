import pandas as pd
import os
import numpy as np




# All links for data exctraction
url_glaciers_list = "https://doi.glamos.ch/data/glacier_list/glacier_list.csv"
url_length_change = "https://doi.glamos.ch/data/lengthchange/lengthchange.csv"
url_volume_change = "https://doi.glamos.ch/data/volumechange/volumechange.csv"
url_mass_balance = "https://doi.glamos.ch/data/massbalance/massbalance_observation.csv" #observation period
url_mass_balance2 = "https://doi.glamos.ch/data/massbalance/massbalance_observation_elevationbins.csv" #observation period
url_mass_balance3 = "https://doi.glamos.ch/data/massbalance/massbalance_fixdate.csv" #hydrological year
url_mass_balance4 = "https://doi.glamos.ch/data/massbalance/massbalance_fixdate_elevationbins.csv" #hydrological year
url_davos = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/dav/ogd-nbcn_dav_m.csv"
url_sion = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/sio/ogd-nbcn_sio_m.csv"
url_altdorf = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/alt/ogd-nbcn_alt_m.csv"
url_metadata = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/ogd-nbcn_meta_parameters.csv"

# Path to data folders
data_path = "project-glaciers/data"
raw_data_path = "project-glaciers/data/raw_data"
os.makedirs(data_path, exist_ok=True)
os.makedirs(raw_data_path, exist_ok=True)



# Get all raw datasets and store them in csv files

raw_list = pd.read_csv(url_glaciers_list, delimiter='\t')
raw_length = pd.read_csv(url_length_change, delimiter='\t')
raw_volume = pd.read_csv(url_volume_change, sep=r'\s+')
raw_mass_balance_op = pd.read_csv(url_mass_balance, delimiter='\t')
raw_mass_balance_op_eb = pd.read_csv(url_mass_balance2, delimiter='\t')
raw_mass_balance_hy = pd.read_csv(url_mass_balance3, delimiter='\t')
raw_mass_balance_hy_eb = pd.read_csv(url_mass_balance4, delimiter='\t')

raw_list.to_csv(os.path.join(raw_data_path, "glaciers_list_raw.csv"), index=False)
raw_length.to_csv(os.path.join(raw_data_path, "length_change_raw.csv"), index=False)
raw_volume.to_csv(os.path.join(raw_data_path, "volume_change_raw.csv"), index=False)
raw_mass_balance_op.to_csv(os.path.join(raw_data_path, "mass_balance_op_raw.csv"), index=False)
raw_mass_balance_op_eb.to_csv(os.path.join(raw_data_path, "mass_balance_op_eb_raw.csv"), index=False)
raw_mass_balance_hy.to_csv(os.path.join(raw_data_path, "mass_balance_hy_raw.csv"), index=False)
raw_mass_balance_hy_eb.to_csv(os.path.join(raw_data_path, "mass_balance_hy_eb_raw.csv"), index=False)



# Process the list CSV to get a clean list of all glaciers in a new CSV file.
# Get rid of all irrelevant information
initial_list_df = pd.read_csv(
    url_glaciers_list,
    delimiter='\t',
    skiprows=4
)
glaciers_list_df = initial_list_df.drop(index=[1, 2])  # Delete rows 1 & 2

# Split data into several columns to get clean information
glaciers_list_df = glaciers_list_df[
    'SWISS GLACIER LIST (AVAILABLE DATA)'
].str.split(',', expand=True)
glaciers_list_df = glaciers_list_df.reset_index(drop=True)

# Make first row appear as the column indices
new_headers_list = glaciers_list_df.iloc[0]
glaciers_list_df = glaciers_list_df[1:]
glaciers_list_df.columns = new_headers_list

# Rename columns to include measurement units
glaciers_list_df.rename(
    columns={
        'glacier area': 'glacier area (km2)',
        'survey year for glacier area': 'survey year for glacier area (yyyy)',
        'coordx': 'coordx (X_LV95)',
        'coordy': 'coordy (Y_LV95)'
    },
    inplace=True
)



# Process Length change
initial_length_df = pd.read_csv(
    url_length_change,
    delimiter='\t',
    skiprows=4  # Skip the first 4 rows
)
length_change_df = initial_length_df.drop(index=[1, 2])  # Delete rows 1 & 2

# Split data into several columns to get clean information
length_change_df = length_change_df[
    'SWISS GLACIER LENGTH CHANGE'
].str.split(',', expand=True)
length_change_df = length_change_df.reset_index(drop=True)

# Make first row appear as the column indices
new_headers_length = length_change_df.iloc[0]
length_change_df = length_change_df[1:]
length_change_df.columns = new_headers_length

# Rename all columns that contain numerical values to include the units
length_change_df.rename(
    columns={
        'start date of observation': 'start date of observation (yyyy-mm-dd)',
        'end date of observation': 'end date of observation (yyyy-mm-dd)',
        'length change': 'length change (m)',
        'elevation of glacier tongue': 'elevation of glacier tongue (m asl.)'
    },
    inplace=True
)



# Process volume change

initial_volume_df = pd.read_csv(url_volume_change, sep = r'\s+', engine = 'python')
volume_change_df = initial_volume_df.drop(index = [0,1,2,4])
volume_change_df = volume_change_df.reset_index(drop = True)
volume_change_df.columns = volume_change_df.iloc[0]
volume_change_df = volume_change_df.drop(0)
volume_change_df = volume_change_df.drop(columns = [';'])

volume_change_df['merged_15-26'] = (
    volume_change_df.iloc[:, 15:26]
    .astype(str)
    .apply(lambda x: ' '.join(x), axis=1)
)

volume_change_df = volume_change_df.drop(
    columns=[c for c in volume_change_df.columns
             if str(c) in ['None', 'NaN', 'Name'] or pd.isna(c)]
)

volume_change_df.rename(columns={'merged_15-26': 'Name'}, inplace=True)

# Remove all 'None' or NaN that appear in the last column because of the merger
volume_change_df['Name'] = (
    volume_change_df['Name']
    .str.replace('None', '', regex=False)
    .str.replace('nan', '', regex=False)
    .str.replace(' - ', ' -', regex=False)
    .str.strip()
)

cols = volume_change_df.columns.tolist()

# Move the last column to the front
cols = [cols[-1]] + cols[:-1]

# Reorder the DataFrame
volume_change_df = volume_change_df[cols]

# Rename all columns that contain numerical values to include the units
volume_change_df.rename(
    columns={
        'date_start': 'date_start (yyyymmdd)',
        'date_end': 'date_end (yyyymmdd)',
        'A_start': 'A_start (km2)',
        'outline_start': 'outline_start (yyyy)',
        'A_end': 'A_end (km2)',
        'outline_end': 'outline_end (yyyy)',
        'dV': 'dV (km3)',
        'dh_mean': 'dh_mean (m)',
        'Bgeod': 'Bgeod (mw.e.a-1)',
        'sigma': 'sigma (mw.e.)',
        'covered': 'covered (%)',
        'rho_dv': 'rho_dv (kgm-3)',
        'Name': 'glacier name'
    },
    inplace=True
)




# Mass balance observation period
initial_mass_balance_df = pd.read_csv(
    url_mass_balance,
    delimiter='\t',
    skiprows=4
)
mass_balance_df = initial_mass_balance_df.drop(index=[1, 2])

mass_balance_df = mass_balance_df[
    'SWISS GLACIER MASS BALANCE (OBSERVATION PERIOD)'
].str.split(',', expand=True)
mass_balance_df = mass_balance_df.reset_index(drop=True)

# Merge columns 13-16 into a single column
mass_balance_df['merged columns 13,14,15,16'] = (
    mass_balance_df[mass_balance_df.columns[13]].astype(str) + ' ' +
    mass_balance_df[mass_balance_df.columns[14]].astype(str) + ' ' +
    mass_balance_df[mass_balance_df.columns[15]].astype(str) + ' ' +
    mass_balance_df[mass_balance_df.columns[16]].astype(str)
)

# Drop the original columns 13-16
mass_balance_df = mass_balance_df.drop(
    columns=[
        mass_balance_df.columns[13],
        mass_balance_df.columns[14],
        mass_balance_df.columns[15],
        mass_balance_df.columns[16]
    ]
)


new_headers_mb_op = mass_balance_df.iloc[0]
mass_balance_df = mass_balance_df[1:]
mass_balance_df.columns = new_headers_mb_op

mass_balance_df = mass_balance_df.rename(columns={'observer None None None': 'observer'})

mass_balance_df['observer'] = (
    mass_balance_df['observer']
    .str.replace('None', '', regex=False)
    .str.replace(' - ', ' -', regex=False)
    .str.strip()
)

mass_balance_df.rename(
    columns={
        'start date of observation': 'start date of observation (yyyy-mm-dd)',
        'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)',
        'end date of observation': 'end date of observation (yyyy-mm-dd)',
        'winter mass balance': 'winter mass balance (mm w.e.)',
        'summer mass balance': 'summer mass balance (mm w.e.)',
        'annual mass balance': 'annual mass balance (mm w.e.)',
        'equilibrium line altitude': 'equilibrium line altitude (m asl.)',
        'accumulation area ratio': 'accumulation area ratio (%)',
        'glacier area': 'glacier area (km2)',
        'minimum elevation of glacier': 'minimum elevation of glacier (m asl.)',
        'maximum elevation of glacier': 'maximum elevation of glacier (m asl.)'
    },
    inplace=True
)




# Mass balance observation period with elevation bins

initial_mass_balance_eb_df = pd.read_csv(
    url_mass_balance2,
    delimiter='\t',
    skiprows=4
)

mass_balance_eb_df = initial_mass_balance_eb_df.drop(index=[1, 2])
mass_balance_eb_df = mass_balance_eb_df[
    'SWISS GLACIER MASS BALANCE (OBSERVATION PERIOD) ELEVATION BINS'
].str.split(',', expand=True)
mass_balance_eb_df = mass_balance_eb_df.reset_index(drop=True)

new_headers_mb_op_eb = mass_balance_eb_df.iloc[0]
mass_balance_eb_df = mass_balance_eb_df[1:]
mass_balance_eb_df.columns = new_headers_mb_op_eb

mass_balance_eb_df.rename(
    columns={
        'start date of observation': 'start date of observation (yyyy-mm-dd)',
        'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)',
        'end date of observation': 'end date of observation (yyyy-mm-dd)',
        'winter mass balance': 'winter mass balance (mm w.e.)',
        'summer mass balance': 'summer mass balance (mm w.e.)',
        'annual mass balance': 'annual mass balance (mm w.e.)',
        'area of elevation bin': 'area of elevation bin (km2)',
        'lower elevation of bin': 'lower elevation of bin (m asl.)',
        'upper elevation of bin': 'upper elevation of bin (m asl.)'
    },
    inplace=True
)



# Mass balance hydrological year

initial_mass_balance_hy_df = pd.read_csv(
    url_mass_balance3,
    delimiter='\t',
    skiprows=4
)

mass_balance_hy_df = initial_mass_balance_hy_df.drop(index=[1, 2])
mass_balance_hy_df = mass_balance_hy_df[
    'SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR)'
].str.split(',', expand=True)
mass_balance_hy_df = mass_balance_hy_df.reset_index(drop=True)

# Merging the last 4 columns into a new column
mass_balance_hy_df['merged columns 13,14,15,16'] = (
    mass_balance_hy_df[mass_balance_hy_df.columns[13]].astype(str) + ' ' +
    mass_balance_hy_df[mass_balance_hy_df.columns[14]].astype(str) + ' ' +
    mass_balance_hy_df[mass_balance_hy_df.columns[15]].astype(str) + ' ' +
    mass_balance_hy_df[mass_balance_hy_df.columns[16]].astype(str)
)

mass_balance_hy_df = mass_balance_hy_df.drop(
    columns=[
        mass_balance_hy_df.columns[13],
        mass_balance_hy_df.columns[14],
        mass_balance_hy_df.columns[15],
        mass_balance_hy_df.columns[16]
    ]
)

new_headers_mb_hy = mass_balance_hy_df.iloc[0]
mass_balance_hy_df = mass_balance_hy_df[1:]
mass_balance_hy_df.columns = new_headers_mb_hy

mass_balance_hy_df = mass_balance_hy_df.rename(
    columns={'observer None None None': 'observer'}
)

mass_balance_hy_df['observer'] = (
    mass_balance_hy_df['observer']
    .str.replace('None', '', regex=False)
    .str.replace(' - ', ' -', regex=False)
    .str.strip()
)

mass_balance_hy_df.rename(
    columns={
        'start date of observation': 'start date of observation (yyyy-mm-dd)',
        'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)',
        'end date of observation': 'end date of observation (yyyy-mm-dd)',
        'winter mass balance': 'winter mass balance (mm w.e.)',
        'summer mass balance': 'summer mass balance (mm w.e.)',
        'annual mass balance': 'annual mass balance (mm w.e.)',
        'equilibrium line altitude': 'equilibrium line altitude (m asl.)',
        'accumulation area ratio': 'accumulation area ratio (%)',
        'glacier area': 'glacier area (km2)',
        'minimum elevation of glacier': 'minimum elevation of glacier (m asl.)',
        'maximum elevation of glacier': 'maximum elevation of glacier (m asl.)'
    },
    inplace=True
)



# Mass balance hydrological year with elevation bins

initial_mass_balance_hy_eb_df = pd.read_csv(
    url_mass_balance4,
    delimiter='\t',
    skiprows=4
)

mass_balance_hy_eb_df = initial_mass_balance_hy_eb_df.drop(index=[1, 2])
mass_balance_hy_eb_df = mass_balance_hy_eb_df[
    'SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR) ELEVATION BINS'
].str.split(',', expand=True)
mass_balance_hy_eb_df = mass_balance_hy_eb_df.reset_index(drop=True)

# Merging the last 4 columns into a new column
mass_balance_hy_eb_df['merged columns 11,12,13,14'] = (
    mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[11]].astype(str) + ' ' +
    mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[12]].astype(str) + ' ' +
    mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[13]].astype(str) + ' ' +
    mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[14]].astype(str)
)

mass_balance_hy_eb_df = mass_balance_hy_eb_df.drop(
    columns=[
        mass_balance_hy_eb_df.columns[11],
        mass_balance_hy_eb_df.columns[12],
        mass_balance_hy_eb_df.columns[13],
        mass_balance_hy_eb_df.columns[14]
    ]
)

new_headers_mb_hy_eb = mass_balance_hy_eb_df.iloc[0]
mass_balance_hy_eb_df = mass_balance_hy_eb_df[1:]
mass_balance_hy_eb_df.columns = new_headers_mb_hy_eb

mass_balance_hy_eb_df = mass_balance_hy_eb_df.rename(
    columns={'observer None None None': 'observer'}
)

mass_balance_hy_eb_df['observer'] = (
    mass_balance_hy_eb_df['observer']
    .str.replace('None', '', regex=False)
    .str.replace(' - ', ' -', regex=False)
    .str.strip()
)

mass_balance_hy_eb_df.rename(
    columns={
        'start date of observation': 'start date of observation (yyyy-mm-dd)',
        'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)',
        'end date of observation': 'end date of observation (yyyy-mm-dd)',
        'winter mass balance': 'winter mass balance (mm w.e.)',
        'summer mass balance': 'summer mass balance (mm w.e.)',
        'annual mass balance': 'annual mass balance (mm w.e.)',
        'area of elevation bin': 'area of elevation bin (km2)',
        'lower elevation of bin': 'lower elevation of bin (m asl.)',
        'upper elevation of bin': 'upper elevation of bin (m asl.)'
    },
    inplace=True
)



# Create CSV files with the cleaned dataframes to work further
glaciers_list_df.to_csv(os.path.join(data_path, "glaciers_list.csv"), index=False)
length_change_df.to_csv(os.path.join(data_path, "length_change.csv"), index=False)
volume_change_df.to_csv(os.path.join(data_path, "volume_change.csv"), index=False)
mass_balance_df.to_csv(os.path.join(data_path, "mass_balance_op.csv"), index=False)
mass_balance_eb_df.to_csv(os.path.join(data_path, "mass_balance_op_eb.csv"), index=False)
mass_balance_hy_df.to_csv(os.path.join(data_path, "mass_balance_hy.csv"), index=False)
mass_balance_hy_eb_df.to_csv(os.path.join(data_path, "mass_balance_hy_eb.csv"), index=False)




# Weather data extraction

sion_weather_raw = pd.read_csv(url_sion, delimiter = ';')
davos_weather_raw = pd.read_csv(url_davos, delimiter = ';')
altdorf_weather_raw = pd.read_csv(url_altdorf, delimiter = ';')

sion_weather_raw.to_csv(os.path.join(raw_data_path, "weather_data_sion_raw.csv"), index = False)
davos_weather_raw.to_csv(os.path.join(raw_data_path, "weather_data_davos_raw.csv"), index = False)
altdorf_weather_raw.to_csv(os.path.join(raw_data_path, "weather_data_altdorf_raw.csv"), index = False)

# Get a CSV file for the metadata
try:
    metadata_raw = pd.read_csv(
        url_metadata,
        delimiter=';',
        encoding='latin1'
    )
except UnicodeDecodeError:
    try:
        metadata_raw = pd.read_csv(
            url_metadata,
            delimiter=';',
            encoding='cp1252'
        )
    except UnicodeDecodeError:
        try:
            metadata_raw = pd.read_csv(
                url_metadata,
                delimiter=';',
                encoding='utf-16'
            )
        except Exception as e:
            print(f"Failed to read file: {e}")

metadata_raw.to_csv(
    os.path.join(raw_data_path, "weather_metadata_raw.csv"),
    index=False
)

# Clean metadata
metadata = metadata_raw[['parameter_shortname',
                         'parameter_description_en',
                         'parameter_unit'
                    ]]


# Process monthly data to get a clean csv files

sion_weather = pd.read_csv(url_sion, delimiter = ';')
sion_weather['date'] = pd.to_datetime(
    sion_weather['reference_timestamp'],
    format='%d.%m.%Y %H:%M'
)

sion_weather = sion_weather.drop(columns = ['station_abbr'])
sion_weather = sion_weather.set_index('date')

davos_weather = pd.read_csv(url_davos, delimiter = ';')
davos_weather['date'] = pd.to_datetime(
    davos_weather['reference_timestamp'],
    format='%d.%m.%Y %H:%M'
)
davos_weather = davos_weather.drop(columns = ['station_abbr'])
davos_weather = davos_weather.set_index('date')

altdorf_weather = pd.read_csv(url_altdorf, delimiter = ';')
altdorf_weather['date'] = pd.to_datetime(
    altdorf_weather['reference_timestamp'],
    format='%d.%m.%Y %H:%M'
)
altdorf_weather = altdorf_weather.drop(columns = ['station_abbr'])
altdorf_weather = altdorf_weather.set_index('date')

# We only need data between 1914 and 2025
sion_1914 = sion_weather[
    (sion_weather.index >= '1914-10-01') &
    (sion_weather.index < '2025-10-01')
]
sion_1914 = sion_1914.dropna(axis=1)
sion_1914 = sion_1914.drop('reference_timestamp', axis=1)


davos_1914 = davos_weather[
    (davos_weather.index >= '1914-10-01') &
    (davos_weather.index < '2025-10-01')
]
davos_1914 = davos_1914.dropna(axis=1)
davos_1914 = davos_1914.drop('reference_timestamp', axis=1)

altdorf_1914 = altdorf_weather[
    (altdorf_weather.index >= '1914-10-01') &
    (altdorf_weather.index < '2025-10-01')
]
altdorf_1914 = altdorf_1914.dropna(axis=1)
altdorf_1914 = altdorf_1914.drop('reference_timestamp', axis=1)

sion_1914['year-month'] = sion_1914.index.to_series().dt.to_period('M')  # Format: YYYY-MM
sion_1914['year-month'] = sion_1914['year-month'].astype(str)
cols_s = sion_1914.columns.tolist()

# Remove 'year_month' from its current position and insert at the start
cols_s.insert(0, cols_s.pop(cols_s.index('year-month')))

# Reorder the DataFrame
sion_1914 = sion_1914[cols_s]

davos_1914['year-month'] = davos_1914.index.to_series().dt.to_period('M')  # Format: YYYY-MM
davos_1914['year-month'] = davos_1914['year-month'].astype(str)
cols_d = davos_1914.columns.tolist()

# Remove 'year_month' from its current position and insert at the start
cols_d.insert(0, cols_d.pop(cols_d.index('year-month')))

# Reorder the DataFrame
davos_1914 = davos_1914[cols_d]

altdorf_1914['year-month'] = altdorf_1914.index.to_series().dt.to_period('M')  # Format: YYYY-MM
altdorf_1914['year-month'] = altdorf_1914['year-month'].astype(str)
cols_a = altdorf_1914.columns.tolist()

# Remove 'year_month' from its current position and insert at the start
cols_a.insert(0, cols_a.pop(cols_a.index('year-month')))

# Reorder the DataFrame
altdorf_1914 = altdorf_1914[cols_a]

metadata = metadata[metadata['parameter_shortname'].isin(davos_1914.columns)]
metadata = metadata.reset_index(drop=True)
metadata.to_csv(os.path.join(data_path, "weather_metadata.csv"),
                    index=False)


davos = davos_1914[['year-month', 'rhs150m0', 'ths200m0']].copy()
davos.rename(
    columns={
        'rhs150m0': 'total precipitation (mm)',
        'ths200m0': 'mean temperature (°C)'
    },
    inplace=True
)

sion = sion_1914[['year-month', 'rhs150m0', 'ths200m0']].copy()
sion.rename(
    columns={
        'rhs150m0': 'total precipitation (mm)',
        'ths200m0': 'mean temperature (°C)'
    },
    inplace=True
)

altdorf = altdorf_1914[['year-month', 'rhs150m0', 'ths200m0']].copy()
altdorf.rename(
    columns={
        'rhs150m0': 'total precipitation (mm)',
        'ths200m0': 'mean temperature (°C)'
    },
    inplace=True
)

# Save the dataframes in csv files
davos.to_csv(os.path.join(data_path, "weather_data_davos_monthly.csv"), index=False)
sion.to_csv(os.path.join(data_path, "weather_data_sion_monthly.csv"), index=False)
altdorf.to_csv(os.path.join(data_path, "weather_data_altdorf_monthly.csv"), index=False)


# Get hy aggregates for further analysis

sion_p = sion[['total precipitation (mm)']].copy()
hy_data_sion_p = sion_p.groupby(np.arange(len(sion_p)) // 12).sum()

# Generate the date range for hydrological years (October to September)
start_date = '1914-10-01'
end_date = '2024-10-01'
hy_dates = pd.date_range(start = start_date, end = end_date, freq = 'YS-OCT')  # 'AS-OCT' = Annual, starting in October

# Assign the date range as the index
hy_data_sion_p.index = hy_dates


sion_temp = sion[['mean temperature (°C)']].copy()
sion_temp["days_in_month"] = sion_temp.index.days_in_month
sion_temp['temp_times_days'] = sion_temp['mean temperature (°C)'] * sion_temp['days_in_month']

n = 12  # Number of rows per group (12 months)
sion_temp = pd.DataFrame({
    'hy_mean_temperature_c': [
        sion_temp['temp_times_days'].iloc[i:i + n].sum() /
        sion_temp['days_in_month'].iloc[i:i + n].sum()
        for i in range(0, len(sion_temp), n)
    ]
})


sion_temp = sion_temp.round(1)
sion_temp.index = hy_dates


hy_data_sion = pd.concat([hy_data_sion_p, sion_temp], axis=1)
hy_data_sion = hy_data_sion.round(1)
hy_data_sion["date"] = hy_data_sion.index

hy_data_sion.rename(
    columns={'total precipitation (mm)': 'hy total precipitation (mm)'},
    inplace=True
)


# Get sums of precipitation for the hydrological years
davos_p = davos[['total precipitation (mm)']].copy()
hy_data_davos_p = davos_p.groupby(np.arange(len(davos_p)) // 12).sum()

# Generate the date range for hydrological years (October to September)
start_date = '1914-10-01'
end_date = '2024-10-01'
hy_dates = pd.date_range(start = start_date, end = end_date, freq = 'YS-OCT')  # 'AS-OCT' = Annual, starting in October

# Assign the date range as the index
hy_data_davos_p.index = hy_dates


# Compute average temperature for the hydrological years
davos_temp = davos[['mean temperature (°C)']].copy()
davos_temp["days_in_month"] = davos_temp.index.days_in_month
davos_temp['temp*days'] = davos_temp['mean temperature (°C)'] * davos_temp['days_in_month']

n = 12  # Number of rows per group (12 months)
davos_temp = pd.DataFrame({
    'hy mean temperature (°C)': [
        davos_temp['temp*days'].iloc[i:i+n].sum() / 
        davos_temp['days_in_month'].iloc[i:i+n].sum()
        for i in range(0, len(davos_temp), n)
    ]
})



davos_temp = davos_temp.round(1)
davos_temp.index = hy_dates




hy_data_davos = pd.concat([hy_data_davos_p, davos_temp], axis=1)
hy_data_davos = hy_data_davos.round(1)
hy_data_davos['date'] = hy_data_davos.index
hy_data_davos = hy_data_davos.reset_index(drop=True)
hy_data_davos.rename(columns = {'total precipitation (mm)': 'hy total precipitation (mm)'}, inplace=True)



altdorf_p = altdorf[['total precipitation (mm)']].copy()
hy_data_altdorf_p = altdorf_p.groupby(np.arange(len(altdorf_p)) // 12).sum()

# Generate the date range for hydrological years (October to September)
start_date = '1914-10-01'
end_date = '2024-10-01'
hy_dates = pd.date_range(start = start_date, end = end_date, freq = 'YS-OCT')  
# 'AS-OCT' = Annual, starting in October

# Assign the date range as the index
hy_data_altdorf_p.index = hy_dates


altdorf_temp = altdorf[['mean temperature (°C)']].copy()
altdorf_temp["days_in_month"] = altdorf_temp.index.days_in_month
altdorf_temp['temp*days'] = altdorf_temp['mean temperature (°C)'] * altdorf_temp['days_in_month']

n = 12  # Number of rows per group (12 months)
altdorf_temp = pd.DataFrame({
    'hy mean temperature (°C)': [
        altdorf_temp['temp*days'].iloc[i:i+n].sum() / 
        altdorf_temp['days_in_month'].iloc[i:i+n].sum()
        for i in range(0, len(altdorf_temp), n)
    ]
})

altdorf_temp = altdorf_temp.round(1)
altdorf_temp.index = hy_dates


hy_data_altdorf = pd.concat([hy_data_altdorf_p, altdorf_temp], axis=1)
hy_data_altdorf = hy_data_altdorf.round(1)
hy_data_altdorf["date"] = hy_data_altdorf.index
hy_data_altdorf.rename(columns = {'total precipitation (mm)': 'hy total precipitation (mm)'},
                       inplace=True)


hy_data_sion.to_csv(os.path.join(data_path, "weather_data_sion_hy.csv"), index=False)
hy_data_davos.to_csv(os.path.join(data_path, "weather_data_davos_hy.csv"), index = False)
hy_data_altdorf.to_csv(os.path.join(data_path, "weather_data_altdorf_hy.csv"), index = False)


# Get seasonal aggregates and store them in new csv files
# Define winter and summer months
winter_months = [10, 11, 12, 1, 2, 3, 4]   # Oct–Apr
summer_months = [5, 6, 7, 8, 9]            # May–Sep

def classify_season(date):
    if date.month in winter_months:
        return 'winter'
    else:
        return 'summer'

def season_year(date):
    # Winter belongs to the year of January (e.g. winter 1914-15 -> 1915)
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

# Sion
sion_seasonal = sion.copy()
sion_seasonal['season'] = sion_seasonal.index.to_series().apply(classify_season)
sion_seasonal['season_year'] = sion_seasonal.index.to_series().apply(season_year)

seasonal_sum = (
    sion_seasonal[['total precipitation (mm)', 'season', 'season_year']]
    .groupby(['season_year', 'season'])
    .sum()
)

sion_seasonal['days_in_month'] = sion_seasonal.index.days_in_month
sion_seasonal['temp_times_days'] = sion_seasonal['mean temperature (°C)'] * sion_seasonal['days_in_month']

seasonal_temp = (
    sion_seasonal.groupby(['season_year', 'season'])
    .apply(lambda df: df['temp_times_days'].sum() / df['days_in_month'].sum())
    .to_frame(name='mean seasonal temperature (°C)')
)

seasonal_data = pd.concat([seasonal_sum, seasonal_temp], axis=1)
seasonal_data = seasonal_data.round(1)

seasonal_data['date'] = seasonal_data.apply(compute_season_date, axis=1)
seasonal_data = seasonal_data.set_index('date').sort_index().reset_index()

weather_sion_winter = seasonal_data.iloc[::2, :].reset_index(drop=True)
weather_sion_summer = seasonal_data.iloc[1::2, :].reset_index(drop=True)

weather_sion_summer.rename(
    columns={
        'total precipitation (mm)': 'summer total precipitation (mm)',
        'mean seasonal temperature (°C)': 'summer mean temperature (°C)'
    },
    inplace=True
)

weather_sion_winter.rename(
    columns={
        'total precipitation (mm)': 'winter total precipitation (mm)',
        'mean seasonal temperature (°C)': 'winter mean temperature (°C)'
    },
    inplace=True
)

# Davos
davos_seasonal = davos.copy()
davos_seasonal['season'] = davos_seasonal.index.to_series().apply(classify_season)
davos_seasonal['season_year'] = davos_seasonal.index.to_series().apply(season_year)

seasonal_sum = (
    davos_seasonal[['total precipitation (mm)', 'season', 'season_year']]
    .groupby(['season_year', 'season'])
    .sum()
)

davos_seasonal['days_in_month'] = davos_seasonal.index.days_in_month
davos_seasonal['temp_times_days'] = davos_seasonal['mean temperature (°C)'] * davos_seasonal['days_in_month']

seasonal_temp_d = (
    davos_seasonal.groupby(['season_year', 'season'])
    .apply(lambda df: df['temp_times_days'].sum() / df['days_in_month'].sum())
    .to_frame(name='mean seasonal temperature (°C)')
)

seasonal_data_d = pd.concat([seasonal_sum, seasonal_temp_d], axis=1)
seasonal_data_d = seasonal_data_d.round(1)

seasonal_data_d['date'] = seasonal_data_d.apply(compute_season_date, axis=1)
seasonal_data_d = seasonal_data_d.set_index('date').sort_index().reset_index()

weather_davos_winter = seasonal_data_d.iloc[::2, :].reset_index(drop=True)
weather_davos_summer = seasonal_data_d.iloc[1::2, :].reset_index(drop=True)

weather_davos_summer.rename(
    columns={
        'total precipitation (mm)': 'summer total precipitation (mm)',
        'mean seasonal temperature (°C)': 'summer mean temperature (°C)'
    },
    inplace=True
)

weather_davos_winter.rename(
    columns={
        'total precipitation (mm)': 'winter total precipitation (mm)',
        'mean seasonal temperature (°C)': 'winter mean temperature (°C)'
    },
    inplace=True
)

# Altdorf
altdorf_seasonal = altdorf.copy()
altdorf_seasonal['season'] = altdorf_seasonal.index.to_series().apply(classify_season)
altdorf_seasonal['season_year'] = altdorf_seasonal.index.to_series().apply(season_year)

seasonal_sum = (
    altdorf_seasonal[['total precipitation (mm)', 'season', 'season_year']]
    .groupby(['season_year', 'season'])
    .sum()
)

altdorf_seasonal['days_in_month'] = altdorf_seasonal.index.days_in_month
altdorf_seasonal['temp_times_days'] = altdorf_seasonal['mean temperature (°C)'] * altdorf_seasonal['days_in_month']

seasonal_temp_a = (
    altdorf_seasonal.groupby(['season_year', 'season'])
    .apply(lambda df: df['temp_times_days'].sum() / df['days_in_month'].sum())
    .to_frame(name='mean seasonal temperature (°C)')
)

seasonal_data_a = pd.concat([seasonal_sum, seasonal_temp_a], axis=1)
seasonal_data_a = seasonal_data_a.round(1)

seasonal_data_a['date'] = seasonal_data_a.apply(compute_season_date, axis=1)
seasonal_data_a = seasonal_data_a.set_index('date').sort_index().reset_index()

weather_altdorf_winter = seasonal_data_a.iloc[::2, :].reset_index(drop=True)
weather_altdorf_summer = seasonal_data_a.iloc[1::2, :].reset_index(drop=True)

weather_altdorf_summer.rename(
    columns={
        'total precipitation (mm)': 'summer total precipitation (mm)',
        'mean seasonal temperature (°C)': 'summer mean temperature (°C)'
    },
    inplace=True
)

weather_altdorf_winter.rename(
    columns={
        'total precipitation (mm)': 'winter total precipitation (mm)',
        'mean seasonal temperature (°C)': 'winter mean temperature (°C)'
    },
    inplace=True
)



weather_sion_summer.to_csv(os.path.join(data_path, "weather_data_sion_summer.csv"), index=False)
weather_sion_winter.to_csv(os.path.join(data_path, "weather_data_sion_winter.csv"), index=False)

weather_davos_summer.to_csv(os.path.join(data_path, "weather_data_davos_summer.csv"), index=False)
weather_davos_winter.to_csv(os.path.join(data_path, "weather_data_davos_winter.csv"), index=False)

weather_altdorf_summer.to_csv(os.path.join(data_path, "weather_data_altdorf_summer.csv"), index=False)
weather_altdorf_winter.to_csv(os.path.join(data_path, "weather_data_altdorf_winter.csv"), index=False)



# Get weather deviations from norm 1961-1990 and store data in new csv files and pivot dataframes
davos = pd.read_csv('project-glaciers/data/raw_data/weather_data_davos_raw.csv')
sion = pd.read_csv('project-glaciers/data/raw_data/weather_data_sion_raw.csv')
altdorf = pd.read_csv('project-glaciers/data/raw_data/weather_data_altdorf_raw.csv')


sion['date'] = pd.to_datetime(
    sion['reference_timestamp'],
    format = '%d.%m.%Y %H:%M'
)

sion = sion.set_index('date')

sion = sion[
    (sion.index >= '1914-10-01') &
    (sion.index < '2025-10-01')
]


davos['date'] = pd.to_datetime(
    davos['reference_timestamp'],
    format = '%d.%m.%Y %H:%M'
)
davos = davos.set_index('date')

davos = davos[
    (davos.index >= '1914-10-01') &
    (davos.index < '2025-10-01')
]


altdorf['date'] = pd.to_datetime(
    altdorf['reference_timestamp'],
    format = '%d.%m.%Y %H:%M'
)
altdorf = altdorf.set_index('date')

altdorf = altdorf[
    (altdorf.index >= '1914-10-01') &
    (altdorf.index < '2025-10-01')
]



sion_t = sion[['ths200m0']]
davos_t = davos[['ths200m0']]
altdorf_t = altdorf[['ths200m0']]


# Temperature Deviations for Sion

sion_t = sion_t.reset_index(drop=True)

# Step 1: Add a helper column to group every 12 rows
sion_t['year'] = sion_t.index // 12

# Step 2: Add a helper column for the position within each group
sion_t['month'] = sion_t.index % 12

# Step 3: Pivot the DataFrame
sion_monthly_temp_dev = sion_t.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'ths200m0')


# Define the new column names
monthly_observations_t = [
    'october_td', 'november_td', 'december_td',
    'january_td', 'february_td', 'march_td',
    'april_td', 'may_td', 'june_td',
    'july_td', 'august_td', 'september_td'
]

# Rename the columns
sion_monthly_temp_dev.columns = monthly_observations_t

# Create the new column with the start date of the hydrological year
sion_monthly_temp_dev['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(sion_monthly_temp_dev))
]

sion_monthly_temp_dev = sion_monthly_temp_dev.reset_index(drop=True)

sion_monthly_temp_dev[['october_td']] = sion_monthly_temp_dev[['october_td']] - 9.5
sion_monthly_temp_dev[['november_td']] = sion_monthly_temp_dev[['november_td']] - 3.4
sion_monthly_temp_dev[['december_td']] = sion_monthly_temp_dev[['december_td']] + 0.4
sion_monthly_temp_dev[['january_td']] = sion_monthly_temp_dev[['january_td']] + 0.8
sion_monthly_temp_dev[['february_td']] = sion_monthly_temp_dev[['february_td']] - 1.6
sion_monthly_temp_dev[['march_td']] = sion_monthly_temp_dev[['march_td']] - 5.3
sion_monthly_temp_dev[['april_td']] = sion_monthly_temp_dev[['april_td']] - 9.4
sion_monthly_temp_dev[['may_td']] = sion_monthly_temp_dev[['may_td']] - 13.7
sion_monthly_temp_dev[['june_td']] = sion_monthly_temp_dev[['june_td']] - 17.0
sion_monthly_temp_dev[['july_td']] = sion_monthly_temp_dev[['july_td']] - 19.1
sion_monthly_temp_dev[['august_td']] = sion_monthly_temp_dev[['august_td']] - 17.9
sion_monthly_temp_dev[['september_td']] = sion_monthly_temp_dev[['september_td']] - 14.6


# Temperature deviations for Davos

davos_t = davos_t.reset_index(drop=True)

# Step 1: Add a helper column to group every 12 rows
davos_t['year'] = davos_t.index // 12

# Step 2: Add a helper column for the position within each group
davos_t['month'] = davos_t.index % 12

# Step 3: Pivot the DataFrame
davos_monthly_temp_dev = davos_t.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'ths200m0')



# Rename the columns
davos_monthly_temp_dev.columns = monthly_observations_t

# Create the new column with the start date of the hydrological year
davos_monthly_temp_dev['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(davos_monthly_temp_dev))
]

davos_monthly_temp_dev = davos_monthly_temp_dev.reset_index(drop=True)

davos_monthly_temp_dev[['october_td']] = davos_monthly_temp_dev[['october_td']] - 4.7
davos_monthly_temp_dev[['november_td']] = davos_monthly_temp_dev[['november_td']] + 1.0
davos_monthly_temp_dev[['december_td']] = davos_monthly_temp_dev[['december_td']] + 4.4
davos_monthly_temp_dev[['january_td']] = davos_monthly_temp_dev[['january_td']] + 5.3
davos_monthly_temp_dev[['february_td']] = davos_monthly_temp_dev[['february_td']] + 4.7
davos_monthly_temp_dev[['march_td']] = davos_monthly_temp_dev[['march_td']] + 2.2
davos_monthly_temp_dev[['april_td']] = davos_monthly_temp_dev[['april_td']] - 1.3
davos_monthly_temp_dev[['may_td']] = davos_monthly_temp_dev[['may_td']] - 5.9
davos_monthly_temp_dev[['june_td']] = davos_monthly_temp_dev[['june_td']] - 9.0
davos_monthly_temp_dev[['july_td']] = davos_monthly_temp_dev[['july_td']] - 11.3
davos_monthly_temp_dev[['august_td']] = davos_monthly_temp_dev[['august_td']] - 10.8
davos_monthly_temp_dev[['september_td']] = davos_monthly_temp_dev[['september_td']] - 8.3


# Temperature deviations for Altdorf

altdorf_t = altdorf_t.reset_index(drop=True)

# Step 1: Add a helper column to group every 12 rows
altdorf_t['year'] = altdorf_t.index // 12

# Step 2: Add a helper column for the position within each group
altdorf_t['month'] = altdorf_t.index % 12

# Step 3: Pivot the DataFrame
altdorf_monthly_temp_dev = altdorf_t.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'ths200m0')



# Rename the columns
altdorf_monthly_temp_dev.columns = monthly_observations_t

# Create the new column with the start date of the hydrological year
altdorf_monthly_temp_dev['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(altdorf_monthly_temp_dev))
]

altdorf_monthly_temp_dev = altdorf_monthly_temp_dev.reset_index(drop=True)

altdorf_monthly_temp_dev[['october_td']] = altdorf_monthly_temp_dev[['october_td']] - 9.8
altdorf_monthly_temp_dev[['november_td']] = altdorf_monthly_temp_dev[['november_td']] - 4.6
altdorf_monthly_temp_dev[['december_td']] = altdorf_monthly_temp_dev[['december_td']] - 1.0
altdorf_monthly_temp_dev[['january_td']] = altdorf_monthly_temp_dev[['january_td']] - 0.5
altdorf_monthly_temp_dev[['february_td']] = altdorf_monthly_temp_dev[['february_td']] -1.9
altdorf_monthly_temp_dev[['march_td']] = altdorf_monthly_temp_dev[['march_td']] -4.6
altdorf_monthly_temp_dev[['april_td']] = altdorf_monthly_temp_dev[['april_td']] - 8.2
altdorf_monthly_temp_dev[['may_td']] = altdorf_monthly_temp_dev[['may_td']] - 12.5
altdorf_monthly_temp_dev[['june_td']] = altdorf_monthly_temp_dev[['june_td']] - 15.5
altdorf_monthly_temp_dev[['july_td']] = altdorf_monthly_temp_dev[['july_td']] - 17.6
altdorf_monthly_temp_dev[['august_td']] = altdorf_monthly_temp_dev[['august_td']] - 16.8
altdorf_monthly_temp_dev[['september_td']] = altdorf_monthly_temp_dev[['september_td']] - 14.0




# Precipitation deviation for Sion

sion_precipitation = sion[['rhs150m0']].copy()
sion_precipitation = sion_precipitation.reset_index(drop = True)

# Step 1: Add a helper column to group every 12 rows
sion_precipitation['year'] = sion_precipitation.index // 12

# Step 2: Add a helper column for the position within each group
sion_precipitation['month'] = sion_precipitation.index % 12

# Step 3: Pivot the DataFrame
sion_precipitation = sion_precipitation.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'rhs150m0')

# Define the new column names
monthly_observations_p = [
    'october_pd', 'november_pd', 'december_pd',
    'january_pd', 'february_pd', 'march_pd',
    'april_pd', 'may_pd', 'june_pd',
    'july_pd', 'august_pd', 'september_pd'
]
# Rename the columns
sion_precipitation.columns = monthly_observations_p

# Create the new column with the start date of the hydrological year
sion_precipitation['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(sion_precipitation))
]

sion_precipitation = sion_precipitation.reset_index(drop=True)


sion_precipitation[['october_pd']] = sion_precipitation[['october_pd']] - 50
sion_precipitation[['november_pd']] = sion_precipitation[['november_pd']] - 60
sion_precipitation[['december_pd']] = sion_precipitation[['december_pd']] - 61
sion_precipitation[['january_pd']] = sion_precipitation[['january_pd']] - 53
sion_precipitation[['february_pd']] = sion_precipitation[['february_pd']] - 57
sion_precipitation[['march_pd']] = sion_precipitation[['march_pd']] - 48
sion_precipitation[['april_pd']] = sion_precipitation[['april_pd']] - 36
sion_precipitation[['may_pd']] = sion_precipitation[['may_pd']] - 41
sion_precipitation[['june_pd']] = sion_precipitation[['june_pd']] - 52
sion_precipitation[['july_pd']] = sion_precipitation[['july_pd']] - 48
sion_precipitation[['august_pd']] = sion_precipitation[['august_pd']] - 55
sion_precipitation[['september_pd']] = sion_precipitation[['september_pd']] - 38



# Precipitation deviations for Davos

davos_precipitation = davos[['rhs150m0']].copy()
davos_precipitation = davos_precipitation.reset_index()

# Step 1: Add a helper column to group every 12 rows
davos_precipitation['year'] = davos_precipitation.index // 12

# Step 2: Add a helper column for the position within each group
davos_precipitation['month'] = davos_precipitation.index % 12

# Step 3: Pivot the DataFrame
davos_precipitation = davos_precipitation.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'rhs150m0')

# Rename the columns
davos_precipitation.columns = monthly_observations_p

# Create the new column with the start date of the hydrological year
davos_precipitation['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(davos_precipitation))
]

davos_precipitation = davos_precipitation.reset_index(drop=True)

davos_precipitation[['october_pd']] = davos_precipitation[['october_pd']] - 58
davos_precipitation[['november_pd']] = davos_precipitation[['november_pd']] - 66
davos_precipitation[['december_pd']] = davos_precipitation[['december_pd']] - 65
davos_precipitation[['january_pd']] = davos_precipitation[['january_pd']] - 68
davos_precipitation[['february_pd']] = davos_precipitation[['february_pd']] - 59
davos_precipitation[['march_pd']] = davos_precipitation[['march_pd']] - 60
davos_precipitation[['april_pd']] = davos_precipitation[['april_pd']] - 55
davos_precipitation[['may_pd']] = davos_precipitation[['may_pd']] - 91
davos_precipitation[['june_pd']] = davos_precipitation[['june_pd']] - 120
davos_precipitation[['july_pd']] = davos_precipitation[['july_pd']] - 132
davos_precipitation[['august_pd']] = davos_precipitation[['august_pd']] - 135
davos_precipitation[['september_pd']] = davos_precipitation[['september_pd']] - 90



# Precipitation deviations for Altdorf

altdorf_precipitation = altdorf[['rhs150m0']].copy()
altdorf_precipitation = altdorf_precipitation.reset_index()

# Step 1: Add a helper column to group every 12 rows
altdorf_precipitation['year'] = altdorf_precipitation.index // 12

# Step 2: Add a helper column for the position within each group
altdorf_precipitation['month'] = altdorf_precipitation.index % 12

# Step 3: Pivot the DataFrame
altdorf_precipitation = altdorf_precipitation.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'rhs150m0')

# Rename the columns
altdorf_precipitation.columns = monthly_observations_p

# Create the new column with the start date of the hydrological year
altdorf_precipitation['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(altdorf_precipitation))
]

altdorf_precipitation = altdorf_precipitation.reset_index(drop=True)

altdorf_precipitation[['october_pd']] = altdorf_precipitation[['october_pd']] - 73
altdorf_precipitation[['november_pd']] = altdorf_precipitation[['november_pd']] - 81
altdorf_precipitation[['december_pd']] = altdorf_precipitation[['december_pd']] - 75
altdorf_precipitation[['january_pd']] = altdorf_precipitation[['january_pd']] - 71
altdorf_precipitation[['february_pd']] = altdorf_precipitation[['february_pd']] - 68
altdorf_precipitation[['march_pd']] = altdorf_precipitation[['march_pd']] - 71
altdorf_precipitation[['april_pd']] = altdorf_precipitation[['april_pd']] - 86
altdorf_precipitation[['may_pd']] = altdorf_precipitation[['may_pd']] - 99
altdorf_precipitation[['june_pd']] = altdorf_precipitation[['june_pd']] - 130
altdorf_precipitation[['july_pd']] = altdorf_precipitation[['july_pd']] - 132
altdorf_precipitation[['august_pd']] = altdorf_precipitation[['august_pd']] - 133
altdorf_precipitation[['september_pd']] = altdorf_precipitation[['september_pd']] - 86

sion_precipitation = sion_precipitation.round(1)
davos_precipitation = davos_precipitation.round(1)
altdorf_precipitation = altdorf_precipitation.round(1)



# Create new precipitation columns
for df_name in ['altdorf_precipitation',
                    'sion_precipitation',
                    'davos_precipitation']:
    df = globals()[df_name]
    df['opt_season_pd'] = df[
        ['october_pd',
             'november_pd', 'december_pd',
             'january_pd', 'february_pd']
    ].sum(axis=1)

    df['opt_season+march_pd'] = df[
        ['october_pd', 'november_pd', 'december_pd',
         'january_pd', 'february_pd', 'march_pd']
    ].sum(axis=1)

    df['winter_pd'] = df[
        ['october_pd', 'november_pd', 'december_pd',
         'january_pd', 'february_pd', 'march_pd', 'april_pd']
    ].sum(axis=1)


# Create new temperature deviation columns for all temperature dataframes
for df_name in ['sion_monthly_temp_dev',
                    'davos_monthly_temp_dev',
                    'altdorf_monthly_temp_dev']:
    df = globals()[df_name]

    # Calculate seasonal_td (weighted average of May-August)
    # May (31), June (30), July (31), August (31) = total 123 days
    df['opt_season_td'] = (
        df['may_td'] * 31 +
        df['june_td'] * 30 +
        df['july_td'] * 31 +
        df['august_td'] * 31
    ) / 123

    # Calculate summer_td (weighted average of May-September)
    # May (31), June (30), July (31), August (31), September (30) = total 153 days
    df['summer_td'] = (
        df['may_td'] * 31 +
        df['june_td'] * 30 +
        df['july_td'] * 31 +
        df['august_td'] * 31 +
        df['september_td'] * 30
    ) / 153

sion_monthly_temp_dev.to_csv(os.path.join(data_path, "weather_dev6190_sion_temp.csv"), index=False)
davos_monthly_temp_dev.to_csv(os.path.join(data_path, "weather_dev6190_davos_temp.csv"), index=False)
altdorf_monthly_temp_dev.to_csv(os.path.join(data_path, "weather_dev6190_altdorf_temp.csv"), index=False)

sion_precipitation.to_csv(os.path.join(data_path, "weather_dev6190_sion_prec.csv"), index=False)
davos_precipitation.to_csv(os.path.join(data_path, "weather_dev6190_davos_prec.csv"), index=False)
altdorf_precipitation.to_csv(os.path.join(data_path, "weather_dev6190_altdorf_prec.csv"), index=False)


# Same logic for 1991-2020 norms
davos = pd.read_csv('project-glaciers/data/raw_data/weather_data_davos_raw.csv')
sion = pd.read_csv('project-glaciers/data/raw_data/weather_data_sion_raw.csv')
altdorf = pd.read_csv('project-glaciers/data/raw_data/weather_data_altdorf_raw.csv')


sion['date'] = pd.to_datetime(
    sion['reference_timestamp'],
    format = '%d.%m.%Y %H:%M'
)

sion = sion.set_index('date')

sion = sion[
    (sion.index >= '1914-10-01') &
    (sion.index < '2025-10-01')
]


davos['date'] = pd.to_datetime(
    davos['reference_timestamp'],
    format = '%d.%m.%Y %H:%M'
)
davos = davos.set_index('date')

davos = davos[
    (davos.index >= '1914-10-01') &
    (davos.index < '2025-10-01')
]


altdorf['date'] = pd.to_datetime(
    altdorf['reference_timestamp'],
    format = '%d.%m.%Y %H:%M'
)
altdorf = altdorf.set_index('date')

altdorf = altdorf[
    (altdorf.index >= '1914-10-01') &
    (altdorf.index < '2025-10-01')
]



sion_dev_t = sion[['th9120mv']]
davos_dev_t = davos[['th9120mv']]
altdorf_dev_t = altdorf[['th9120mv']]


# Temperature Deviation for Sion
sion_dev_t = sion_dev_t.reset_index(drop=True)

# Step 1: Add a helper column to group every 12 rows
sion_dev_t['year'] = sion_dev_t.index // 12

# Step 2: Add a helper column for the position within each group
sion_dev_t['month'] = sion_dev_t.index % 12

# Step 3: Pivot the DataFrame
sion_monthly_temp_dev = sion_dev_t.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'th9120mv')


# Define the new column names
monthly_observations = [
    'october_td', 'november_td', 'december_td',
    'january_td', 'february_td', 'march_td',
    'april_td', 'may_td', 'june_td',
    'july_td', 'august_td', 'september_td'
]

# Rename the columns
sion_monthly_temp_dev.columns = monthly_observations

# Create the new column with the start date of the hydrological year
sion_monthly_temp_dev['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(sion_monthly_temp_dev))
]

sion_monthly_temp_dev = sion_monthly_temp_dev.reset_index(drop=True)

sion_monthly_temp_dev


# Temperature deviation for Davos

davos_dev_t = davos_dev_t.reset_index(drop=True)

# Step 1: Add a helper column to group every 12 rows
davos_dev_t['year'] = davos_dev_t.index // 12

# Step 2: Add a helper column for the position within each group
davos_dev_t['month'] = davos_dev_t.index % 12

# Step 3: Pivot the DataFrame
davos_monthly_temp_dev = davos_dev_t.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'th9120mv')



# Rename the columns
davos_monthly_temp_dev.columns = monthly_observations

# Create the new column with the start date of the hydrological year
davos_monthly_temp_dev['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(davos_monthly_temp_dev))
]

davos_monthly_temp_dev = davos_monthly_temp_dev.reset_index(drop=True)



# Temperature deviation for Altdorf

altdorf_dev_t = altdorf_dev_t.reset_index(drop=True)

# Step 1: Add a helper column to group every 12 rows
altdorf_dev_t['year'] = altdorf_dev_t.index // 12

# Step 2: Add a helper column for the position within each group
altdorf_dev_t['month'] = altdorf_dev_t.index % 12

# Step 3: Pivot the DataFrame
altdorf_monthly_temp_dev = altdorf_dev_t.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'th9120mv')



# Rename the columns
altdorf_monthly_temp_dev.columns = monthly_observations

# Create the new column with the start date of the hydrological year
altdorf_monthly_temp_dev['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(altdorf_monthly_temp_dev))
]

altdorf_monthly_temp_dev = altdorf_monthly_temp_dev.reset_index(drop=True)


# Precipitation deviation for Sion

sion_precipitation = sion[['rhs150m0']].copy()
sion_precipitation = sion_precipitation.reset_index(drop = True)

# Step 1: Add a helper column to group every 12 rows
sion_precipitation['year'] = sion_precipitation.index // 12

# Step 2: Add a helper column for the position within each group
sion_precipitation['month'] = sion_precipitation.index % 12

# Step 3: Pivot the DataFrame
sion_precipitation = sion_precipitation.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'rhs150m0')

# Define the new column names
monthly_observations_pd = [
    'october_pd', 'november_pd', 'december_pd',
    'january_pd', 'february_pd', 'march_pd',
    'april_pd', 'may_pd', 'june_pd',
    'july_pd', 'august_pd', 'september_pd'
]
# Rename the columns
sion_precipitation.columns = monthly_observations_pd

# Create the new column with the start date of the hydrological year
sion_precipitation['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(sion_precipitation))
]

sion_precipitation = sion_precipitation.reset_index(drop=True)


sion_precipitation[['october_pd']] = sion_precipitation[['october_pd']] - 43
sion_precipitation[['november_pd']] = sion_precipitation[['november_pd']] - 50
sion_precipitation[['december_pd']] = sion_precipitation[['december_pd']] - 68
sion_precipitation[['january_pd']] = sion_precipitation[['january_pd']] - 52
sion_precipitation[['february_pd']] = sion_precipitation[['february_pd']] - 40
sion_precipitation[['march_pd']] = sion_precipitation[['march_pd']] - 37
sion_precipitation[['april_pd']] = sion_precipitation[['april_pd']] - 34
sion_precipitation[['may_pd']] = sion_precipitation[['may_pd']] - 52
sion_precipitation[['june_pd']] = sion_precipitation[['june_pd']] - 48
sion_precipitation[['july_pd']] = sion_precipitation[['july_pd']] - 62
sion_precipitation[['august_pd']] = sion_precipitation[['august_pd']] - 60
sion_precipitation[['september_pd']] = sion_precipitation[['september_pd']] - 38



# Precipitation deviation for Davos

davos_precipitation = davos[['rhs150m0']].copy()
davos_precipitation = davos_precipitation.reset_index()

# Step 1: Add a helper column to group every 12 rows
davos_precipitation['year'] = davos_precipitation.index // 12

# Step 2: Add a helper column for the position within each group
davos_precipitation['month'] = davos_precipitation.index % 12

# Step 3: Pivot the DataFrame
davos_precipitation = davos_precipitation.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'rhs150m0')

# Rename the columns
davos_precipitation.columns = monthly_observations_pd

# Create the new column with the start date of the hydrological year
davos_precipitation['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(davos_precipitation))
]

davos_precipitation = davos_precipitation.reset_index(drop=True)

davos_precipitation[['october_pd']] = davos_precipitation[['october_pd']] - 77
davos_precipitation[['november_pd']] = davos_precipitation[['november_pd']] - 71
davos_precipitation[['december_pd']] = davos_precipitation[['december_pd']] - 68
davos_precipitation[['january_pd']] = davos_precipitation[['january_pd']] - 70
davos_precipitation[['february_pd']] = davos_precipitation[['february_pd']] - 52
davos_precipitation[['march_pd']] = davos_precipitation[['march_pd']] - 57
davos_precipitation[['april_pd']] = davos_precipitation[['april_pd']] - 54
davos_precipitation[['may_pd']] = davos_precipitation[['may_pd']] - 89
davos_precipitation[['june_pd']] = davos_precipitation[['june_pd']] - 129
davos_precipitation[['july_pd']] = davos_precipitation[['july_pd']] - 133
davos_precipitation[['august_pd']] = davos_precipitation[['august_pd']] - 150
davos_precipitation[['september_pd']] = davos_precipitation[['september_pd']] - 96



# Precipitation deviation for Altdorf

altdorf_precipitation = altdorf[['rhs150m0']].copy()
altdorf_precipitation = altdorf_precipitation.reset_index()

# Step 1: Add a helper column to group every 12 rows
altdorf_precipitation['year'] = altdorf_precipitation.index // 12

# Step 2: Add a helper column for the position within each group
altdorf_precipitation['month'] = altdorf_precipitation.index % 12

# Step 3: Pivot the DataFrame
altdorf_precipitation = altdorf_precipitation.pivot(index = 'year',
                                   columns = 'month',
                                   values = 'rhs150m0')

# Rename the columns
altdorf_precipitation.columns = monthly_observations_pd

# Create the new column with the start date of the hydrological year
altdorf_precipitation['hydrological year'] = [
    f"{1914 + i}-{1915 + i}" for i in range(len(altdorf_precipitation))
]

altdorf_precipitation = altdorf_precipitation.reset_index(drop=True)

altdorf_precipitation[['october_pd']] = altdorf_precipitation[['october_pd']] - 84
altdorf_precipitation[['november_pd']] = altdorf_precipitation[['november_pd']] - 81
altdorf_precipitation[['december_pd']] = altdorf_precipitation[['december_pd']] - 83
altdorf_precipitation[['january_pd']] = altdorf_precipitation[['january_pd']] - 70
altdorf_precipitation[['february_pd']] = altdorf_precipitation[['february_pd']] - 59
altdorf_precipitation[['march_pd']] = altdorf_precipitation[['march_pd']] - 72
altdorf_precipitation[['april_pd']] = altdorf_precipitation[['april_pd']] - 81
altdorf_precipitation[['may_pd']] = altdorf_precipitation[['may_pd']] - 117
altdorf_precipitation[['june_pd']] = altdorf_precipitation[['june_pd']] - 138
altdorf_precipitation[['july_pd']] = altdorf_precipitation[['july_pd']] - 141
altdorf_precipitation[['august_pd']] = altdorf_precipitation[['august_pd']] - 154
altdorf_precipitation[['september_pd']] = altdorf_precipitation[['september_pd']] - 105

sion_precipitation = sion_precipitation.round(1)
davos_precipitation = davos_precipitation.round(1)
altdorf_precipitation = altdorf_precipitation.round(1)



# Create new precipitation columns
for df_name in ['altdorf_precipitation',
                    'sion_precipitation',
                    'davos_precipitation']:
    df = globals()[df_name]
    df['opt_season_pd'] = df[['october_pd',
                                  'november_pd', 'december_pd',
                                  'january_pd', 'february_pd']].sum(axis=1)
    df['opt_season+march_pd'] = df[['october_pd', 'november_pd',
                                        'december_pd', 'january_pd',
                                        'february_pd','march_pd']].sum(axis=1)
    df['winter_pd'] = df[['october_pd',
                              'november_pd', 'december_pd',
                              'january_pd', 'february_pd',
                              'march_pd', 'april_pd']].sum(axis=1)


# Create new temperature deviation columns for all temperature dataframes
for df_name in ['sion_monthly_temp_dev',
                    'davos_monthly_temp_dev',
                    'altdorf_monthly_temp_dev']:
    df = globals()[df_name]

    # Calculate seasonal_td (weighted average of May-August)
    # May (31), June (30), July (31), August (31) = total 123 days
    df['opt_season_td'] = (
        df['may_td'] * 31 +
        df['june_td'] * 30 +
        df['july_td'] * 31 +
        df['august_td'] * 31
    ) / 123

    # Calculate summer_td (weighted average of May-September)
    # May (31), June (30), July (31), August (31), September (30) = total 153 days
    df['summer_td'] = (
        df['may_td'] * 31 +
        df['june_td'] * 30 +
        df['july_td'] * 31 +
        df['august_td'] * 31 +
        df['september_td'] * 30
    ) / 153


sion_monthly_temp_dev.to_csv(os.path.join(data_path, "weather_dev9120_sion_temp.csv"), index=False)
davos_monthly_temp_dev.to_csv(os.path.join(data_path, "weather_dev9120_davos_temp.csv"), index=False)
altdorf_monthly_temp_dev.to_csv(os.path.join(data_path, "weather_dev9120_altdorf_temp.csv"), index=False)

sion_precipitation.to_csv(os.path.join(data_path, "weather_dev9120_sion_prec.csv"), index=False)
davos_precipitation.to_csv(os.path.join(data_path, "weather_dev9120_davos_prec.csv"), index=False)
altdorf_precipitation.to_csv(os.path.join(data_path, "weather_dev9120_altdorf_prec.csv"), index=False)


