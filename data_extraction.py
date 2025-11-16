import pandas as pd
import os

url_complete_list = "https://doi.glamos.ch/data/glacier_list/glacier_list.csv"
url_length_change = "https://doi.glamos.ch/data/lengthchange/lengthchange.csv"
url_volume_change = "https://doi.glamos.ch/data/volumechange/volumechange.csv"
url_mass_balance = "https://doi.glamos.ch/data/massbalance/massbalance_observation.csv" # observation period
url_mass_balance2 = "https://doi.glamos.ch/data/massbalance/massbalance_observation_elevationbins.csv" # observation period
url_mass_balance3 = "https://doi.glamos.ch/data/massbalance/massbalance_fixdate.csv" # hydrological year
url_mass_balance4 = "https://doi.glamos.ch/data/massbalance/massbalance_fixdate_elevationbins.csv" # hydrological year

# Extract data & clean 


# Complete list 

initial_list_df = pd.read_csv(url_complete_list, delimiter = '\t', skiprows = 4) #skip the first 4 rows
complete_list_df = initial_list_df.drop(index = [1, 2]) # delete rows 1 & 2
complete_list_df = complete_list_df['SWISS GLACIER LIST (AVAILABLE DATA)'].str.split(';', expand = True) # split data in several columns to get clean information
complete_list_df = complete_list_df.reset_index(drop = True)

# make first row appear as the column indices
new_headers_c = complete_list_df.iloc[0]
complete_list_df = complete_list_df[1:]
complete_list_df.columns = new_headers_c

# Rename columns to include measurement units
complete_list_df.rename(columns = {'glacier area': 'glacier area (km2)'}, inplace = True)
complete_list_df.rename(columns = {'survey year for glacier area': 'survey year for glacier area (yyyy)'}, inplace = True)
complete_list_df.rename(columns = {'coordx': 'coordx (X_LV95)'}, inplace = True)
complete_list_df.rename(columns = {'coordy': 'coordy (Y_LV95)'}, inplace = True)



# Length change

initial_length_df = pd.read_csv(url_length_change, delimiter = '\t', skiprows = 4) # skip the first 4 rows
length_change_df = initial_length_df.drop(index = [1, 2]) # delete rows 1 & 2
length_change_df = length_change_df['SWISS GLACIER LENGTH CHANGE'].str.split(',', expand = True) # split data in several columns to get clean information
length_change_df = length_change_df.reset_index(drop = True)

# Merging the last three columns that were split unintendedly
length_change_df['merged columns'] = length_change_df[length_change_df.columns[9]].astype(str) + ' ' + length_change_df[length_change_df.columns[8]].astype(str) + ' ' + length_change_df[length_change_df.columns[10]].astype(str)

# Drop the original last three columns (they are no longer needed)
length_change_df = length_change_df.drop(columns=[length_change_df.columns[10], length_change_df.columns[9], length_change_df.columns[8]])

# Make first row appear as the column indices
new_headers = length_change_df.iloc[0]
length_change_df = length_change_df[1:]
length_change_df.columns = new_headers
length_change_df = length_change_df.rename(columns = {'None observer None': 'observer'})

# Remove all 'None' that appear in the last column because of the merger
length_change_df['observer'] = (
    length_change_df['observer']
        .str.replace('None', '', regex = False)   # remove all 'None'
        .str.replace(' - ', ' -', regex = False)  # fix leftover hyphen spacing
        .str.strip()                            # remove leading/trailing spaces
)

# Rename all columns that contain numerical values to include the units

length_change_df.rename(columns = {'start date of observation': 'start date of observation (yyyy-mm-dd)'}, inplace = True)
length_change_df.rename(columns = {'end date of observation': 'end date of observation (yyyy-mm-dd)'}, inplace = True)
length_change_df.rename(columns = {'length change': 'length change (m)'}, inplace = True)
length_change_df.rename(columns = {'elevation of glacier tongue': 'elevation of glacier tongue (m asl.)'}, inplace = True)



# Volume change

initial_volume_df = pd.read_csv(url_volume_change, sep = r'\s+', engine = 'python')
volume_change_df = initial_volume_df.drop(index = [0,1,2,4])
volume_change_df = volume_change_df.reset_index(drop = True)
volume_change_df.columns = volume_change_df.iloc[0]
volume_change_df = volume_change_df.drop(0)
volume_change_df = volume_change_df.drop(columns = [';'])

volume_change_df['merged_15-16'] = volume_change_df.iloc[:, 15:26].astype(str).apply(lambda x: ' '.join(x), axis = 1)
volume_change_df = volume_change_df.drop(columns = [c for c in volume_change_df.columns if str(c) in ['None', 'NaN', 'Name'] or pd.isna(c)])
volume_change_df.rename(columns = {'merged_15-16': 'Name'}, inplace = True)

# Remove all 'None' or NaN that appear in the last column because of the merger
volume_change_df['Name'] = (volume_change_df['Name'].str.replace('None', '', regex = False).str.replace('nan', '', regex = False).str.replace(' - ', ' -', regex = False).str.strip())
cols = volume_change_df.columns.tolist()

# Move the last column to the front
cols = [cols[-1]] + cols[:-1]

# Reorder the DataFrame
volume_change_df = volume_change_df[cols]

# Rename all columns that contain numerical values to include the units
volume_change_df.rename(columns = {'date_start': 'date_start (yyyymmdd)'}, inplace = True)
volume_change_df.rename(columns = {'date_end': 'date_end (yyyymmdd)'}, inplace = True)
volume_change_df.rename(columns = {'A_start': 'A_start (km2)'}, inplace = True)
volume_change_df.rename(columns = {'outline_start': 'outline_start (yyyy)'}, inplace = True)
volume_change_df.rename(columns = {'A_end': 'A_end (km2)'}, inplace = True)
volume_change_df.rename(columns = {'outline_end': 'outline_end (yyyy)'}, inplace = True)
volume_change_df.rename(columns = {'dV': 'dV (km3)'}, inplace = True)
volume_change_df.rename(columns = {'dh_mean': 'dh_mean (m)'}, inplace = True)
volume_change_df.rename(columns = {'Bgeod': 'Bgeod (mw.e.a-1)'}, inplace = True)
volume_change_df.rename(columns = {'sigma': 'sigma (mw.e.)'}, inplace = True)
volume_change_df.rename(columns = {'covered': 'covered (%)'}, inplace = True)
volume_change_df.rename(columns = {'rho_dv': 'rho_dv (kgm-3)'}, inplace = True)



# Mass balance observation period

initial_mass_balance_df = pd.read_csv(url_mass_balance, delimiter = '\t', skiprows = 4)
mass_balance_df = initial_mass_balance_df.drop(index = [1,2])
mass_balance_df = mass_balance_df['SWISS GLACIER MASS BALANCE (OBSERVATION PERIOD)'].str.split(',', expand = True)
mass_balance_df = mass_balance_df.reset_index(drop = True)
mass_balance_df['merged columns 13,14,15,16'] = mass_balance_df[mass_balance_df.columns[13]].astype(str) + ' ' + mass_balance_df[mass_balance_df.columns[14]].astype(str) + ' ' + mass_balance_df[mass_balance_df.columns[15]].astype(str) + ' ' + mass_balance_df[mass_balance_df.columns[16]].astype(str)
mass_balance_df = mass_balance_df.drop(
    columns = [mass_balance_df.columns[13],
                   mass_balance_df.columns[14],
                   mass_balance_df.columns[15],
                   mass_balance_df.columns[16]])
new_headers_mb = mass_balance_df.iloc[0]
mass_balance_df = mass_balance_df[1:]
mass_balance_df.columns = new_headers_mb
mass_balance_df = mass_balance_df.rename(columns = {'observer None None None': 'observer'})
mass_balance_df['observer'] = (mass_balance_df['observer'].str.replace('None', '', regex = False).str.replace(' - ', ' -', regex = False).str.strip())

mass_balance_df.rename(columns = {'start date of observation': 'start date of observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_df.rename(columns = {'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_df.rename(columns = {'end date of observation': 'end date of observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_df.rename(columns = {'winter mass balance': 'winter mass balance (mm w.e.)'}, inplace = True)
mass_balance_df.rename(columns = {'summer mass balance': 'summer mass balance (mm w.e.)'}, inplace = True)
mass_balance_df.rename(columns = {'annual mass balance': 'annual mass balance (mm w.e.)'}, inplace = True)
mass_balance_df.rename(columns = {'equilibrium line altitude': 'equilibrium line altitude (m asl.)'}, inplace = True)
mass_balance_df.rename(columns = {'accumulation area ratio': 'accumulation area ratio (%)'}, inplace = True)
mass_balance_df.rename(columns = {'glacier area': 'glacier area (km2)'}, inplace = True)
mass_balance_df.rename(columns = {'minimum elevation of glacier': 'minimum elevation of glacier (m asl.)'}, inplace = True)
mass_balance_df.rename(columns = {'maximum elevation of glacier': 'maximum elevation of glacier (m asl.)'}, inplace = True)



# Mass balance observation period with elevation bins

initial_mass_balance_eb_df = pd.read_csv(url_mass_balance2, delimiter = '\t', skiprows = 4)
mass_balance_eb_df = initial_mass_balance_eb_df.drop(index = [1,2])
mass_balance_eb_df = mass_balance_eb_df['SWISS GLACIER MASS BALANCE (OBSERVATION PERIOD) ELEVATION BINS'].str.split(',', expand = True)
mass_balance_eb_df = mass_balance_eb_df.reset_index(drop = True)
new_headers_mb_eb = mass_balance_eb_df.iloc[0]
mass_balance_eb_df = mass_balance_eb_df[1:]
mass_balance_eb_df.columns = new_headers_mb_eb

mass_balance_eb_df.rename(columns = {'start date of observation': 'start date of observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_eb_df.rename(columns = {'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_eb_df.rename(columns = {'end date of observation': 'end date of observation (yyyy-mm-dd)'}, inplace=True)
mass_balance_eb_df.rename(columns = {'winter mass balance': 'winter mass balance (mm w.e.)'}, inplace = True)
mass_balance_eb_df.rename(columns = {'summer mass balance': 'summer mass balance (mm w.e.)'}, inplace = True)
mass_balance_eb_df.rename(columns = {'annual mass balance': 'annual mass balance (mm w.e.)'}, inplace = True)
mass_balance_eb_df.rename(columns = {'area of elevation bin': 'area of elevation bin (km2)'}, inplace = True)
mass_balance_eb_df.rename(columns = {'lower elevation of bin': 'lower elevation of bin (m asl.)'}, inplace = True)
mass_balance_eb_df.rename(columns = {'upper elevation of bin': 'upper elevation of bin (m asl.)'}, inplace = True)



# Mass balance hydrological year

initial_mass_balance_hy_df = pd.read_csv(url_mass_balance3, delimiter = '\t', skiprows=4)
mass_balance_hy_df = initial_mass_balance_hy_df.drop(index = [1,2])
mass_balance_hy_df = mass_balance_hy_df['SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR)'].str.split(',', expand = True)
mass_balance_hy_df = mass_balance_hy_df.reset_index(drop = True)
# Merging the last 4 columns into a new column to get fewer less important columns and eventually cleaner data
mass_balance_hy_df['merged columns 13,14,15,16'] = mass_balance_hy_df[mass_balance_hy_df.columns[13]].astype(str) + ' ' + mass_balance_hy_df[mass_balance_hy_df.columns[14]].astype(str) + ' ' + mass_balance_hy_df[mass_balance_hy_df.columns[15]].astype(str) + ' ' + mass_balance_hy_df[mass_balance_hy_df.columns[16]].astype(str)
mass_balance_hy_df = mass_balance_hy_df.drop(
    columns = [mass_balance_hy_df.columns[13],
                   mass_balance_hy_df.columns[14],
                   mass_balance_hy_df.columns[15],
                   mass_balance_hy_df.columns[16]])
new_headers_mb_hy = mass_balance_hy_df.iloc[0]
mass_balance_hy_df = mass_balance_hy_df[1:]
mass_balance_hy_df.columns = new_headers_mb_hy
mass_balance_hy_df = mass_balance_hy_df.rename(columns = {'observer None None None': 'observer'})
mass_balance_hy_df['observer'] = (mass_balance_hy_df['observer'].str.replace('None', '', regex = False).str.replace(' - ', ' -', regex = False).str.strip())

mass_balance_hy_df.rename(columns = {'start date of observation': 'start date of observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)'}, inplace=True)
mass_balance_hy_df.rename(columns = {'end date of observation': 'end date of observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'winter mass balance': 'winter mass balance (mm w.e.)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'summer mass balance': 'summer mass balance (mm w.e.)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'annual mass balance': 'annual mass balance (mm w.e.)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'equilibrium line altitude': 'equilibrium line altitude (m asl.)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'accumulation area ratio': 'accumulation area ratio (%)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'glacier area': 'glacier area (km2)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'minimum elevation of glacier': 'minimum elevation of glacier (m asl.)'}, inplace = True)
mass_balance_hy_df.rename(columns = {'maximum elevation of glacier': 'maximum elevation of glacier (m asl.)'}, inplace = True)



# Mass balance hydrological year with elevation bins

initial_mass_balance_hy_eb_df = pd.read_csv(url_mass_balance4, delimiter = '\t', skiprows = 4)
mass_balance_hy_eb_df = initial_mass_balance_hy_eb_df.drop(index = [1,2])
mass_balance_hy_eb_df = mass_balance_hy_eb_df['SWISS GLACIER MASS BALANCE (HYDROLOGICAL YEAR) ELEVATION BINS'].str.split(',', expand = True)
mass_balance_hy_eb_df = mass_balance_hy_eb_df.reset_index(drop = True)
# Merging the last 4 columns into a new column to get fewer less important columns and eventually cleaner data
mass_balance_hy_eb_df['merged columns 11,12,13,14'] = mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[11]].astype(str) + ' ' +     mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[12]].astype(str) + ' ' +           mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[13]].astype(str) + ' ' + mass_balance_hy_eb_df[mass_balance_hy_eb_df.columns[14]].astype(str)
mass_balance_hy_eb_df = mass_balance_hy_eb_df.drop(
columns = [mass_balance_hy_eb_df.columns[11],
               mass_balance_hy_eb_df.columns[12],
               mass_balance_hy_eb_df.columns[13],
               mass_balance_hy_eb_df.columns[14]])
new_headers_mb_hy_eb = mass_balance_hy_eb_df.iloc[0]
mass_balance_hy_eb_df = mass_balance_hy_eb_df[1:]
mass_balance_hy_eb_df.columns = new_headers_mb_hy_eb
mass_balance_hy_eb_df = mass_balance_hy_eb_df.rename(columns = {'observer None None None': 'observer'})
mass_balance_hy_eb_df['observer'] = (mass_balance_hy_eb_df['observer'].str.replace('None', '', regex = False).str.replace(' - ', ' -', regex = False).str.strip())

mass_balance_hy_eb_df.rename(columns = {'start date of observation': 'start date of observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'end date of winter observation': 'end date of winter observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'end date of observation': 'end date of observation (yyyy-mm-dd)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'winter mass balance': 'winter mass balance (mm w.e.)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'summer mass balance': 'summer mass balance (mm w.e.)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'annual mass balance': 'annual mass balance (mm w.e.)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'area of elevation bin': 'area of elevation bin (km2)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'lower elevation of bin': 'lower elevation of bin (m asl.)'}, inplace = True)
mass_balance_hy_eb_df.rename(columns = {'upper elevation of bin': 'upper elevation of bin (m asl.)'}, inplace = True)



# Create CSV files with the cleaned dataframes to work further

folder_path = "project_glaciers/data_extraction"
os.makedirs(folder_path, exist_ok=True)

complete_list_df.to_csv(os.path.join(folder_path, "complete_list.csv"), index = False)
length_change_df.to_csv(os.path.join(folder_path, "length_change.csv"), index = False)
volume_change_df.to_csv(os.path.join(folder_path, "volume_change.csv"), index = False)
mass_balance_df.to_csv(os.path.join(folder_path, "mass_balance_op.csv"), index = False)
mass_balance_eb_df.to_csv(os.path.join(folder_path, "mass_balance_op_eb.csv"), index = False)
mass_balance_hy_df.to_csv(os.path.join(folder_path, "mass_balance_hy.csv"), index = False)
mass_balance_hy_eb_df.to_csv(os.path.join(folder_path, "mass_balance_hy_eb.csv"), index = False)

