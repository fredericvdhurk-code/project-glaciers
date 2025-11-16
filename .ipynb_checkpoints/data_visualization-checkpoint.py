import pandas as pd
import matplotlib.pyplot as plt

length_change_df = pd.read_csv('project_glaciers/data_extraction/length_change.csv')
volume_change_df = pd.read_csv('project_glaciers/data_extraction/volume_change.csv')
mass_balance_hy_df = pd.read_csv('project_glaciers/data_extraction/mass_balance_hy.csv')
mass_balance_hy_eb_df = pd.read_csv('project_glaciers/data_extraction/mass_balance_hy_eb.csv')




# Rhonegletscher length change visualization

rhone_lc_df = length_change_df[length_change_df['glacier name'] == 'Rhonegletscher']
rhone_lc_df = rhone_lc_df[['start date of observation (yyyy-mm-dd)',
                         'end date of observation (yyyy-mm-dd)',
                         'length change (m)']]
new_row_r_lc = pd.DataFrame({'start date of observation (yyyy-mm-dd)': ['1879-09-01'],
                                  'end date of observation (yyyy-mm-dd)': ['1879-09-01'],
                                  'length change (m)': [0]})
rhone_lc_df = pd.concat([new_row_r_lc, rhone_lc_df], ignore_index=True)
#rhone_df['end_date_splitted'] = rhone_df['end date of observation (yyyy-mm-dd)'].str.split('-')
#rhone_df['year'] = rhone_df['end_date_splitted'].str[0]

# Convert date columns to datetime
rhone_lc_df['date'] = pd.to_datetime(rhone_lc_df['end date of observation (yyyy-mm-dd)'])
rhone_lc_df = rhone_lc_df[['length change (m)', 'date']]
rhone_lc_df.loc[:, 'cumulative length change (m)'] = rhone_lc_df['length change (m)'].cumsum()



plt.figure(figsize=(12, 6))

# Plot the bar chart
plt.bar(
        rhone_lc_df['date'],  # x-axis: dates
        rhone_lc_df['length change (m)'],  # y-axis: length change
        color='skyblue',  # Bar color
        width=300
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Length Change [m]')
plt.title('Rhonegletscher Length Change Over Time')
plt.grid(True, linestyle='--', alpha=0.3)  # Grid only for y-axis

plt.show()



plt.figure(figsize=(10, 6))

# Plot length_change over time
plt.plot(
        rhone_lc_df['date'],
        rhone_lc_df['cumulative length change (m)'],
        linestyle = '-',
        color = 'skyblue',
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Cumulative Length Change [m]')
plt.title('Rhonegletscher Cumulative Length Change Over Time')
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()




# Aletschgletscher length change visualization

aletsch_lc_df = length_change_df[length_change_df['glacier name'] == 'Grosser Aletschgletscher']
aletsch_lc_df = aletsch_lc_df[['start date of observation (yyyy-mm-dd)',
                             'end date of observation (yyyy-mm-dd)',
                             'length change (m)']]
new_row_a_lc = pd.DataFrame({'start date of observation (yyyy-mm-dd)': ['1870-09-01'],
                                  'end date of observation (yyyy-mm-dd)': ['1870-09-01'],
                                  'length change (m)': [0]})
aletsch_lc_df = pd.concat([new_row_a_lc, aletsch_lc_df], ignore_index=True)

# Convert date columns to datetime
aletsch_lc_df['end date'] = pd.to_datetime(aletsch_lc_df['end date of observation (yyyy-mm-dd)'])
aletsch_lc_df = aletsch_lc_df[['length change (m)', 'end date']]
aletsch_lc_df.loc[:, 'cumulative length change (m)'] = aletsch_lc_df['length change (m)'].cumsum()



plt.figure(figsize=(10, 6))

# Plot length_change over time
plt.plot(
        aletsch_lc_df['end date'],
        aletsch_lc_df['cumulative length change (m)'],
        linestyle = '-',
        color = 'lightgrey',
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Cumulative Length Change [m]')
plt.title('Aletschgletscher Cumulative Length Change Over Time')
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()




# Rhonegletscher mass balance visualization


rhone_mb_hy_df = mass_balance_hy_df[mass_balance_hy_df['glacier name'] == 'Rhonegletscher']
rhone_mb_hy_df = rhone_mb_hy_df[['start date of observation (yyyy-mm-dd)',
                                     'end date of observation (yyyy-mm-dd)',
                                     'annual mass balance (mm w.e.)'
                                ]]
rhone_mb_hy_df = rhone_mb_hy_df.reset_index(drop=True)
rhone_mb_hy_df = rhone_mb_hy_df.drop(rhone_mb_hy_df.index[0:28])
rhone_mb_hy_df = rhone_mb_hy_df.reset_index(drop=True)
new_row_r_mb = pd.DataFrame({'start date of observation (yyyy-mm-dd)': ['2006-10-01'],
                                  'end date of observation (yyyy-mm-dd)': ['2006-10-01'],
                                  'annual mass balance (mm w.e.)': [0]})
rhone_mb_hy_df = pd.concat([new_row_r_mb, rhone_mb_hy_df], ignore_index=True)

# Convert date columns to datetime
rhone_mb_hy_df['end date'] = pd.to_datetime(rhone_mb_hy_df['end date of observation (yyyy-mm-dd)'])
rhone_mb_hy_df = rhone_mb_hy_df[['annual mass balance (mm w.e.)', 'end date']]
rhone_mb_hy_df.loc[:, 'cumulative annual mass balance (mm w.e.)'] = rhone_mb_hy_df['annual mass balance (mm w.e.)'].cumsum()



plt.figure(figsize=(10, 6))

# Plot length_change over time
plt.plot(
        rhone_mb_hy_df['end date'],
        rhone_mb_hy_df['cumulative annual mass balance (mm w.e.)'],
        linestyle = '-',
        color = 'skyblue',
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Cumulative Annual Mass Balance [mm w.e.]')
plt.title('Rhonegletscher Cumulative Annual Mass Balance Over Time (Hydrological Year)')
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()



plt.figure(figsize=(12, 6))

# Plot the bar chart
plt.bar(
        rhone_mb_hy_df['end date'],  # x-axis: dates
        rhone_mb_hy_df['annual mass balance (mm w.e.)'],  # y-axis: length change
        color = 'skyblue',  # Bar color
        width = 200
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Annual Mass Balance [mm w.e.]')
plt.title('Rhonegletscher Annual Mass Balance Over Time (Hydrological Year)')
plt.grid(True, linestyle='--', alpha=0.3)  # Grid only for y-axis

plt.show()




# Aletschgletscher mass balance visualization


aletsch_mb_hy_df = mass_balance_hy_df[mass_balance_hy_df['glacier name'] == 'Grosser Aletschgletscher']
aletsch_mb_hy_df = aletsch_mb_hy_df[['start date of observation (yyyy-mm-dd)',
                                     'end date of observation (yyyy-mm-dd)',
                                     'annual mass balance (mm w.e.)'
                                    ]]
aletsch_mb_hy_df = aletsch_mb_hy_df.reset_index(drop=True)
new_row_a_mb = pd.DataFrame({'start date of observation (yyyy-mm-dd)': ['1914-10-01'],
                                  'end date of observation (yyyy-mm-dd)': ['1914-10-01'],
                                  'annual mass balance (mm w.e.)': [0]})
aletsch_mb_hy_df = pd.concat([new_row_a_mb, aletsch_mb_hy_df], ignore_index=True)

aletsch_mb_hy_df_2007 = aletsch_mb_hy_df.drop(aletsch_mb_hy_df.index[0:93])
aletsch_mb_hy_df_2007 = aletsch_mb_hy_df_2007.reset_index(drop=True)
new_row_a_mb_2007 = pd.DataFrame({'start date of observation (yyyy-mm-dd)': ['2006-09-30'],
                                  'end date of observation (yyyy-mm-dd)': ['2006-09-30'],
                                  'annual mass balance (mm w.e.)': [0]})
aletsch_mb_hy_df_2007 = pd.concat([new_row_a_mb_2007, aletsch_mb_hy_df_2007], ignore_index=True)


# Convert date columns to datetime
aletsch_mb_hy_df['end date'] = pd.to_datetime(aletsch_mb_hy_df['end date of observation (yyyy-mm-dd)'])
aletsch_mb_hy_df = aletsch_mb_hy_df[['annual mass balance (mm w.e.)', 'end date']]
aletsch_mb_hy_df.loc[:, 'cumulative annual mass balance (mm w.e.)'] = aletsch_mb_hy_df['annual mass balance (mm w.e.)'].cumsum()


aletsch_mb_hy_df_2007['end date'] = pd.to_datetime(aletsch_mb_hy_df_2007['end date of observation (yyyy-mm-dd)'])
aletsch_mb_hy_df_2007 = aletsch_mb_hy_df_2007[['annual mass balance (mm w.e.)', 'end date']]
aletsch_mb_hy_df_2007.loc[:, 'cumulative annual mass balance (mm w.e.)'] = aletsch_mb_hy_df_2007['annual mass balance (mm w.e.)'].cumsum()



plt.figure(figsize=(12, 6))

# Plot the bar chart
plt.bar(
        aletsch_mb_hy_df['end date'],  
        aletsch_mb_hy_df['annual mass balance (mm w.e.)'],  
        color = 'lightgrey',  
        width = 200
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Annual Mass Balance [mm w.e.]')
plt.title('Aletschgletscher Annual Mass Balance Over Time (Hydrological Year)')
plt.grid(True, linestyle='--', alpha=0.3)  # Grid only for y-axis

plt.show()



plt.figure(figsize=(10, 6))

# Plot length_change over time
plt.plot(
        aletsch_mb_hy_df['end date'],
        aletsch_mb_hy_df['cumulative annual mass balance (mm w.e.)'],
        linestyle = '-',
        color = 'skyblue',
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Cumulative Annual Mass Balance [mm w.e.]')
plt.title('Aletschgletscher Cumulative Annual Mass Balance Over Time (Hydrological Year)')
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()



plt.figure(figsize=(12, 6))

# Plot the bar chart
plt.bar(
        aletsch_mb_hy_df_2007['end date'],  
        aletsch_mb_hy_df_2007['annual mass balance (mm w.e.)'],  
        color = 'lightgrey',  
        width = 200
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Annual Mass Balance [mm w.e.]')
plt.title('Aletschgletscher Annual Mass Balance Over Time (Hydrological Year)')
plt.grid(True, linestyle='--', alpha=0.3)  # Grid only for y-axis

plt.show()



plt.figure(figsize=(10, 6))

# Plot length_change over time
plt.plot(
        aletsch_mb_hy_df_2007['end date'],
        aletsch_mb_hy_df_2007['cumulative annual mass balance (mm w.e.)'],
        linestyle = '-',
        color = 'lightgrey',
)

# Customize the plot
plt.xlabel('Year')
plt.ylabel('Cumulative Annual Mass Balance [mm w.e.]')
plt.title('Aletschgletscher Cumulative Annual Mass Balance Over Time (Hydrological Year)')
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()