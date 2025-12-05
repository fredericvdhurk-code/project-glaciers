import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.backends.backend_pdf import PdfPages
import os
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels.api as sm
from io import StringIO
import sys




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
davos_dev_temp = pd.read_csv('project-glaciers/data/weather_dev6190_davos_temp.csv').iloc[::-1].reset_index(drop=True)
davos_dev_prec = pd.read_csv('project-glaciers/data/weather_dev6190_davos_prec.csv').iloc[::-1].reset_index(drop=True)
sion_dev_temp = pd.read_csv('project-glaciers/data/weather_dev6190_sion_temp.csv').iloc[::-1].reset_index(drop=True)
sion_dev_prec = pd.read_csv('project-glaciers/data/weather_dev6190_sion_prec.csv').iloc[::-1].reset_index(drop=True)
altdorf_dev_temp = pd.read_csv('project-glaciers/data/weather_dev6190_altdorf_temp.csv').iloc[::-1].reset_index(drop=True)
altdorf_dev_prec = pd.read_csv('project-glaciers/data/weather_dev6190_altdorf_prec.csv').iloc[::-1].reset_index(drop=True)

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
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig

def plot_glacier_length_change_bar(glacier_df, glacier_name, figsize=(12, 6)):
    if glacier_df is None:
        return None
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig

def plot_mass_balance_bar(mb_df, glacier_name, balance_type, figsize=(12, 6)):
    if mb_df is None:
        return None
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig

def plot_cumulative_mass_balance(mb_df, glacier_name, figsize=(12, 6)):
    if mb_df is None:
        return None
    annual_mb_df = mb_df.resample("YS").last()
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig

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
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig

# --- Weather Plotting Functions ---
def plot_summer_temperature(summer_df, city_name, norm6190, norm9120, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig

def plot_winter_precipitation(winter_df, city_name, norm6190, norm9120, figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize)
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
    return fig

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
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.scatter(temp_data[temp_column],
                mass_balance_summer['summer mass balance (mm w.e.)'])
    ax1.set_xlabel('Summer Mean Temperature (°C)')
    ax1.set_ylabel('Summer Mass Balance (mm w.e.)')
    ax1.set_title(f"{glacier_name} Summer Mass Balance vs. Temperature")
    ax1.grid(True)
    plt.tight_layout()

    fig2 = None
    if mass_balance_winter is not None and precip_data is not None:
        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.scatter(precip_data[precip_column],
                    mass_balance_winter['winter mass balance (mm w.e.)'])
        ax2.set_xlabel('Winter Total Precipitation (mm)')
        ax2.set_ylabel('Winter Mass Balance (mm w.e.)')
        ax2.set_title(f"{glacier_name} Winter Mass Balance vs. Precipitation")
        ax2.grid(True)
        plt.tight_layout()

    return fig1, fig2

# --- Regression Analysis (unchanged) ---
def load_data(norm_period):
    mass_balance_df = pd.read_csv('project-glaciers/data/mass_balance_hy.csv').iloc[::-1].reset_index(drop=True)
    if norm_period == "1961-1990":
        davos_dev_temp = pd.read_csv('project-glaciers/data/weather_dev6190_davos_temp.csv').iloc[::-1].reset_index(drop=True)
        davos_dev_prec = pd.read_csv('project-glaciers/data/weather_dev6190_davos_prec.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_temp = pd.read_csv('project-glaciers/data/weather_dev6190_sion_temp.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_prec = pd.read_csv('project-glaciers/data/weather_dev6190_sion_prec.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_temp = pd.read_csv('project-glaciers/data/weather_dev6190_altdorf_temp.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_prec = pd.read_csv('project-glaciers/data/weather_dev6190_altdorf_prec.csv').iloc[::-1].reset_index(drop=True)
    else:
        davos_dev_temp = pd.read_csv('project-glaciers/data/weather_dev9120_davos_temp.csv').iloc[::-1].reset_index(drop=True)
        davos_dev_prec = pd.read_csv('project-glaciers/data/weather_dev9120_davos_prec.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_temp = pd.read_csv('project-glaciers/data/weather_dev9120_sion_temp.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_prec = pd.read_csv('project-glaciers/data/weather_dev9120_sion_prec.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_temp = pd.read_csv('project-glaciers/data/weather_dev9120_altdorf_temp.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_prec = pd.read_csv('project-glaciers/data/weather_dev9120_altdorf_prec.csv').iloc[::-1].reset_index(drop=True)
    glacier_mappings = {
        'Grosser Aletschgletscher': {'temp': sion_dev_temp, 'prec': sion_dev_prec},
        'Allalingletscher': {'temp': sion_dev_temp, 'prec': sion_dev_prec},
        'Griesgletscher': {'temp': sion_dev_temp, 'prec': sion_dev_prec},
        'Schwarzberggletscher': {'temp': sion_dev_temp, 'prec': sion_dev_prec},
        'Hohlaubgletscher': {'temp': sion_dev_temp, 'prec': sion_dev_prec},
        'Glacier du Giétro': {'temp': sion_dev_temp, 'prec': sion_dev_prec},
        'Silvrettagletscher': {'temp': davos_dev_temp, 'prec': davos_dev_prec},
        'Claridenfirn': {'temp': altdorf_dev_temp, 'prec': altdorf_dev_prec}
    }
    return mass_balance_df, glacier_mappings

def run_regression_analysis(mass_balance_df, glacier_mappings, temp_cols, prec_cols, analysis_name, norm_period):
    captured_output = StringIO()
    sys.stdout = captured_output
    print(f"\n{'='*80}")
    print(f"{analysis_name} ANALYSIS USING {norm_period} CLIMATE NORMS")
    print('='*80)
    for glacier_name, weather_data in glacier_mappings.items():
        print(f"\n{'='*80}")
        print(f"{analysis_name} for {glacier_name} ({norm_period} norms)")
        print('='*80)
        try:
            mb_data = mass_balance_df[mass_balance_df['glacier name'] == glacier_name][['annual mass balance (mm w.e.)']].copy()
            mb_data = mb_data.reset_index(drop=True)
            if len(mb_data) == 0:
                print(f"No mass balance data found for {glacier_name}")
                continue
            if glacier_name == 'Claridenfirn':
                temp_df = weather_data['temp'][temp_cols].copy()
                prec_df = weather_data['prec'][prec_cols].copy()
                temp_years = weather_data['temp']['hydrological year']
                prec_years = weather_data['prec']['hydrological year']
                temp_df = temp_df[~temp_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)
                prec_df = prec_df[~prec_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)
            else:
                temp_df = weather_data['temp'][temp_cols].reset_index(drop=True)
                prec_df = weather_data['prec'][prec_cols].reset_index(drop=True)
            max_index = min(len(mb_data) - 1, len(temp_df) - 1, len(prec_df) - 1)
            temp_df = temp_df.iloc[0:max_index+1]
            prec_df = prec_df.iloc[0:max_index+1]
            reg_data = mb_data.iloc[0:max_index+1].copy()
            reg_data = pd.concat([reg_data, temp_df, prec_df], axis=1)
            X = reg_data[temp_cols + prec_cols]
            y = reg_data['annual mass balance (mm w.e.)']
            X = sm.add_constant(X)
            reg_data_clean = pd.concat([X, y], axis=1).dropna()
            if len(reg_data_clean) == 0:
                print(f"No valid data remaining for {glacier_name} after cleaning")
                continue
            model = sm.OLS(reg_data_clean['annual mass balance (mm w.e.)'], reg_data_clean.drop('annual mass balance (mm w.e.)', axis=1)).fit()
            print(f"\nNumber of observations: {len(reg_data_clean)}")
            print("\nRegression Summary:")
            print(model.summary())
            print("\nCoefficient Interpretation:")
            for param in model.params.index:
                if param == 'const':
                    print(f"Intercept (normal mass balance): {model.params[param]:.2f} (p={model.pvalues[param]:.4f})")
                else:
                    print(f"{param}: {model.params[param]:.2f} (p={model.pvalues[param]:.4f})")
            print("\nVariance Inflation Factors (VIF):")
            vif_data = pd.DataFrame()
            vif_data["Variable"] = X.columns
            vif_data["VIF"] = [variance_inflation_factor(reg_data_clean.drop('annual mass balance (mm w.e.)', axis=1).values, i) for i in range(len(X.columns))]
            print(vif_data)
            print(f"\nR-squared: {model.rsquared:.4f}")
            print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        except Exception as e:
            print(f"Error processing {glacier_name}: {str(e)}")
    sys.stdout = sys.__stdout__
    return captured_output.getvalue()

def run_all_analyses_for_glacier(glacier_name, norm_periods=["1961-1990", "1991-2020"]):
    outputs = {}
    for norm_period in norm_periods:
        mass_balance_df, glacier_mappings = load_data(norm_period)
        monthly_temp_cols = ['may_td', 'june_td', 'july_td', 'august_td', 'september_td']
        monthly_prec_cols = ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd', 'march_pd', 'april_pd']
        outputs[f"monthly_{norm_period}"] = run_regression_analysis(
            mass_balance_df, {glacier_name: glacier_mappings[glacier_name]},
            monthly_temp_cols, monthly_prec_cols, "MONTHLY DEVIATIONS", norm_period
        )
        opt_temp_cols = ['opt_season_td']
        opt_prec_cols = ['opt_season_pd']
        outputs[f"optimal_{norm_period}"] = run_regression_analysis(
            mass_balance_df, {glacier_name: glacier_mappings[glacier_name]},
            opt_temp_cols, opt_prec_cols, "OPTIMAL SEASONAL DEVIATIONS", norm_period
        )
        season_temp_cols = ['summer_td']
        season_prec_cols = ['winter_pd']
        outputs[f"seasonal_{norm_period}"] = run_regression_analysis(
            mass_balance_df, {glacier_name: glacier_mappings[glacier_name]},
            season_temp_cols, season_prec_cols, "SUMMER/WINTER SEASONAL DEVIATIONS", norm_period
        )
    return outputs

def plot_regression_text(text, title, figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.set_title(title, fontsize=12)
    ax.text(0.01, 0.99, text, fontsize=8, family='monospace', va='top')
    plt.tight_layout()
    return fig

# --- Main Function to Save All Plots ---
def save_glacier_plots_individually(
    length_change_df,
    mass_balance_hy_df,
    mass_balance_hy_eb_df,
    glaciers,
    output_dir,
    sion_summer,
    sion_winter,
    altdorf_summer,
    altdorf_winter,
    davos_summer,
    davos_winter,
    glaciers_weather_mapping
):
    os.makedirs(output_dir, exist_ok=True)
    for glacier in glaciers:
        glacier_df = prepare_glacier_data(glacier, length_change_df)
        mb_df = prepare_mass_balance_data(glacier, mass_balance_hy_df, "annual")

        output_path = os.path.join(output_dir, f"{glacier.replace(' ', '_')}_analyses.pdf")
        with PdfPages(output_path) as pdf:
            # Glacier plots
            if glacier_df is not None:
                fig1 = plot_glacier_length_change_bar(glacier_df, glacier)
                if fig1:
                    pdf.savefig(fig1)
                    plt.close(fig1)
            if glacier_df is not None:
                fig2 = plot_glacier_cumulative_length_change(glacier_df, glacier)
                if fig2:
                    pdf.savefig(fig2)
                    plt.close(fig2)
            if mb_df is not None:
                fig3 = plot_mass_balance_bar(mb_df, glacier, "annual")
                if fig3:
                    pdf.savefig(fig3)
                    plt.close(fig3)
            if mb_df is not None:
                fig4 = plot_cumulative_mass_balance(mb_df, glacier)
                if fig4:
                    pdf.savefig(fig4)
                    plt.close(fig4)
            fig5 = plot_mass_balance_for_glaciers_eb(mass_balance_hy_eb_df, glacier)
            if fig5:
                pdf.savefig(fig5)
                plt.close(fig5)

            # Weather plots
            if glacier in ['Grosser Aletschgletscher', 'Allalingletscher', 'Hohlaubgletscher', 'Griesgletscher', 'Schwarzberggletscher', 'Glacier du Giétro']:
                fig6 = plot_summer_temperature(sion_summer, 'Sion', sion_summer_norm_6190_t, sion_summer_norm_9120_t)
                if fig6:
                    pdf.savefig(fig6)
                    plt.close(fig6)
                fig7 = plot_winter_precipitation(sion_winter, 'Sion', sion_winter_norm_6190_p, sion_winter_norm_9120_p)
                if fig7:
                    pdf.savefig(fig7)
                    plt.close(fig7)
            elif glacier == 'Claridenfirn':
                fig6 = plot_summer_temperature(altdorf_summer, 'Altdorf', altdorf_summer_norm_6190_t, altdorf_summer_norm_9120_t)
                if fig6:
                    pdf.savefig(fig6)
                    plt.close(fig6)
                fig7 = plot_winter_precipitation(altdorf_winter, 'Altdorf', altdorf_winter_norm_6190_p, altdorf_winter_norm_9120_p)
                if fig7:
                    pdf.savefig(fig7)
                    plt.close(fig7)
            elif glacier == 'Silvrettagletscher':
                fig6 = plot_summer_temperature(davos_summer, 'Davos', davos_summer_norm_6190_t, davos_summer_norm_9120_t)
                if fig6:
                    pdf.savefig(fig6)
                    plt.close(fig6)
                fig7 = plot_winter_precipitation(davos_winter, 'Davos', davos_winter_norm_6190_p, davos_winter_norm_9120_p)
                if fig7:
                    pdf.savefig(fig7)
                    plt.close(fig7)

            # Mass balance vs. weather plots
            if glacier in glaciers_weather_mapping:
                mapping = glaciers_weather_mapping[glacier]
                fig8, fig9 = plot_mass_balance_weather(
                    glacier,
                    mapping["temp_data"],
                    mapping["temp_column"],
                    mapping["mass_balance_summer"],
                    mapping["mass_balance_winter"],
                    mapping["precip_data"],
                    mapping["precip_column"]
                )
                if fig8:
                    pdf.savefig(fig8)
                    plt.close(fig8)
                if fig9:
                    pdf.savefig(fig9)
                    plt.close(fig9)

            # Regression summaries
            regression_outputs = run_all_analyses_for_glacier(glacier)
            for key, output in regression_outputs.items():
                fig = plot_regression_text(output, f"Regression: {key.replace('_', ' ').title()}")
                pdf.savefig(fig)
                plt.close(fig)

        print(f"Saved plots for {glacier} to {output_path}")

# --- Load Data ---
length_change_df = pd.read_csv('project-glaciers/data/length_change.csv')
mass_balance_hy_df = pd.read_csv('project-glaciers/data/mass_balance_hy.csv')
mass_balance_hy_eb_df = pd.read_csv('project-glaciers/data/mass_balance_hy_eb.csv')
davos_summer = pd.read_csv('project-glaciers/data/weather_data_davos_summer.csv')
davos_winter = pd.read_csv('project-glaciers/data/weather_data_davos_winter.csv')
altdorf_summer = pd.read_csv('project-glaciers/data/weather_data_altdorf_summer.csv')
altdorf_winter = pd.read_csv('project-glaciers/data/weather_data_altdorf_winter.csv')
sion_summer = pd.read_csv('project-glaciers/data/weather_data_sion_summer.csv')
sion_winter = pd.read_csv('project-glaciers/data/weather_data_sion_winter.csv')

# Convert date columns to datetime
for df in [altdorf_summer, altdorf_winter, davos_summer, davos_winter, sion_summer, sion_winter]:
    df['date'] = pd.to_datetime(df['date'])

# Climate norms
altdorf_summer_norm_6190_t = 15.3
altdorf_winter_norm_6190_p = 525
davos_summer_norm_6190_t = 9.1
davos_winter_norm_6190_p = 431
sion_summer_norm_6190_t = 16.5
sion_winter_norm_6190_p = 365
altdorf_summer_norm_9120_t = 16.7
altdorf_winter_norm_9120_p = 530
davos_summer_norm_9120_t = 10.5
davos_winter_norm_9120_p = 449
sion_summer_norm_9120_t = 18.1
sion_winter_norm_9120_p = 324

# Filter Claridenfirn summer and winter data
dates_to_exclude_summer = pd.to_datetime(['1994-05-01', '1995-05-01'])
dates_to_exclude_winter = pd.to_datetime(['1993-10-01', '1994-10-01'])
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
glaciers_weather_mapping = {
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

# --- Run ---
glaciers = [
    'Grosser Aletschgletscher',
    'Silvrettagletscher',
    'Claridenfirn',
    'Griesgletscher',
    'Allalingletscher',
    'Schwarzberggletscher',
    'Hohlaubgletscher',
    'Glacier du Giétro'
]
save_glacier_plots_individually(
    length_change_df,
    mass_balance_hy_df,
    mass_balance_hy_eb_df,
    glaciers,
    '/files/project-glaciers/results',
    sion_summer,
    sion_winter,
    altdorf_summer,
    altdorf_winter,
    davos_summer,
    davos_winter,
    glaciers_weather_mapping
)
