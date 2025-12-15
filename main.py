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
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import TimeSeriesSplit
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import your functions from the src package
from src import (
    process_glaciers_list,
    process_length_change,
    process_mass_balance_hy,
    process_mass_balance_hy_eb,
    process_city_weather,
    calculate_hydrological_year_aggregates,
    calculate_seasonal_aggregates,
    calculate_weather_deviations_1961_1990,
    calculate_weather_deviations_1991_2020,
    prepare_glacier_data,
    prepare_mass_balance_data,
    plot_glacier_length_change_bar,
    plot_glacier_cumulative_length_change,
    plot_mass_balance_bar,
    plot_cumulative_mass_balance,
    plot_mass_balance_for_glaciers_eb,
    plot_summer_temperature,
    plot_winter_precipitation,
    plot_mass_balance_weather,
    load_data_regression,
    run_regression_analysis,
    run_all_analyses_for_glacier,
    plot_regression_text,
    save_glacier_plots_individually,
    load_data_model,
    prepare_monthly_data_model,
    prepare_seasonal_data_model,
    prepare_optimal_season_data_model,
    train_model,
    create_glacier_model_pdf
)



def main():
    print("running main.py...")
    # Path to data folders
    data_path = "/files/project-glaciers/data"
    raw_data_path = "/files/project-glaciers/data/raw_data"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(raw_data_path, exist_ok=True)

    # Load raw glacier data
    raw_glaciers_df = pd.read_csv(os.path.join(raw_data_path, "glaciers_list_raw.csv"), delimiter='\t', skiprows=4)
    raw_length_df = pd.read_csv(os.path.join(raw_data_path, "length_change_raw.csv"), delimiter='\t', skiprows=4)

    # Process glacier data
    glaciers_list_clean = process_glaciers_list(raw_glaciers_df)
    length_change_clean = process_length_change(raw_length_df)

    # Load raw mass balance data
    raw_mass_balance_hy_df = pd.read_csv(os.path.join(raw_data_path, "mass_balance_hy_raw.csv"), delimiter='\t', skiprows=4)
    raw_mass_balance_hy_eb_df = pd.read_csv(os.path.join(raw_data_path, "mass_balance_hy_eb_raw.csv"), delimiter='\t', skiprows=4)

    # Process mass balance data
    mass_balance_hy_clean = process_mass_balance_hy(raw_mass_balance_hy_df)
    mass_balance_hy_eb_clean = process_mass_balance_hy_eb(raw_mass_balance_hy_eb_df)

    # Save cleaned glacier and mass balance data
    glaciers_list_clean.to_csv(os.path.join(data_path, "glaciers_list.csv"), index=False)
    length_change_clean.to_csv(os.path.join(data_path, "length_change.csv"), index=False)
    mass_balance_hy_clean.to_csv(os.path.join(data_path, "mass_balance_hy.csv"), index=False)
    mass_balance_hy_eb_clean.to_csv(os.path.join(data_path, "mass_balance_hy_eb.csv"), index=False)

    # Load raw weather data
    sion_weather_raw = pd.read_csv(os.path.join(raw_data_path, "weather_data_sion_raw.csv"), delimiter=',')
    davos_weather_raw = pd.read_csv(os.path.join(raw_data_path, "weather_data_davos_raw.csv"), delimiter=',')
    altdorf_weather_raw = pd.read_csv(os.path.join(raw_data_path, "weather_data_altdorf_raw.csv"), delimiter=',')
    weather_metadata_raw = pd.read_csv(os.path.join(raw_data_path, "weather_metadata_raw.csv"), delimiter=',')

    # Process weather data for each city
    sion = process_city_weather(sion_weather_raw)
    davos = process_city_weather(davos_weather_raw)
    altdorf = process_city_weather(altdorf_weather_raw)

    # Save cleaned monthly weather data
    sion.to_csv(os.path.join(data_path, "weather_data_sion_monthly.csv"), index=False)
    davos.to_csv(os.path.join(data_path, "weather_data_davos_monthly.csv"), index=False)
    altdorf.to_csv(os.path.join(data_path, "weather_data_altdorf_monthly.csv"), index=False)

    # Calculate hydrological year aggregates for each city
    hy_data_sion = calculate_hydrological_year_aggregates(sion)
    hy_data_davos = calculate_hydrological_year_aggregates(davos)
    hy_data_altdorf = calculate_hydrological_year_aggregates(altdorf)

    # Save hydrological year aggregates
    hy_data_sion.to_csv(os.path.join(data_path, "weather_data_sion_hy.csv"), index=False)
    hy_data_davos.to_csv(os.path.join(data_path, "weather_data_davos_hy.csv"), index=False)
    hy_data_altdorf.to_csv(os.path.join(data_path, "weather_data_altdorf_hy.csv"), index=False)

    # Calculate seasonal aggregates for each city
    sion_winter, sion_summer = calculate_seasonal_aggregates(sion)
    davos_winter, davos_summer = calculate_seasonal_aggregates(davos)
    altdorf_winter, altdorf_summer = calculate_seasonal_aggregates(altdorf)

    # Save seasonal aggregates
    sion_summer.to_csv(os.path.join(data_path, "weather_data_sion_summer.csv"), index=False)
    sion_winter.to_csv(os.path.join(data_path, "weather_data_sion_winter.csv"), index=False)
    davos_winter.to_csv(os.path.join(data_path, "weather_data_davos_winter.csv"), index=False)
    davos_summer.to_csv(os.path.join(data_path, "weather_data_davos_summer.csv"), index=False)
    altdorf_summer.to_csv(os.path.join(data_path, "weather_data_altdorf_summer.csv"), index=False)
    altdorf_winter.to_csv(os.path.join(data_path, "weather_data_altdorf_winter.csv"), index=False)

    # Load and prepare data for deviations
    sion_before_dev = pd.read_csv(os.path.join(raw_data_path, "weather_data_sion_raw.csv"))
    sion_before_dev['date'] = pd.to_datetime(sion_before_dev['reference_timestamp'], format='%d.%m.%Y %H:%M')
    sion_before_dev = sion_before_dev.set_index('date')
    sion_before_dev = sion_before_dev[(sion_before_dev.index >= '1914-10-01') & (sion_before_dev.index < '2025-10-01')]

    davos_before_dev = pd.read_csv(os.path.join(raw_data_path, "weather_data_davos_raw.csv"))
    davos_before_dev['date'] = pd.to_datetime(davos_before_dev['reference_timestamp'], format='%d.%m.%Y %H:%M')
    davos_before_dev = davos_before_dev.set_index('date')
    davos_before_dev = davos_before_dev[(davos_before_dev.index >= '1914-10-01') & (davos_before_dev.index < '2025-10-01')]

    altdorf_before_dev = pd.read_csv(os.path.join(raw_data_path, "weather_data_altdorf_raw.csv"))
    altdorf_before_dev['date'] = pd.to_datetime(altdorf_before_dev['reference_timestamp'], format='%d.%m.%Y %H:%M')
    altdorf_before_dev = altdorf_before_dev.set_index('date')
    altdorf_before_dev = altdorf_before_dev[(altdorf_before_dev.index >= '1914-10-01') & (altdorf_before_dev.index < '2025-10-01')]

    # Calculate deviations from 1961-1990 norms
    sion_temp_dev_6190, sion_precip_dev6190 = calculate_weather_deviations_1961_1990(sion_before_dev, 'sion')
    davos_temp_dev6190, davos_precip_dev6190 = calculate_weather_deviations_1961_1990(davos_before_dev, 'davos')
    altdorf_temp_dev6190, altdorf_precip_dev6190 = calculate_weather_deviations_1961_1990(altdorf_before_dev, 'altdorf')

    # Calculate deviations from 1991-2020 norms
    sion_temp_dev_9120, sion_precip_dev_9120 = calculate_weather_deviations_1991_2020(sion_before_dev, 'sion')
    davos_temp_dev_9120, davos_precip_dev_9120 = calculate_weather_deviations_1991_2020(davos_before_dev, 'davos')
    altdorf_temp_dev_9120, altdorf_precip_dev_9120 = calculate_weather_deviations_1991_2020(altdorf_before_dev, 'altdorf')

    # Save deviation data
    sion_temp_dev_6190.to_csv(os.path.join(data_path, "weather_dev6190_sion_temp.csv"), index=False)
    sion_precip_dev6190.to_csv(os.path.join(data_path, "weather_dev6190_sion_prec.csv"), index=False)
    sion_temp_dev_9120.to_csv(os.path.join(data_path, "weather_dev9120_sion_temp.csv"), index=False)
    sion_precip_dev_9120.to_csv(os.path.join(data_path, "weather_dev9120_sion_prec.csv"), index=False)

    davos_temp_dev6190.to_csv(os.path.join(data_path, "weather_dev6190_davos_temp.csv"), index=False)
    davos_precip_dev6190.to_csv(os.path.join(data_path, "weather_dev6190_davos_prec.csv"), index=False)
    davos_temp_dev_9120.to_csv(os.path.join(data_path, "weather_dev9120_davos_temp.csv"), index=False)
    davos_precip_dev_9120.to_csv(os.path.join(data_path, "weather_dev9120_davos_prec.csv"), index=False)

    altdorf_temp_dev6190.to_csv(os.path.join(data_path, "weather_dev6190_altdorf_temp.csv"), index=False)
    altdorf_precip_dev6190.to_csv(os.path.join(data_path, "weather_dev6190_altdorf_prec.csv"), index=False)
    altdorf_temp_dev_9120.to_csv(os.path.join(data_path, "weather_dev9120_altdorf_temp.csv"), index=False)
    altdorf_precip_dev_9120.to_csv(os.path.join(data_path, "weather_dev9120_altdorf_prec.csv"), index=False)

    # --- Load Data for Plots ---
    length_change_df = pd.read_csv(os.path.join(data_path, 'length_change.csv'))
    mass_balance_hy_df = pd.read_csv(os.path.join(data_path, 'mass_balance_hy.csv'))
    mass_balance_hy_eb_df = pd.read_csv(os.path.join(data_path, 'mass_balance_hy_eb.csv'))
    davos_summer = pd.read_csv(os.path.join(data_path, 'weather_data_davos_summer.csv'))
    davos_winter = pd.read_csv(os.path.join(data_path, 'weather_data_davos_winter.csv'))
    altdorf_summer = pd.read_csv(os.path.join(data_path, 'weather_data_altdorf_summer.csv'))
    altdorf_winter = pd.read_csv(os.path.join(data_path, 'weather_data_altdorf_winter.csv'))
    sion_summer = pd.read_csv(os.path.join(data_path, 'weather_data_sion_summer.csv'))
    sion_winter = pd.read_csv(os.path.join(data_path, 'weather_data_sion_winter.csv'))

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
        glaciers_weather_mapping,
        sion_summer_norm_6190_t,
        sion_winter_norm_6190_p,
        altdorf_summer_norm_6190_t,
        altdorf_winter_norm_6190_p,
        davos_summer_norm_6190_t,
        davos_winter_norm_6190_p,
        sion_summer_norm_9120_t,
        sion_winter_norm_9120_p,
        altdorf_summer_norm_9120_t,
        altdorf_winter_norm_9120_p,
        davos_summer_norm_9120_t,
        davos_winter_norm_9120_p,
    )
    print("Starting comprehensive glacier mass balance model analysis with random splits...")

    # Load data
    mass_balance_df, weather_data = load_data_model()

    # Process each glacier
    for glacier in glaciers:
        print(f"\n{'='*80}")
        print(f"Analyzing {glacier}")
        print('='*80)

        try:
            # Prepare data for all three models
            X_monthly, y_monthly, _ = prepare_monthly_data_model(glacier, mass_balance_df, weather_data)
            X_seasonal, y_seasonal, _ = prepare_seasonal_data_model(glacier, mass_balance_df, weather_data)
            X_optimal, y_optimal, _ = prepare_optimal_season_data_model(glacier, mass_balance_df, weather_data)

            # Train all three models
            monthly_result = train_model(X_monthly, y_monthly, glacier, "Monthly Deviations")
            seasonal_result = train_model(X_seasonal, y_seasonal, glacier, "Seasonal Deviations")
            optimal_result = train_model(X_optimal, y_optimal, glacier, "Optimal Seasonal Deviations")

            # Create combined PDF for this glacier
            create_glacier_model_pdf(glacier, {
                "Monthly Deviations": monthly_result,
                "Seasonal Deviations": seasonal_result,
                "Optimal Seasonal Deviations": optimal_result
            })

            print(f"Completed analysis for {glacier}")

        except Exception as e:
            print(f"Error analyzing {glacier}: {str(e)}")
            continue

    print("\nAll model analyses complete!")


if __name__ == "__main__":
    main()

