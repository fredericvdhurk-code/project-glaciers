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

# --- Regression Analysis ---
def load_data_regression(norm_period):
    mass_balance_df = pd.read_csv('/files/project-glaciers/data/mass_balance_hy.csv').iloc[::-1].reset_index(drop=True)
    if norm_period == "1961-1990":
        davos_dev_temp = pd.read_csv('/files/project-glaciers/data/weather_dev6190_davos_temp.csv').iloc[::-1].reset_index(drop=True)
        davos_dev_prec = pd.read_csv('/files/project-glaciers/data/weather_dev6190_davos_prec.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_temp = pd.read_csv('/files/project-glaciers/data/weather_dev6190_sion_temp.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_prec = pd.read_csv('/files/project-glaciers/data/weather_dev6190_sion_prec.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_temp = pd.read_csv('/files/project-glaciers/data/weather_dev6190_altdorf_temp.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_prec = pd.read_csv('/files/project-glaciers/data/weather_dev6190_altdorf_prec.csv').iloc[::-1].reset_index(drop=True)
    else:
        davos_dev_temp = pd.read_csv('/files/project-glaciers/data/weather_dev9120_davos_temp.csv').iloc[::-1].reset_index(drop=True)
        davos_dev_prec = pd.read_csv('/files/project-glaciers/data/weather_dev9120_davos_prec.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_temp = pd.read_csv('/files/project-glaciers/data/weather_dev9120_sion_temp.csv').iloc[::-1].reset_index(drop=True)
        sion_dev_prec = pd.read_csv('/files/project-glaciers/data/weather_dev9120_sion_prec.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_temp = pd.read_csv('/files/project-glaciers/data/weather_dev9120_altdorf_temp.csv').iloc[::-1].reset_index(drop=True)
        altdorf_dev_prec = pd.read_csv('/files/project-glaciers/data/weather_dev9120_altdorf_prec.csv').iloc[::-1].reset_index(drop=True)
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
        mass_balance_df, glacier_mappings = load_data_regression(norm_period)
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

