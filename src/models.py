import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold


# Configuration
data_path = "/files/project-glaciers/data"
results_path = "/files/project-glaciers/results/model_results"
os.makedirs(results_path, exist_ok=True)

# Glaciers to analyze
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

# Weather station mapping
glacier_weather_map = {
    'Grosser Aletschgletscher': 'sion',
    'Allalingletscher': 'sion',
    'Hohlaubgletscher': 'sion',
    'Griesgletscher': 'sion',
    'Schwarzberggletscher': 'sion',
    'Glacier du Giétro': 'sion',
    'Silvrettagletscher': 'davos',
    'Claridenfirn': 'altdorf'
}

def load_data_model():
    """Load all necessary data files"""
    print("Loading data...")
    mass_balance_df = pd.read_csv(os.path.join(data_path, "mass_balance_hy.csv"))

    weather_data = {}
    for station in ['sion', 'davos', 'altdorf']:
        try:
            weather_data[f'{station}_temp'] = pd.read_csv(
                os.path.join(data_path, f"weather_dev9120_{station}_temp.csv")
            )
            weather_data[f'{station}_prec'] = pd.read_csv(
                os.path.join(data_path, f"weather_dev9120_{station}_prec.csv")
            )
            print(f"Successfully loaded {station} weather data")
        except FileNotFoundError as e:
            print(f"Warning: Could not load weather data for {station}: {e}")
            continue

    return mass_balance_df, weather_data

def prepare_monthly_data_model(glacier_name, mass_balance_df, weather_data):
    """Prepare data for a specific glacier using monthly deviations"""
    print(f"\nPreparing monthly deviation data for {glacier_name}...")
    station = glacier_weather_map.get(glacier_name)
    if station is None:
        raise ValueError(f"No weather station mapped for glacier: {glacier_name}")

    if f'{station}_temp' not in weather_data or f'{station}_prec' not in weather_data:
        raise ValueError(f"No monthly deviation data available for station: {station}")

    mb_data = mass_balance_df[mass_balance_df['glacier name'] == glacier_name][
        ['annual mass balance (mm w.e.)']
    ].copy().reset_index(drop=True)

    if len(mb_data) == 0:
        raise ValueError(f"No mass balance data found for glacier: {glacier_name}")

    temp_cols = ['may_td', 'june_td', 'july_td', 'august_td', 'september_td']
    precip_cols = ['october_pd', 'november_pd', 'december_pd', 'january_pd',
                  'february_pd', 'march_pd', 'april_pd']

    if glacier_name == 'Claridenfirn':
        temp_df = weather_data[f'{station}_temp'][temp_cols].copy()
        precip_df = weather_data[f'{station}_prec'][precip_cols].copy()

        if 'hydrological year' in weather_data[f'{station}_temp'].columns:
            temp_years = weather_data[f'{station}_temp']['hydrological year']
            precip_years = weather_data[f'{station}_prec']['hydrological year']
            temp_df = temp_df[~temp_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)
            precip_df = precip_df[~precip_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)
    else:
        temp_df = weather_data[f'{station}_temp'][temp_cols].reset_index(drop=True)
        precip_df = weather_data[f'{station}_prec'][precip_cols].reset_index(drop=True)

    max_index = min(len(mb_data) - 1, len(temp_df) - 1, len(precip_df) - 1)
    temp_df = temp_df.iloc[0:max_index+1]
    precip_df = precip_df.iloc[0:max_index+1]
    mb_data = mb_data.iloc[0:max_index+1]

    X = pd.concat([temp_df, precip_df], axis=1)
    y = mb_data['annual mass balance (mm w.e.)']

    data = pd.concat([X, y], axis=1).dropna()
    if len(data) == 0:
        raise ValueError("No data remaining after dropping missing values")

    X = data.drop('annual mass balance (mm w.e.)', axis=1)
    y = data['annual mass balance (mm w.e.)']

    return X, y, data

def prepare_seasonal_data_model(glacier_name, mass_balance_df, weather_data):
    """Prepare data for a specific glacier using seasonal deviations"""
    print(f"\nPreparing seasonal deviation data for {glacier_name}...")
    station = glacier_weather_map.get(glacier_name)
    if station is None:
        raise ValueError(f"No weather station mapped for glacier: {glacier_name}")

    if f'{station}_temp' not in weather_data or f'{station}_prec' not in weather_data:
        raise ValueError(f"No seasonal deviation data available for station: {station}")

    mb_data = mass_balance_df[mass_balance_df['glacier name'] == glacier_name][
        ['annual mass balance (mm w.e.)']
    ].copy().reset_index(drop=True)

    if len(mb_data) == 0:
        raise ValueError(f"No mass balance data found for glacier: {glacier_name}")

    temp_df = weather_data[f'{station}_temp'][['may_td', 'june_td', 'july_td', 'august_td', 'september_td']].copy()
    temp_df['summer_temp_dev'] = temp_df.mean(axis=1)

    precip_df = weather_data[f'{station}_prec'][['october_pd', 'november_pd', 'december_pd',
                                                   'january_pd', 'february_pd', 'march_pd', 'april_pd']].copy()
    precip_df['winter_precip_dev'] = precip_df.mean(axis=1)

    if glacier_name == 'Claridenfirn':
        if 'hydrological year' in weather_data[f'{station}_temp'].columns:
            temp_years = weather_data[f'{station}_temp']['hydrological year']
            precip_years = weather_data[f'{station}_prec']['hydrological year']
            temp_df = temp_df[~temp_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)
            precip_df = precip_df[~precip_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)

    max_index = min(len(mb_data) - 1, len(temp_df) - 1, len(precip_df) - 1)
    temp_df = temp_df.iloc[0:max_index+1]
    precip_df = precip_df.iloc[0:max_index+1]
    mb_data = mb_data.iloc[0:max_index+1]

    X = pd.concat([temp_df[['summer_temp_dev']], precip_df[['winter_precip_dev']]], axis=1)
    y = mb_data['annual mass balance (mm w.e.)']

    data = pd.concat([X, y], axis=1).dropna()
    if len(data) == 0:
        raise ValueError("No data remaining after dropping missing values")

    X = data.drop('annual mass balance (mm w.e.)', axis=1)
    y = data['annual mass balance (mm w.e.)']

    return X, y, data

def prepare_optimal_season_data_model(glacier_name, mass_balance_df, weather_data):
    """Prepare data for a specific glacier using optimal seasonal deviation variables"""
    print(f"\nPreparing optimal seasonal deviation data for {glacier_name}...")
    station = glacier_weather_map.get(glacier_name)
    if station is None:
        raise ValueError(f"No weather station mapped for glacier: {glacier_name}")

    if f'{station}_temp' not in weather_data or f'{station}_prec' not in weather_data:
        raise ValueError(f"No optimal seasonal deviation data available for station: {station}")

    mb_data = mass_balance_df[mass_balance_df['glacier name'] == glacier_name][
        ['annual mass balance (mm w.e.)']
    ].copy().reset_index(drop=True)

    if len(mb_data) == 0:
        raise ValueError(f"No mass balance data found for glacier: {glacier_name}")

    if 'opt_season_td' in weather_data[f'{station}_temp'].columns:
        temp_df = weather_data[f'{station}_temp'][['opt_season_td']].copy()
    else:
        summer_cols = ['may_td', 'june_td', 'july_td', 'august_td', 'september_td']
        available_cols = [col for col in summer_cols if col in weather_data[f'{station}_temp'].columns]
        if available_cols:
            temp_df = weather_data[f'{station}_temp'][available_cols].mean(axis=1).to_frame(name='opt_season_td')
        else:
            raise ValueError(f"No suitable temperature columns found for {station}")

    if 'opt_season_pd' in weather_data[f'{station}_prec'].columns:
        precip_df = weather_data[f'{station}_prec'][['opt_season_pd']].copy()
    else:
        winter_cols = ['october_pd', 'november_pd', 'december_pd', 'january_pd', 'february_pd', 'march_pd', 'april_pd']
        available_cols = [col for col in winter_cols if col in weather_data[f'{station}_prec'].columns]
        if available_cols:
            precip_df = weather_data[f'{station}_prec'][available_cols].mean(axis=1).to_frame(name='opt_season_pd')
        else:
            raise ValueError(f"No suitable precipitation columns found for {station}")

    if glacier_name == 'Claridenfirn':
        if 'hydrological year' in weather_data[f'{station}_temp'].columns:
            temp_years = weather_data[f'{station}_temp']['hydrological year']
            precip_years = weather_data[f'{station}_prec']['hydrological year']
            temp_df = temp_df[~temp_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)
            precip_df = precip_df[~precip_years.isin(['1993-1994', '1994-1995'])].reset_index(drop=True)

    max_index = min(len(mb_data) - 1, len(temp_df) - 1, len(precip_df) - 1)
    temp_df = temp_df.iloc[0:max_index+1]
    precip_df = precip_df.iloc[0:max_index+1]
    mb_data = mb_data.iloc[0:max_index+1]

    X = pd.concat([temp_df, precip_df], axis=1)
    y = mb_data['annual mass balance (mm w.e.)']

    data = pd.concat([X, y], axis=1).dropna()
    if len(data) == 0:
        raise ValueError("No data remaining after dropping missing values")

    X = data.drop('annual mass balance (mm w.e.)', axis=1)
    y = data['annual mass balance (mm w.e.)']

    X.columns = ['optimal_summer_temp_dev', 'optimal_winter_precip_dev']

    return X, y, data

def train_model(X, y, glacier_name, model_type):
    """Train and evaluate a linear regression model with random split"""
    print(f"\nTraining {model_type} model for {glacier_name}...")

    if len(X) < 5:
        raise ValueError(f"Not enough data points ({len(X)}) to train model for {glacier_name}")

    # Create pipeline with standardization
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0)  # Using Ridge for better stability
    )

    # Calculate cross-validation RMSE
    print("Calculating cross-validation RMSE...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X, y,
        cv=kf,
        scoring='neg_root_mean_squared_error'
    )
    avg_cv_rmse = -cv_scores.mean()
    cv_rmse_std = cv_scores.std()
    print(f"Cross-validation RMSE: {avg_cv_rmse:.2f} (±{cv_rmse_std:.2f})")


    # Random train-test split (70-30) with shuffling
    print("Split type: Random 70-30 split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate on both training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print(f"Training RMSE: {rmse_train:.2f}, R²: {r2_train:.4f}")
    print(f"Test RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")

    # Get coefficients and intercept for interpretation
    coefficients = model.named_steps['ridge'].coef_
    intercept = model.named_steps['ridge'].intercept_
    feature_names = X.columns

    print("\nModel Coefficients and Intercept:")
    for name, coef in zip(feature_names, coefficients):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {intercept:.4f}")

    return {
        'model': model,
        'cv_rmse': avg_cv_rmse,
        'cv_rmse_std': cv_rmse_std,
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'rmse_test': rmse_test,
        'r2_test': r2_test,
        'feature_names': feature_names,
        'coefficients': coefficients,
        'intercept': intercept,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test
    }

def create_glacier_model_pdf(glacier_name, results):
    """Create a single PDF for a glacier with all three models"""
    pdf_path = os.path.join(results_path, f"{glacier_name.replace(' ', '_')}_predictions.pdf")

    with PdfPages(pdf_path) as pdf:
        # Add title page
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 size
        ax.axis('off')
        ax.set_title(f"Glacier Mass Balance Model Results: {glacier_name}", fontsize=16, pad=20)
        pdf.savefig(fig)
        plt.close()

        # Add results for each model
        for model_type, result in results.items():
            # Create plot
            fig = plt.figure(figsize=(11.69, 8.27))

            # Create indices for plotting
            train_idx = np.arange(len(result['y_train']))
            test_idx = np.arange(len(result['y_train']), len(result['y_train']) + len(result['y_test']))

            # Plot training data as solid lines
            plt.plot(
                train_idx,
                result['y_train'],
                color='#1f77b4',  # Blue
                label='Actual (Train)',
                linewidth=2
            )
            plt.plot(
                train_idx,
                result['y_pred_train'],
                color='#ff7f0e',  # Orange
                label='Predicted (Train)',
                linestyle='--',
                linewidth=2
            )

            # Plot test data as solid lines
            plt.plot(
                test_idx,
                result['y_test'],
                color='#2ca02c',  # Green
                label='Actual (Test)',
                linewidth=2
            )
            plt.plot(
                test_idx,
                result['y_pred_test'],
                color='#d62728',  # Red
                label='Predicted (Test)',
                linestyle='--',
                linewidth=2
            )

            # Add vertical line to separate train and test data
            plt.axvline(x=len(result['y_train'])-0.5, color='gray', linestyle=':', linewidth=1)

            plt.ylabel("Annual Mass Balance [mm w.e.]", fontsize=12)

            # Add metrics to title
            plt.title(f"{model_type} Model\n"
                    f"Random 70-30 Split\n"
                    f"CV RMSE: {result['cv_rmse']:.2f} (±{result['cv_rmse_std']:.2f})\n"
                    f"Train RMSE: {result['rmse_train']:.2f}, Test RMSE: {result['rmse_test']:.2f}\n"
                    f"Train R²: {result['r2_train']:.4f}, Test R²: {result['r2_test']:.4f}", fontsize=14)

            # Add legend
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2, fontsize=10)

            # Add grid and improve layout
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Add text annotation for train/test separation
            plt.text(len(result['y_train'])/2, plt.ylim()[0] - (plt.ylim()[1]-plt.ylim()[0])*0.1,
                    'Training Data', ha='center', va='top', fontsize=10)
            plt.text(len(result['y_train']) + len(result['y_test'])/2, plt.ylim()[0] - (plt.ylim()[1]-plt.ylim()[0])*0.1,
                    'Test Data', ha='center', va='top', fontsize=10)

            pdf.savefig(fig)
            plt.close()

            # Create and add results table
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.axis('off')

            # Create table data
            table_data = [
                ["Metric", "Value"],
                ["Cross-Validation RMSE", f"{result['cv_rmse']:.2f} (±{result['cv_rmse_std']:.2f})"],
                ["Training RMSE", f"{result['rmse_train']:.2f}"],
                ["Training R²", f"{result['r2_train']:.4f}"],
                ["Test RMSE", f"{result['rmse_test']:.2f}"],
                ["Test R²", f"{result['r2_test']:.4f}"],
                ["", ""],
                ["Feature", "Coefficient"],
            ]

            # Add feature coefficients
            for name, coef in zip(result['feature_names'], result['coefficients']):
                table_data.append([name, f"{coef:.4f}"])

            # Add intercept
            table_data.append(["Intercept", f"{result['intercept']:.4f}"])

            # Create table
            table = ax.table(cellText=table_data, colWidths=[0.3, 0.3], loc='center', cellLoc='center')

            # Style table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Add title
            ax.set_title(f"{model_type} Model - Performance Metrics and Coefficients", fontsize=14, pad=20)

            pdf.savefig(fig)
            plt.close()

    print(f"Saved combined results for {glacier_name} to {pdf_path}")