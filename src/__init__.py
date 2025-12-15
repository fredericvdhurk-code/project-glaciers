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



from .data_loader import (
    process_glaciers_list, process_length_change,
    process_mass_balance_hy, process_mass_balance_hy_eb,
    clean_weather_metadata, process_city_weather,
    calculate_hydrological_year_aggregates,
    calculate_seasonal_aggregates,
    calculate_weather_deviations_1961_1990,
    calculate_weather_deviations_1991_2020
    )

from .visualisation import (
    prepare_glacier_data, prepare_mass_balance_data,
    plot_glacier_cumulative_length_change,
    plot_glacier_length_change_bar, 
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
    save_glacier_plots_individually
    )

from .models import (
    load_data_model,
    prepare_monthly_data_model,
    prepare_seasonal_data_model,
    prepare_optimal_season_data_model,
    train_model,
    create_glacier_model_pdf
)
