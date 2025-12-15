Frédéric van den Hurk 
2240 5344

# Glacier Mass Balance Analysis with Weather Data

**Predicting Glacier Mass Balance using Summer Temperature and Winter Precipitation Data.**

---

## Overview
This project aims to analyze the mass balance of glaciers by correlating weather data (temperature and precipitation) with glacier volume changes. This provides insights into how climate change and weather variability affect glacier health over time. 
Linear regression ML models are used to predict glacier mass balance.
The analyses are performed on 8 different glaciers of the swiss alps with help of weather data from 3 different weather stations spread across the relevant regions

---

## Features
- **Data Integration**: Combines glacier mass balance data with historical time-series weather datasets.
- **Visualization**: Generates plots of mass balance, weather data and relationships.
- **Statistical Analysis**: Builds OLS linear regression models to quantify the effects of weather components on glacier mass balance.
- **ML Analysis**: Uses linear regression ML models to predict glacier mass balance using the same weather components as in the ordinary OLS model.

---

## Technologies & Tools
- **Programming Language**: Python
- **Libraries**:
  - `pandas` for data manipulation
  - `numpy` for numerical computations
  - `matplotlib` for visualization
  - `statsmodels` for statistical modelling
  - `sklearn` for ML modelling
- **Data Sources**:
  - Glacier data: GLAMOS website
  - Weather data: Meteoswiss homogenous weather time-series 

- **Key Results**:
  - RMSE: range from 390 to 1000
  - R²: up to 0.77

---

## Installation
1. **Clone the repository**:
   ```bash
   git clone [https://github.com/fredericvdhurk-code/project-glaciers]
   cd project-glaciers

---

## Run
1. **Run main.py**
    see code functions in src folder
    see the results as pdf documents in the results folder