Frédéric van den Hurk 
2240 5344

# Analysis of eight Swiss glaciers

**Predicting glacier balance with summer temperature and winter precipitation.**

## Research question: How can we assess the impacts of weather components on glacier mass balance, and can we use these to predict glacier retreat?
Using Python, I analyse time-series data to visualise trends by means of figures, quantify relationships with linear regressions and build a machine learning model that predicts annual mass balance. 

---

# Setup

## Access the directory
```
cd project-glaciers
```
## Create environment
```
conda env create -f environment.yml
conda activate project-glaciers
```
## Usage
```
python main.py
```
Expected output: Figures and tables in PDF files displaying data and result.

## Project Structure
```
project-glaciers/
├── main.py              # Main entry point
├── src/                 # Source code
│   ├── __init__.py      # Functions to import from src
│   ├── data_loader.py   # Data loading/preprocessing
│   ├── models.py        # ML Model training and visualisation
│   └── visualisation.py # Visualisation of data and relationships
├── results/             # Output plots and metrics in pdf files
│   └── model_results    # ML model plots and metrics in pdf files
├── data/                # Processed data
│   └── raw_data         # Uploaded raw data
├── README.md
├── .gitignore           
├── PROPOSAL.md          # Initial project proposal
└── environment.yml      # Dependencies
```
## Results
- Good RMSE for 3 of the glaciers (between 300 and 500 mm w.e.)
- Good R² values for the visualisation regressions (up to 0.77)

## Requirements
- Python 3.11
- pandas, numpy, matplotlib, statsmodels, scipy, scikit-learn
