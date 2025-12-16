Frédéric van den Hurk 
2240 5344

# Analysis of eight Swiss glaciers

**Predicting glacier balance with summer temperature and winter precipitation.**

---

## Research question: How can we assess the impacts of weather components on glacier mass balance, and can we use these to predict glacier retreat?

---

## Setup

# Create environment
conda env create -f environment.yml
conda activate project-glaciers

---

## Usage

python main.py

Expected output: Figures and tables displaying data and results.

---

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
├── notebooks            # Jupyter notebooks for visualisation
│   └── data_notebooks   # Jupyter notebooks for data uploading
├── README.md
├── PROPOSAL.md          # Initial project proposal
└── environment.yml      # Dependencies
```
## Results
- Good RMSE for 3 of the glaciers (between 300 and 500 mm w.e.)
- Good R² values for the visualisation regressions (up to 0.77)

---

## Requirements
- Python 3.11
- pandas, numpy, matplotlib, statsmodels, scipy, scikit-learn