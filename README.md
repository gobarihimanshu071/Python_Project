# Rainfall Analysis and Prediction System

A comprehensive Python-based system for analyzing and predicting rainfall patterns across different subdivisions of India using machine learning techniques.

## Overview

This project provides tools for:
- Loading and cleaning rainfall data
- Visualizing rainfall patterns over time
- Analyzing monthly rainfall distributions
- Training machine learning models to predict rainfall patterns
- Generating rainfall forecasts using ARIMA models
- Creating various visualizations including heatmaps and 3D plots

## Features

- **Data Processing**
  - Automated data cleaning and preprocessing
  - Handling missing values using median imputation
  - Feature engineering for rainfall classification

- **Visualization**
  - Yearly rainfall trends
  - Monthly rainfall distribution analysis
  - Correlation heatmaps
  - 3D rainfall visualization
  - Average rainfall heatmaps

- **Machine Learning**
  - Multiple model implementations (Random Forest, Gradient Boosting, XGBoost)
  - Model evaluation and comparison
  - Feature importance analysis
  - Rainfall classification

- **Forecasting**
  - ARIMA-based rainfall forecasting
  - Long-term rainfall predictions

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- statsmodels

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels
```

## Usage

The main script `rainfall.py` contains several functions for different analyses:

```python
# Load and clean data
data = load_and_clean("Sub_Division_IMD_2017.csv")

# Visualize yearly trends
plot_yearly_trend(data, "KERALA")

# Analyze monthly patterns
plot_monthly_spread(data)

# Train machine learning models
train_rain_model(data)

# Generate forecasts
forecast_annual_rainfall_arima(data, forecast_years=10)
```

## Data

The project uses the `Sub_Division_IMD_2017.csv` dataset, which contains:
- Monthly rainfall data
- Annual rainfall totals
- Subdivision information
- Yearly records

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
