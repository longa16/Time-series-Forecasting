# Time Series Forecasting

## Project goal
This project aims to forecast future sales from historical daily sales data. The objective is to understand trends and seasonality in customer demand in order to support logistics planning, stock forecasting, and resource optimization.

The dataset used is the file `train.csv`, which contains daily sales data for multiple stores and products, with the following columns:

- `date`
- `store`
- `item`
- `sales`

## Data and context
The project analyzes a daily time series over several years. The main steps are:

1. Loading the dataset and converting the `date` column to datetime.
2. Aggregating sales by day to obtain the overall daily sales series.
3. Visual analysis of trend and seasonal variations.
4. Modeling with Prophet to forecast the next 90 days.
5. Evaluating model performance using MAE and MAPE.

## Tech stack
- Python 3.13
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Prophet

## Project workflow

### 1. Exploratory analysis
The first step is to inspect the data, verify its structure, and visualize daily sales. The aggregated series makes it possible to observe overall behavioral patterns over time.

### 2. Preparation for modeling
The model is built from a Prophet-compatible dataframe with:

- `ds`: date
- `y`: total sales per day

The data split is as follows:

- Training: all observations before the last 90-day period
- Test: the last 90 days, used to evaluate forecast quality

### 3. Modeling with Prophet
The model is configured with annual seasonality enabled and without daily seasonality.

```python
model = Prophet(yearly_seasonality=True, daily_seasonality=False)
model.fit(train)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

### 4. Performance evaluation
The project compares model predictions with actual sales in the test window using:

- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

## Observed results
The analysis and forecasts show that the series presents:

- a generally positive trend,
- marked seasonal variation,
- peaks during certain periods of the year,
- good ability of the model to follow the main demand movements.

The project notebook includes:

- daily total sales,
- a comparison between actual and predicted values over the last 3 months,
- the model's trend and seasonality components.

## Repository structure
- `train.csv`: historical sales data
- `Untitled.ipynb`: exploration, modeling, and evaluation notebook
- `main.py`: minimal project entry point
- `pyproject.toml`: Python project configuration and dependencies

## Installation

Using `uv` (recommended):

```bash
uv venv
uv pip install -e .
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -e .
```

## Usage
The project is mainly used through the Jupyter notebook in `Untitled.ipynb`. It contains all steps for cleaning, visualization, modeling, and validation.

To launch the notebook with `uv`:

```bash
uv run jupyter notebook
```

## Conclusion
This project illustrates a standard time series forecasting workflow applied to retail sales, using a robust Prophet model to capture trends and seasonal effects without requiring overly complex feature engineering.

