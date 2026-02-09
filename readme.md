#  Time Series Forecasting 

## Business context
Inventory management is a major challenge for retailers. Understocking leads to lost sales, while overstocking increases capital expenditures.
This project aims to build a **time series** model capable of predicting future sales based on five years of historical data, in order to optimize logistics and human resource planning.

## Stack
* **Language :** Python 3.x
* **Libraries :** Pandas, Matplotlib, Scikit-learn.
* **Modelisation :** Prophet (Facebook Core Data Science).
* **Metrics :** MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error).

## Methodology

### 1. Exploratory Analysis and Decomposition
* Analysis of the stationary and components of the time series.
* Additive decomposition to isolate :
    * **Trend :** Linear growth observed over 5 years.
    * **Seasonality :** strong weekly and annual patterns.

### 2. Modelisation with Prophet
The Prophet algorithm is used for its robustness in the face of missing data and its ability to handle multiple seasonalities without complex engineering.
* **Configuration:** Enable annual seasonality (`yearly_seasonality=True`).
* **Split:** Training over 4 years and 9 months, Testing over the last 3 months.
* 
##  Insights & Recommandations

Analysis of the model's components revealed clear consumption patterns:

| Component | Observation | Actionable Recommendation |
| :--- | :--- | :--- |
| **Weekly** | Major peak on Sunday and Saturday. Low on Monday. | **Staffing:** Increase staffing levels on weekends. Schedule inventory and restocking on Mondays. |
| **Annual** | Peaks in **July/August** (Summer) and **December** (Holidays). | **Logistics:** Secure supplies 2 months before these peaks to avoid stockouts. |
| **Trend** | Steady growth. | **Strategy:** Plan an increase in the overall purchasing budget for year N+1. |

##  Performance
The model manages to track trends and seasonal peaks with sufficient accuracy for operational planning.

* **MAPE (Percentage Error):** ~12% 
* The model accurately captures the dynamics of year-end peaks.

