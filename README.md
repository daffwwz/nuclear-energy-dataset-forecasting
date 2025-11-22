# nuclear-energy-dataset-forecasting
This repository should include only time series forecasting methods for analysis.

## Tools
Python, Matplotlib, Seaborn, Pandas, Sci-Kit Learn, XGBoost, ARIMA, Auto-ARIMA, SARIMA

### XGBoost
XGBoost model is commonly used for classification and regression gradient boosting problem, but it also can be used for time series forecasting with some added features. The notebook code first read the CSV file after EDA process from the `nuclear-energy-dataset-anaylsis` repository. After that, new features such as `Year`, `Quarter`, `Month`, `DayOfWeek`, `DayOfYear`, `lag_7`, `rolling_mean_1`, `rolling_std_1`, `rolling_mean_7`, `rolling_std_7`, `rolling_mean_30`, `rolling_std_30`, and y normalization using standard scaler is added. Data then splitted into train and test and splitted using `TimeSeriesSplit()` for cross-validation. Model then fitted into the XGBoost for training and testing, resulted in the plot below and these evaulation metrics:
1. RMSE from the test prediction: 108.80
2. RMSE from the train prediction: 108.19
3. error tolerance: 10.81941211946799
4. diff: 0.6098441612242311
5. The model generalizes well to unseen data. (shows the indication that the model is not overfitting or underfitting)

### ARIMA (Auto Regressive Integrated Moving Average)
ARIMA is a class of models that forecast a time series data based on the lags (Auto Regressive), differencing (Integrated), the residual errors (Moving Average). This model assumes the data is stationary or not evolving over time, so we need to check whether the data is stationary using the Augmented Dickey-Fuller method, which measures the p-value for the hypothesis test. The ARIMA model has parameters p, d, and q, which are determined using the autocorrelation function (ACF) for p, differencing for d, and the partial autocorrelation function (PACF) for q. Once the parameters are known, the data is divided into training and testing sets and is ready to be trained using the model. The following evaluation metrics are used in this forecasting model:
1. Mean Absolute Error (MAE): 0.2242
2. Mean Absolute Percentage Error (MAPE): 0.3463
3. Root Mean Squared Error (RMSE): 0.2769

