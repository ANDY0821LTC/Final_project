# Databricks notebook source
df= spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/Netgear_price_change.csv")

df = df.toPandas()
#df.display()
df.columns = df.iloc[0]  
df = df.drop(df.index[0])
df.display()

# COMMAND ----------

df['Model'] = df['Model'].astype(str)
df['SRP($)'] = df['SRP($)'].astype(int)

df3 = df.iloc[:,2:]

# COMMAND ----------

df3

# COMMAND ----------

df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/price.csv")
df = df.toPandas()
df

# COMMAND ----------

import pandas as pd

model_names = df['Model']
numeric_data = df.drop('Model', axis=1)


# Convert the columns containing price change to numeric
numeric_data = numeric_data.apply(pd.to_numeric)

# Interpolate the missing values using linear interpolation
interpolated_data = numeric_data.interpolate(method='linear', axis=1)

# Merge the interpolated data with the model names
interpolated_df = pd.concat([model_names, interpolated_data], axis=1)

# Print the DataFrame with interpolated values
interpolated_df.display()

# COMMAND ----------

import pandas as pd

df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/stock.csv")
df1 = df1.toPandas()
df1 = df1.iloc[:, :-1]
df1.columns = df1.iloc[0]  
df1 = df1.drop(df1.index[0])
df1.display()

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# COMMAND ----------

# Extract the sales data for a specific model (e.g., RAX10)
model_data = interpolated_df[interpolated_df['Model'] == 'RAX10']
# Convert the sales data to float
model_data = model_data.iloc[:, 2:].astype(float)  

# COMMAND ----------

# Convert column names to datetime format
model_data.columns = pd.to_datetime(model_data.columns, format='%b-%y')

# COMMAND ----------

sales_ts = pd.Series(model_data.values.flatten())

# COMMAND ----------

# Perform ARIMA forecasting
model_arima = ARIMA(sales_ts, order=(1, 1, 1))  # Specify the order(p, d, q)
model_arima_fit = model_arima.fit()
arima_forecast = model_arima_fit.predict(start=len(sales_ts), end=len(sales_ts) + 30)

# COMMAND ----------

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(arima_forecast.index, arima_forecast.values, label='ARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Time Series Forecasting')
plt.legend()
plt.show()

# COMMAND ----------

# Perform SARIMA forecasting
model_sarima = SARIMAX(sales_ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Specify the order(p, d, q) and seasonal_order(P, D, Q, S)
model_sarima_fit = model_sarima.fit()
sarima_forecast = model_sarima_fit.predict(start=len(sales_ts), end=len(sales_ts) + 30)

# COMMAND ----------

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(sarima_forecast.index, sarima_forecast.values, label='SARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Time Series Forecasting')
plt.legend()
plt.show()

# COMMAND ----------

# Perform Prophet forecastingimage.png
df_prophet = pd.DataFrame({'ds': model_data.columns, 'y': sales_ts.values})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], format='%b-%y')
model_prophet = Prophet()
model_prophet.fit(df_prophet)
future_dates = model_prophet.make_future_dataframe(periods=12, freq='M')  # Forecast for the next 12 months
prophet_forecast = model_prophet.predict(future_dates)

# COMMAND ----------

# Plot the forecasts
plt.figure(figsize=(10, 6))
plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual')
plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'], label='Prophet')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Time Series Forecasting')
plt.legend()
plt.show()

# COMMAND ----------


