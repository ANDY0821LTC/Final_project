# Databricks notebook source
asus_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/asus.csv")

# COMMAND ----------

asus_df.show()

# COMMAND ----------

tplink_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/tplink.csv")

# COMMAND ----------

tplink_df.display()

# COMMAND ----------

from pyspark.sql.functions import col, to_timestamp, col, lit

tplink_df = tplink_df.withColumnRenamed("CPU(No.of cores)", "CPU")

# COMMAND ----------

from pyspark.sql.functions import lit

tplink_df = tplink_df.select(*[c for c in tplink_df.columns if c != "_c0"])
tplink_df = tplink_df.withColumn("RAM", lit("Null"))
tplink_df = tplink_df.withColumn("Flash", lit("Null"))
tplink_df = tplink_df.select("Name", "Wifi", "Network Standard","Price","CPU","RAM","Flash","Antenna")
tplink_df.show()

# COMMAND ----------

linksys_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/linksys.csv")
linksys_df.show()

# COMMAND ----------

linksys_df = linksys_df.select("Name", "Wifi", "Network Standard","Price","CPU","RAM","Flash","Antenna")
linksys_df.display()

# COMMAND ----------

netgear_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/Netgear_record.csv")
netgear_df.show()

# COMMAND ----------

netgear_df = netgear_df.withColumnRenamed("Part No.", "Name")
netgear_df = netgear_df.withColumnRenamed("Wifi Speed", "Wifi")
netgear_df = netgear_df.withColumnRenamed("SRP($)", "Price")
netgear_df = netgear_df.withColumnRenamed("CPU(No.of cores)", "CPU")
netgear_df = netgear_df.withColumnRenamed("RAM(MB)", "RAM")
netgear_df = netgear_df.withColumnRenamed("Flash(MB)", "Flash")
netgear_df = netgear_df.withColumnRenamed("Description", "Network Standard")
netgear_df = netgear_df.select("Name", "Wifi", "Network Standard","Price","CPU","RAM","Flash","Antenna")
netgear_df.display()

# COMMAND ----------

combined_df = asus_df.union(tplink_df).union(linksys_df).union(netgear_df)
combined_df = combined_df.withColumnRenamed("Price($)", "Price")
combined_df = combined_df.withColumnRenamed("CPU(No.of cores)", "CPU")
combined_df = combined_df.withColumnRenamed("RAM(MB)", "RAM")
combined_df = combined_df.withColumnRenamed("Flash(MB)", "Flash")
combined_df.display()

# COMMAND ----------

import pandas as pd

pdf = combined_df.toPandas()
df_copy = spark.createDataFrame(pdf)

# COMMAND ----------

df_copy.show(180)

# COMMAND ----------

df_copy = df_copy.na.drop()
df_copy.show(180)

# COMMAND ----------

summarydf = df_copy.toPandas()
summarydf.display()

# COMMAND ----------

final_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/final_record.csv")
summarydf = final_df.toPandas()
summarydf.display()

# COMMAND ----------

summarydf['Name'] = summarydf['Name'].astype(str)
summarydf['Wifi'] = summarydf['Wifi'].astype(str)
summarydf['Network Standard'] = summarydf['Network Standard'].astype(str)
summarydf['Price'] = summarydf['Price'].astype(float)
summarydf['CPU'] = summarydf['CPU'].astype(int)
summarydf['RAM'] = summarydf['RAM'].astype(int)
summarydf['Flash'] = summarydf['Flash'].astype(int)
summarydf['Antenna'] = summarydf['Antenna'].astype(int)

# COMMAND ----------

summarydf.info()

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_le=pd.DataFrame(summarydf)
data_le['Name'] = labelencoder.fit_transform(data_le['Name'])
data_le

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
data_whole_ohe = onehotencoder.fit_transform(summarydf['Wifi']).toarray()

# COMMAND ----------

data_whole_ohe

# COMMAND ----------

data_dum = pd.get_dummies(summarydf)
pd.DataFrame(data_dum)

# COMMAND ----------

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


X = summarydf[['CPU', 'Wifi']] 
y = summarydf['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

model = LinearRegression()
model.fit(X_train, y_train)

# COMMAND ----------

y_predicted = model.predict(X_test)

# COMMAND ----------

lin_reg = LinearRegression().fit(X_train, y_train)
predictions = lin_reg.predict(X_test)

# COMMAND ----------

from sklearn.metrics import mean_squared_error

print('Mean squared error: ', mean_squared_error(y_test, predictions))
print('Coefficients: ', lin_reg.coef_)

# COMMAND ----------

pd.DataFrame(zip(X.columns, model.coef_))

# COMMAND ----------

print(r2_score(y_test, y_predicted))

# COMMAND ----------

sns.regplot(x=y_test, y=y_predicted)

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_recall_fscore_support


# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
encoded = enc.fit_transform(summarydf[['Network Standard']]).toarray()
summarydf = summarydf.join(pd.DataFrame(encoded, columns=enc.categories_[0]))
summarydf = summarydf.drop('Network Standard', axis=1)

# COMMAND ----------

summarydf 

# COMMAND ----------



# COMMAND ----------

netgear_price_df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/ltc08211997ltc@gmail.com/Netgear_price_change.csv")
netgear_price_df.show()

# COMMAND ----------

df = netgear_price_df.toPandas()

# COMMAND ----------

df.display()

# COMMAND ----------

df.columns = df.iloc[0]  
df = df.drop(df.index[0])
df.display()

# COMMAND ----------

df['Model'] = df['Model'].astype(str)

# COMMAND ----------

cols = list(df.columns[1:33]) 
for col in cols:
  df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# COMMAND ----------

df.info()

# COMMAND ----------

df.display()

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.ar_model import AutoReg


# COMMAND ----------

df = df.T
df

# COMMAND ----------

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

#df.set_index('Model', inplace=True)

time_series_df = []
for model in df.columns:
  
  print(f"Modelling for {model}")
  time_series_df.append(model)
  # Select timeseries 
  y = df[model]  

  # Split into train and test
  train, test = y[:24], y[24:]

  # Fit SARIMAX model 
  order = (1,1,1)
  model = SARIMAX(train, order=order)
  fitted = model.fit()

  # Predict and calculate MSE
  forecast = fitted.predict(start=len(train), end=len(train)+len(test)-1)
  mse = mean_squared_error(test, forecast)

  print(f"MSE for {model}: {mse}")
  time_series_df.append(mse)

# COMMAND ----------

time_series_df

# COMMAND ----------

df1 = pd.DataFrame(time_series_df)
df1

# COMMAND ----------

new_df = pd.DataFrame({
    'name': df1.iloc[1::2, 0].values,
    'mse': df1.iloc[::2, 0].values
})

# Reset the index of the new DataFrame
new_df.reset_index(drop=True, inplace=True)

# Print the new DataFrame
print(new_df)

# COMMAND ----------


