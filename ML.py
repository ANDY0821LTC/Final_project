# Databricks notebook source
import pandas as pd
pdf = pd.read_csv("/dbfs/FileStore/shared_uploads/ltc08211997ltc@gmail.com/all_record.csv")

# COMMAND ----------

pdf.display()

# COMMAND ----------

pdf['Name'] = pdf['Name'].astype(str)
pdf['Wifi'] = pdf['Wifi'].astype(str)
pdf['Wifi Speed'] = pdf['Wifi Speed'].astype(int)
pdf['Network Standard'] = pdf['Network Standard'].astype(str)
pdf['Price'] = pdf['Price'].astype(float)
pdf['CPU'] = pdf['CPU'].astype(int)
pdf['RAM'] = pdf['RAM'].astype(int)
pdf['Flash'] = pdf['Flash'].astype(int)
pdf['Antenna'] = pdf['Antenna'].astype(int)
pdf['Number of Satellites'] = pdf['Number of Satellites'].astype(int)

# COMMAND ----------

pdf.info()

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming your dataset is loaded into a pandas DataFrame called 'data'
X = pdf[['Wifi Speed', 'CPU', 'RAM', 'Flash','Number of Satellites']]  # Feature columns
y = pdf['Price']  # Target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared Score:', r2)

# COMMAND ----------

import matplotlib.pyplot as plt

# Assuming you have already trained the Linear Regression model and obtained predictions (y_pred) and actual values (y_test)

# Plot the actual prices vs. predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual Prices vs. Predicted Prices')
plt.show()

# COMMAND ----------

import seaborn as sns

sns.regplot(x=y_test, y=y_pred)

# COMMAND ----------


