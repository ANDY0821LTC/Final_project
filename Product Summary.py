# Databricks notebook source
import pandas as pd

pdf = pd.read_csv("/dbfs/FileStore/shared_uploads/ltc08211997ltc@gmail.com/all_record.csv")

# COMMAND ----------

pdf.display()

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt


pdf = pdf[['Brand', 'Price']]
mean_prices = pdf.groupby('Brand')['Price'].mean()
brand_stats = pdf.groupby('Brand')['Price'].describe()

# Plot a bar chart to visualize the mean prices of each brand
mean_prices.plot(kind='bar', xlabel='Brand', ylabel='Mean Price', title='Mean Prices by Brand')
plt.show()



# COMMAND ----------

print(brand_stats)

# COMMAND ----------

import numpy as np
from scipy import stats
from scipy.stats import ttest_ind

# COMMAND ----------

brand1_prices = pdf.loc[pdf['Brand'] == 'Brand1', 'Price']
brand2_prices = pdf.loc[pdf['Brand'] == 'Brand2', 'Price']
t_statistic, p_value = stats.ttest_ind(brand1_prices, brand2_prices)
print(f'T-test result: t-statistic = {t_statistic}, p-value = {p_value}')

# COMMAND ----------


