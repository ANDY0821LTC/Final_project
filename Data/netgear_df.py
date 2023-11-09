# Databricks notebook source
df1 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/Netgear_price_change.csv")

# COMMAND ----------

df1.show()

# COMMAND ----------



# COMMAND ----------

df2 = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/netgear_record.csv")

# COMMAND ----------

df2.show()

# COMMAND ----------


