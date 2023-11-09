# Databricks notebook source
import requests
import time
import pandas as pd

url = 'https://odinapi.asus.com/recent-data/apiv2/SeriesFilterResult?SystemCode=asus&WebsiteCode=hk&ProductLevel1Code=networking-iot-servers&ProductLevel2Code=wifi-routers&PageSize=20&PageIndex=2&CategoryName=&SeriesName=ASUS-WiFi-Routers&SubSeriesName=&Spec=&SubSpec=&PriceMin=&PriceMax=&Sort=Recommend&siteID=www&sitelang='

header = {"user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'}
response = requests.get(url, headers=header)

df = []
data = response.json()

for product in data['Result']['ProductList']:
  name = product['Name'].replace('<h2>','').replace('</h2>','')
  wifi = product['ModelSpec'].split(' ')[0]
  df.append([name,wifi])

df1 = pd.DataFrame(df,columns=['Name','Wifi'])


# COMMAND ----------

df1

# COMMAND ----------

df2 = pd.DataFrame(df1)

# COMMAND ----------

type(df2)

# COMMAND ----------

import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np

url = 'https://www.asus.com/hk/networking-iot-servers/wifi-routers/asus-wifi-routers/filter?Series=ASUS-WiFi-Routers'

header = {"user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'}
response = requests.get(url, headers=header)
http_string = response.text
html = BeautifulSoup(http_string, 'html.parser')

df1 = []
router = html.find_all('div',{"id":"productListContainer"})

name= html.select('div.ProductCardNormalStore2__viewBox__3NgQ2 > a > h2')
wifi_speed = html.select('div.itemModelSpec.ProductCardNormalStore2__itemModelSpec__cg96R > ul > li:nth-child(2)')


# COMMAND ----------

wifi_array = np.array(wifi_speed)
routers_array = np.array(name)

# COMMAND ----------

df = pd.DataFrame(routers_array,columns=['Name'])
df['Wifi'] = pd.DataFrame(wifi_array,columns=['Wifi'])
asus_df = df

# COMMAND ----------

asus_df 

# COMMAND ----------

asus = pd.concat([df2, asus_df]).reset_index(drop=True) 

# COMMAND ----------

asus

# COMMAND ----------

asus.at[2, 'Wifi'] = 'AX6000'
asus.at[8, 'Wifi'] = 'AX1800'
asus.at[17, 'Wifi'] = 'AX3000'
asus.at[18, 'Wifi'] = 'AX5400'
asus.at[19, 'Wifi'] = 'AX6000'
asus.at[20, 'Wifi'] = 'AX4200'
asus.at[21, 'Wifi'] = 'AX4200'
asus.at[22, 'Wifi'] = 'AX5700'
asus.at[23, 'Wifi'] = 'AXE7800'
asus.at[24, 'Wifi'] = 'AX1800'
asus.at[25, 'Wifi'] = 'AX1800'
asus.at[26, 'Wifi'] = 'AX1800'
asus.at[27, 'Wifi'] = 'AX1800'
asus.at[28, 'Wifi'] = 'AX5400'
asus.at[29, 'Wifi'] = 'AX5700'
asus.at[30, 'Wifi'] = 'AX5400'
asus.at[31, 'Wifi'] = 'AX1800'
asus.at[32, 'Wifi'] = 'AX3000'
asus.at[33, 'Wifi'] = 'AX6000'
asus.at[34, 'Wifi'] = 'AXE6000'
asus.at[35, 'Wifi'] = 'AX11000'
asus.at[36, 'Wifi'] = 'AX6000'

# COMMAND ----------

asus

# COMMAND ----------

display(asus)

# COMMAND ----------


