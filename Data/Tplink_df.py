# Databricks notebook source
import pandas as pd
from bs4 import BeautifulSoup
import requests

new_df1 = []
for page in range(1,3):
  url = "https://hk.store.tp-link.com/collections/all?sort_by=title-ascending&filter.v.price.gte=&filter.v.price.lte=&filter.p.tag=Wi-Fi+6&filter.p.tag=Wi-Fi+6E&filter.p.tag=Wi-Fi+7&page=" + str(page)

  header = {"user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'}
  response = requests.get(url, headers=header)
  http_string = response.text
  html = BeautifulSoup(http_string, 'html.parser')

  router_product = html.find_all('product-item',{'class':"product-item"})

  for item in router_product:
    name = item.select_one('div.product-item-meta').text.split()[:2]
    price = item.select_one('div.price-list.price-list--centered > span:nth-child(1)').text.replace('銷售價格','').replace('\n','').replace('從','')

    new_df1.append([name,price])


df1 = pd.DataFrame(new_df1,columns=['Name','Price'])

df1.to_csv('tplink.csv')

# COMMAND ----------

df1

# COMMAND ----------

df1['Name'] = df1['Name'].apply(lambda x: " ".join(x))

# COMMAND ----------

df1

# COMMAND ----------

# To get model name with wifi from Archer series
import pandas as pd
from bs4 import BeautifulSoup
import requests

tp_df1 = []

url = 'https://www.tp-link.com/zh-hk/home-networking/wifi-router/?filterby=6271%7C6093%7C5730'

header = {"user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'}
response = requests.get(url, headers=header)
http_string = response.text
html = BeautifulSoup(http_string, 'html.parser')

router = html.find_all('li',{'class':"tp-product-item"})

cpu = html.select('div.tp-product-spec')

for item in router:
  name = item.select_one('div.tp-product-model').text
  wifi = item.select_one('h3.tp-product-title').text.replace('雙頻WiFi','').split(' ')[0].split('完整家庭Mesh')[0]
  tp_df1.append([name,wifi])


df3 = pd.DataFrame(tp_df1,columns=['Name','Wifi'])

# COMMAND ----------

df3

# COMMAND ----------

# To get model name and wifi from Deco
import pandas as pd
from bs4 import BeautifulSoup
import requests

tp_df1 = []

url = 'https://www.tp-link.com/zh-hk/home-networking/deco/'

header = {"user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.82 Safari/537.36'}
response = requests.get(url, headers=header)
http_string = response.text
html = BeautifulSoup(http_string, 'html.parser')

router = html.find_all('li',{'class':"tp-product-item"})

cpu = html.select('div.tp-product-spec')

for item in router:
  name = item.select_one('div.tp-product-model').text
  wifi = item.select_one('h3.tp-product-title').text.replace('雙頻WiFi','').split(' ')[0].split('完整家庭Mesh')[0]
  tp_df1.append([name,wifi])


df2 = pd.DataFrame(tp_df1,columns=['Name','Wifi'])

# COMMAND ----------

df2

# COMMAND ----------

data = [df3,df2]
df4 = pd.concat(data,ignore_index=True,sort=False)

# COMMAND ----------

df4

# COMMAND ----------

tp_df = df1.join(df4.set_index('Name'), on='Name')

# COMMAND ----------

tp_df 

# COMMAND ----------

tplink_df = tp_df[:37]

# COMMAND ----------

tplink_df .at[4, 'Wifi'] = 'AX1800'
tplink_df .at[12, 'Wifi'] = 'AX7800'
tplink_df .at[15, 'Wifi'] = 'AX6600'
tplink_df .at[15, 'Wifi'] = 'AX6600'
tplink_df .at[16, 'Wifi'] = 'AX1800'
tplink_df .at[17, 'Wifi'] = 'AX1800'
tplink_df .at[18, 'Wifi'] = 'AX3000'
tplink_df .at[19, 'Wifi'] = 'AX3000'
tplink_df .at[20, 'Wifi'] = 'AXE5400'
tplink_df .at[24, 'Wifi'] = 'AX1800'
tplink_df .at[26, 'Wifi'] = 'AX3000'
tplink_df .at[27, 'Wifi'] = 'AX3000'
tplink_df .at[33, 'Wifi'] = 'AX6000'

# COMMAND ----------

tplink_df

# COMMAND ----------



# COMMAND ----------

type(tplink_df)

# COMMAND ----------



# COMMAND ----------


