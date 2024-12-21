import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import math
import geopandas as gpd
from matplotlib.colors import ListedColormap
import unicodedata

# Import File CSV
df_customer = pd.read_csv('data/df_customer_clean.csv')
df_order = pd.read_csv('data/df_order_clean.csv')
df_order_items = pd.read_csv('data/df_order_items_clean.csv')
df_order_payments = pd.read_csv('data/df_order_payments_clean.csv')
df_product = pd.read_csv('data/df_product_clean.csv')
df_sellers = pd.read_csv('data/df_sellers_clean.csv')
df_geolocation = pd.read_csv('data/df_geolocation_clean.csv')

## Convert datetime
df_order['order_purchase_timestamp'] = pd.to_datetime(df_order['order_purchase_timestamp'])
df_order_items['shipping_limit_date'] = pd.to_datetime(df_order_items['shipping_limit_date'])

## Kelompok order_id
kelompok_cancel_unav = pd.concat([df_order[df_order['order_status']=='canceled']['order_id'],
                                df_order[df_order['order_status']=='unavailable']['order_id']])

kelompok_seller = pd.concat([df_order[df_order['order_status']=='delivered']['order_id'],
                           df_order[df_order['order_status']=='invoiced']['order_id'],
                             df_order[df_order['order_status']=='shipped']['order_id'],
                             df_order[df_order['order_status']=='processing']['order_id'],
                             df_order[df_order['order_status']=='created']['order_id'],
                             df_order[df_order['order_status']=='approved']['order_id']])

kelompok_customer = pd.concat([df_order[df_order['order_status']=='delivered']['order_id'],
                             df_order[df_order['order_status']=='shipped']['order_id'],
                             df_order[df_order['order_status']=='invoiced']['order_id'],
                             df_order[df_order['order_status']=='processing']['order_id'],
                             df_order[df_order['order_status']=='created']['order_id'],
                             df_order[df_order['order_status']=='approved']['order_id']])

## Kumpulan Fungsi

### Menghapus Aksen
def remove_accents(input_str):
  
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([char for char in nfkd_form if not unicodedata.combining(char)])

### Mendapatkan product
def find_prod(prod_select:int, x:list):

    if prod_select in x:
      return True
    else:
      return False

### Mendapatkan pivot_seller dan pivot_order
def create_pivot_seller_and_order(df_order_items: pd.DataFrame,
                                  df_product: pd.DataFrame,
                                  df_order_payments: pd.DataFrame,
                                  df_order: pd.DataFrame) -> tuple:

    """
    Fungsi ini bertujuan untuk menghasilkan Data Frame pivot_seller dan pivot_order

    Parameters:
        df_order_items (pandas DataFrame): Data Frame df_order_items
        df_product (pandas DataFrame): Data Frame df_product
        df_order_payments (pandas DataFrame): Data Frame df_order_payments
        df_order (pandas DataFrame): Data Frame df_order

    Returns:
        tuple(pivot_seller, pivot_order):
        Data Frame pivot_seller dan  Data Frame pivot_order        
    """

    def create_pivot_seller(df_order_items, df_product):

        df_temp = pd.merge(df_order_items, df_product, on='product_id', how='inner')
        pivot_seller = df_temp[df_temp['order_id'].isin(kelompok_seller)].groupby(by='seller_id').agg({
                                                                    'price': ['sum','mean','max', 'min'],
                                                                    'freight_value': ['sum','mean','max', 'min'],
                                                                    'product_id' : lambda x: list(x),
                                                                    'product_category_name' : lambda x: list(x),
                                                                    'shipping_limit_date' : lambda x: list(x)
                                                                    }).sort_values(by=('price','sum'), ascending=False)
        pivot_seller.columns = ['_'.join(col).strip() for col in pivot_seller.columns.values]
        
        return pivot_seller
    
    def create_pivot_order(df_order_payments, df_order):

        df_temp = df_order_items.drop(columns=['order_item_id','shipping_limit_date','seller_id'])
        df_temp = pd.merge(df_temp, df_product, on='product_id', how='inner')
        df_temp = df_temp[df_temp['order_id'].isin(kelompok_customer)].groupby(by='order_id').agg({
                                                                    'price': ['sum','mean','max', 'min'],
                                                                    'freight_value': ['sum','mean','max', 'min'],
                                                                    'product_id' : lambda x: list(x),
                                                                    'product_category_name' : lambda x: list(x)
                                                                    }).sort_values(by=('price','sum'), ascending=False)
        df_temp.columns = ['_'.join(col).strip() for col in df_temp.columns.values]
        
        pivot_order = df_order_payments[df_order_payments['order_id'].isin(kelompok_customer)].groupby(by='order_id').agg({
                                                            'payment_value': ['mean','max', 'min','sum'],
                                                            }).sort_values(by=('payment_value','sum'), ascending=False)
        pivot_order.columns = ['_'.join(col).strip() for col in pivot_order.columns.values]
        pivot_order = pd.merge(pivot_order, df_temp, on='order_id', how='inner')
        
        return pivot_order
    
    return create_pivot_seller(df_order_items, df_product), create_pivot_order(df_order_payments, df_order)

### Mendapatkan df_sellers_merged dan df_customer_merged
def create_df_sellers_and_customer_merged(pivot_seller: pd.DataFrame,
                                          df_sellers: pd.DataFrame,
                                          pivot_order: pd.DataFrame,
                                          df_order: pd.DataFrame,
                                          df_customer: pd.DataFrame) -> tuple:
    
    """
    Fungsi ini bertujuan untuk menghasilkan Data Frame df_sellers_merged dan df_customer_merged

    Parameters:
        pivot_seller (pandas DataFrame): Data Frame pivot_seller
        df_sellers (pandas DataFrame): Data Frame df_sellers
        pivot_order (pandas DataFrame): Data Frame pivot_order
        df_order (pandas DataFrame): Data Frame df_order
        df_customer (pandas DataFrame): Data Frame df_customer

    Returns:
        tuple(df_sellers_merged, df_customer_merged):
        Data Frame df_sellers_merged dan Data Frame df_customer_merged        
    """

    df_sellers_merged = pd.merge(pivot_seller, df_sellers, on='seller_id', how='inner')

    df_order_merged = pd.merge(pivot_order, df_order, on='order_id', how='inner')
    cols = df_order_merged.columns.tolist()
    cols.insert(1, cols.pop(cols.index('customer_id')))
    df_order_merged = df_order_merged[cols]

    df_customer_merged = pd.merge(df_order_merged, df_customer, on='customer_id', how='inner')

    return df_sellers_merged, df_customer_merged

### Mendapatkan monthly_summary
def create_monthly_summary(df_customer_merged: pd.DataFrame) -> pd.DataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan Data Frame monthly_summary

    Parameters:
        df_customer_merged (pandas DataFrame): Data Frame df_customer_merged

    Returns:
        monthly_summary: Data Frame monthly_summary        
    """

    ## Ekstrak tahun dan bulan
    df_test = df_customer_merged.copy()
    df_test['year_month'] = df_customer_merged['order_purchase_timestamp'].dt.to_period('M')

    ## Kelompokkan berdasarkan year_month dan jumlahkan payment_value_sum
    monthly_summary = df_test.groupby('year_month')['payment_value_sum'].sum().reset_index()

    ## Ubah kembali year_month ke string (opsional)
    monthly_summary['year_month'] = monthly_summary['year_month'].astype(str)

    monthly_summary = monthly_summary.drop(index=monthly_summary[monthly_summary['year_month'] == '2018-09'].index)

    return monthly_summary

### Mendapatkan monthly_transactions
def create_monthly_transactions(df_order_items: pd.DataFrame) -> pd.DataFrame:

    daily_transactions = df_order_items[df_order_items['order_id'].isin(kelompok_seller)].copy()
    daily_transactions['shipping_limit_date'] = pd.to_datetime(daily_transactions['shipping_limit_date'])

    ## Ekstrak tahun dan bulan
    daily_transactions['year_month_day'] = daily_transactions['shipping_limit_date'].dt.to_period('D')
    daily_transactions = daily_transactions.sort_values(by='year_month_day', ascending=True).reset_index()
    daily_transactions.drop(columns='index', inplace = True)

    monthly_transactions = daily_transactions.copy()

    ## Ekstrak tahun dan bulan
    monthly_transactions['year_month'] = daily_transactions['year_month_day'].dt.to_timestamp().dt.to_period('M')

    ## Kelompokkan berdasarkan year_month dan jumlahkan payment_value_sum
    monthly_transactions = monthly_transactions.groupby('year_month')['seller_id'].count().reset_index()
    monthly_transactions = monthly_transactions.drop(index=monthly_transactions[monthly_transactions['year_month'] == '2018-09'].index)
    monthly_transactions = monthly_transactions.drop(index=monthly_transactions[monthly_transactions['year_month'] == '2020-02'].index)
    monthly_transactions = monthly_transactions.drop(index=monthly_transactions[monthly_transactions['year_month'] == '2020-04'].index)

    return monthly_transactions

### Mendapatkan df_monthly_seller_state
def create_df_monthly_seller_state(state: str, df_order_items: pd.DataFrame, df_sellers: pd.DataFrame, df_order: pd.DataFrame) -> pd.DataFrame:

    df_test = df_order_items[df_order_items['order_id'].isin(kelompok_seller)][['order_id', 'seller_id', 'price']]
    df_test = pd.merge(df_test, df_sellers[['seller_id', 'seller_city', 'seller_state']], on='seller_id', how='left')
    df_test = pd.merge(df_test, df_order[['order_id', 'order_purchase_timestamp']], on='order_id', how='left')
    
    df_test = df_test[df_test.seller_state == state].groupby(by='order_purchase_timestamp').agg({'price':'sum'})
    df_test = df_test.reset_index()
    
    df_test['year_month'] = df_test['order_purchase_timestamp'].dt.to_period('M')
    df_monthly_seller_state = df_test.groupby(by='year_month').agg({'price':'sum'})
    df_monthly_seller_state = df_monthly_seller_state.drop(index=df_monthly_seller_state[df_monthly_seller_state.index == '2018-09'].index)

    return df_monthly_seller_state

### Mendapatkan df_monthly_seller_city
def create_df_monthly_seller_city(city: str, df_order_items: pd.DataFrame, df_sellers: pd.DataFrame, df_order: pd.DataFrame) -> pd.DataFrame:

    df_test = df_order_items[df_order_items['order_id'].isin(kelompok_seller)][['order_id', 'seller_id', 'price']]
    df_test = pd.merge(df_test, df_sellers[['seller_id', 'seller_state', 'seller_city']], on='seller_id', how='left')
    df_test = pd.merge(df_test, df_order[['order_id', 'order_purchase_timestamp']], on='order_id', how='left')
    
    df_test = df_test[df_test.seller_city == city].groupby(by='order_purchase_timestamp').agg({'price':'sum'})
    df_test = df_test.reset_index()
    
    df_test['year_month'] = df_test['order_purchase_timestamp'].dt.to_period('M')
    df_monthly_seller_city = df_test.groupby(by='year_month').agg({'price':'sum'})
    df_monthly_seller_city = df_monthly_seller_city.drop(index=df_monthly_seller_city[df_monthly_seller_city.index == '2018-09'].index)

    return df_monthly_seller_city

### Mendapatkan df_monthly_customer_state
def create_df_monthly_customer_state(state:str, df_order: pd.DataFrame, df_order_payments: pd.DataFrame, df_customer: pd.DataFrame) -> pd.DataFrame:

    df_test = df_order[df_order.order_id.isin(kelompok_customer)][['order_id', 'customer_id', 'order_purchase_timestamp']]
    df_test = pd.merge(df_test, df_order_payments[['order_id', 'payment_value']], on='order_id', how='left')
    df_test = pd.merge(df_test, df_customer[['customer_id', 'customer_city', 'customer_state']], on='customer_id', how='left')

    df_test = df_test[df_test.customer_state == state].groupby(by='order_purchase_timestamp').agg({'payment_value':'sum'})
    df_test = df_test.reset_index()
    df_test['year_month'] = df_test['order_purchase_timestamp'].dt.to_period('M')
    df_monthly_customer_state = df_test.groupby(by='year_month').agg({'payment_value':'sum'})
    df_monthly_customer_state = df_monthly_customer_state.drop(index=df_monthly_customer_state[df_monthly_customer_state.index == '2018-09'].index)
    
    return df_monthly_customer_state

### Mendapatkan df_monthly_customer_city
def create_df_monthly_customer_city(city:str, df_order: pd.DataFrame, df_order_payments: pd.DataFrame, df_customer: pd.DataFrame) -> pd.DataFrame:

    df_test = df_order[df_order.order_id.isin(kelompok_customer)][['order_id', 'customer_id', 'order_purchase_timestamp']]
    df_test = pd.merge(df_test, df_order_payments[['order_id', 'payment_value']], on='order_id', how='left')
    df_test = pd.merge(df_test, df_customer[['customer_id', 'customer_city', 'customer_state']], on='customer_id', how='left')

    df_test = df_test[df_test.customer_city == city].groupby(by='order_purchase_timestamp').agg({'payment_value':'sum'})
    df_test = df_test.reset_index()
    df_test['year_month'] = df_test['order_purchase_timestamp'].dt.to_period('M')
    df_monthly_customer_city = df_test.groupby(by='year_month').agg({'payment_value':'sum'})
    df_monthly_customer_city = df_monthly_customer_city.drop(index=df_monthly_customer_city[df_monthly_customer_city.index == '2018-09'].index)
    
    return df_monthly_customer_city

### Mendapatkan df_brazil
def create_df_brazil() -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk mengahsilkan data frame peta brazil beserta centroidnya.

    Returns:
        brazil_df: GeoPandas Data Frame brazil_df
    """

    # URL mentah file GeoJSON
    url = "https://raw.githubusercontent.com/luizpedone/municipal-brazilian-geodata/refs/heads/master/data/Brasil.json"
    
    # Membaca file GeoJSON
    brazil_df = gpd.read_file(url)

    # Konversi Latitude Longitude ke Meter
    brazil_df['geometry_crs'] = brazil_df['geometry'].to_crs(epsg=3395)
  
    # Menghitung centroid untuk setiap geometris (polygon atau multipolygon)
    brazil_df['centroid_crs'] = brazil_df.geometry_crs.centroid

    # Konversi kembali ke awal
    brazil_df['centroid'] = brazil_df['centroid_crs'].to_crs(epsg=4326)

    return brazil_df

### Mendapatkan df_cities
def create_df_cities(brazil_df: gpd.GeoDataFrame, colors: list, state_index: int) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_cities yang berisi peta dari kota-kota pada state(negara) yang dipilih.

    Parameters:
        brazil_df (GeoPandas DataFrame): GeoPandas Data Frame brazil_df
        colors (list): list warna yang digunakan pada peta brazil
        state_index (int): state yang dipilih oleh user lalu diambil indexnya

    Returns:
        df_cities (gpd.GeoDataFrame):
        GeoPandas Data Frame df_cities       
    """

    state = brazil_df['UF'][state_index]
    color_ = colors[state_index]
    
    # URL mentah file GeoJSON
    url = f"https://raw.githubusercontent.com/luizpedone/municipal-brazilian-geodata/refs/heads/master/data/{state}.json"
    
    # Membaca file GeoJSON
    df_cities = gpd.read_file(url)
    
  # Menghapus aksen di kolom 'city'
    df_cities['NOME'] = df_cities['NOME'].apply(remove_accents)
    df_cities['NOME'] = df_cities['NOME'].apply(lambda x: x.lower())
    
    # Menampilkan lima baris pertama
    df_cities = df_cities.sort_values(by="NOME")

    return df_cities

### Mendapatkan df_geo_point_cust
def create_df_geo_point_cust(df_customer_merged: pd.DataFrame) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_geo_point_cust yang berisi poin-poin (letak) customer di peta brazil.

    Parameters:
        df_customer_merged (pandas DataFrame): Pandas Data Frame df_customer_merged

    Returns:
        gpd.GeoDataFrame (df_geo_point_cust):
        GeoPandas Data Frame df_geo_point_cust       
    """
  
    df_customer_merged = df_customer_merged.rename(columns={'customer_zip_code_prefix':'geolocation_zip_code_prefix'})
    df_customer_merged = pd.merge(df_customer_merged, df_geolocation[['geolocation_zip_code_prefix','geolocation_lat', 'geolocation_lng']], on='geolocation_zip_code_prefix', how='inner')

    # Convert geolocation_lat dan geolocation_lng menjadi point
    df_geo_point_cust = gpd.GeoDataFrame(df_customer_merged, geometry = gpd.points_from_xy(df_customer_merged.geolocation_lng, df_customer_merged.geolocation_lat))

    # Cleaning dikit Rio de Janeiro state
    df_geo_point_cust.drop(df_geo_point_cust[df_geo_point_cust['customer_state'] == 'RJ'][df_geo_point_cust['geolocation_lng'] <= -45].index, inplace = True)
    df_geo_point_cust[df_geo_point_cust['customer_state'] == 'RJ'][df_geo_point_cust['geolocation_lng'] <= -45]

    return df_geo_point_cust

### Mendapatkan df_geo_point_sel
def create_df_geo_point_sel(df_sellers_merged: pd.DataFrame) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_geo_point_sel yang berisi poin-poin (letak) seller di peta brazil.

    Parameters:
        df_sellers_merged (pandas DataFrame): Pandas Data Frame df_sellers_merged

    Returns:
        gpd.GeoDataFrame (df_geo_point_sel):
        GeoPandas Data Frame df_geo_point_sel       
    """
  
    df_sellers_merged = df_sellers_merged.rename(columns={'seller_zip_code_prefix':'geolocation_zip_code_prefix'})
    df_sellers_merged = pd.merge(df_sellers_merged, df_geolocation[['geolocation_zip_code_prefix','geolocation_lat', 'geolocation_lng']], on='geolocation_zip_code_prefix', how='inner')

    # Convert geolocation_lat dan geolocation_lng menjadi point
    df_geo_point_sel = gpd.GeoDataFrame(df_sellers_merged, geometry = gpd.points_from_xy(df_sellers_merged.geolocation_lng, df_sellers_merged.geolocation_lat))

    return df_geo_point_sel

### Mendapatkan df_product_demand
def create_df_product_demand(product_index: int, df_geo_point_cust: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_product_demand yang berisi poin-poin (letak) product yang diinginkan di peta brazil.

    Parameters:
        df_geo_point_cust (GeoPandas DataFrame): GeoPandas Data Frame df_geo_point_cust
        product_index (int): Index dari product yang dipilih

    Returns:
        gpd.GeoDataFrame (df_product_demand):
        GeoPandas Data Frame df_product_demand       
    """

    all_prod = []
    for li in df_geo_point_cust['product_category_name_<lambda>']:
      for i in li:
        all_prod.append(i)
    
    # Membuat Series dari list dan menghitung nilai
    prod_demand_counts = pd.Series(all_prod).value_counts()

    # Product Category yang dipilih
    product_demand_select = prod_demand_counts.index[product_index]

    df_product_demand = df_geo_point_cust[df_geo_point_cust['product_category_name_<lambda>'].apply(lambda x: find_prod(product_demand_select, x)) == True]
    
    return df_product_demand

### Mendapatkan df_product_supply
def create_df_product_supply(product_index: int, df_geo_point_sel: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_product_supply yang berisi poin-poin (letak) product yang diinginkan di peta brazil.

    Parameters:
        df_geo_point_sel (GeoPandas DataFrame): GeoPandas Data Frame df_geo_point_sel
        product_index (int): Index dari product yang dipilih

    Returns:
        gpd.GeoDataFrame (df_product_supply):
        GeoPandas Data Frame df_product_supply       
    """

    all_prod = []
    for li in df_geo_point_sel['product_category_name_<lambda>']:
      for i in li:
        all_prod.append(i)
    
    # Membuat Series dari list dan menghitung nilai
    prod_supply_counts = pd.Series(all_prod).value_counts()

    # Product Category yang dipilih
    product_supply_select = prod_supply_counts.index[product_index]

    df_product_supply = df_geo_point_sel[df_geo_point_sel['product_category_name_<lambda>'].apply(lambda x: find_prod(product_supply_select, x)) == True]
    
    return df_product_supply

### Mendapatkan df_sellers_city_merged
def create_df_sellers_city_merged(df_sellers_merged: pd.DataFrame) -> pd.DataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan 8 kota berpenghasilan terbesar beserta kategori barang yang terjual 
    yang disajikan ke Data Frame create_df_sellers_city_merged

    Parameters:
        df_sellers_merged (pandas DataFrame): Data Frame df_sellers_merged

    Returns:
        df_sellers_city_merged: Data Frame df_sellers_city_merged
    """    

    df_sellers_city_merged = df_sellers_merged.groupby(by='seller_city').agg({
                                                    'price_sum': 'sum',
                                                    'product_category_name_<lambda>': 'sum'
                                                    }).sort_values(by = ('price_sum'), ascending = False).head(8)
    
    return df_sellers_city_merged

### Mendapatkan kategori barang yang banyak dijual di kota berpenghasilan tertinggi
def return_kategori_di_kota_jual(df_sellers_city_merged: pd.DataFrame) -> list:
    
    """
    Fungsi ini bertujuan untuk menghasilkan 10 kategori yang terjual terbanyak di 8 kota berpenghasilan terbesar
    disajikan ke sebuah list yang berisi nama kota dan data frame yang berisi kategori-kategori barang beserta nominalnya:
    
    penjualan_kategoribarang_di_kota = [[kota_1, DataFrame],...,[kota_8, DataFrame]]
    
    Parameters:
        df_sellers_city_merged (pandas Data Frame): Data Frame df_sellers_city_merged

    Returns:
        penjualan_kategoribarang_di_kota (list): 
        list penjualan_kategoribarang_di_kota = [[kota_1, DataFrame],...,[kota_8, DataFrame]]
    """

    penjualan_kategoribarang_di_kota = []
    for i in range(8):
        df_temp = pd.DataFrame(df_sellers_city_merged['product_category_name_<lambda>'].iloc[i],
                            columns=['product_category_name_<lambda>']
                            ).value_counts().head(10)
        penjualan_kategoribarang_di_kota.append([df_sellers_city_merged.index[i],df_temp])

    return penjualan_kategoribarang_di_kota

### Mendapatkan df_customer_city_merged
def create_df_customer_city_merged(df_customer_merged: pd.DataFrame) -> pd.DataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan 8 kota berpengeluaran terbesar beserta kategori barang yang dibeli 
    yang disajikan ke Data Frame create_df_customer_city_merged

    Parameters:
        df_customer_merged (pandas DataFrame): Data Frame df_customer_merged

    Returns:
        df_customer_city_merged (pandas DataFrame): Data Frame df_customer_city_merged
    """    

    df_customer_city_merged = df_customer_merged.groupby(by='customer_city').agg({
                                                        'payment_value_sum': 'sum',
                                                        'product_category_name_<lambda>': 'sum'
                                                        }).sort_values(by = ('payment_value_sum'), ascending = False).head(8)
    
    return df_customer_city_merged

### Mendapatkan kategori barang yang banyak dibeli di kota berpengeluaran tertinggi
def return_kategori_di_kota_beli(df_customer_city_merged: pd.DataFrame) -> list:
    
    """
    Fungsi ini bertujuan untuk menghasilkan 10 kategori yang dibeli terbanyak di 8 kota berpengeluaran terbesar
    disajikan ke sebuah list yang berisi nama kota dan data frame yang berisi kategori-kategori barang beserta nominalnya:
    
    pemberlian_kategoribarang_di_kota = [[kota_1, DataFrame],...,[kota_8, DataFrame]]
    
    Parameters:
        df_customer_city_merged (pandas Data Frame): Data Frame df_customer_city_merged

    Returns:
        pemberlian_kategoribarang_di_kota (list): 
        list pemberlian_kategoribarang_di_kota = [[kota_1, DataFrame],...,[kota_8, DataFrame]]
    """

    pemberlian_kategoribarang_di_kota = []
    for i in range(8):
        df_temp = pd.DataFrame(df_customer_city_merged['product_category_name_<lambda>'].iloc[i],
                            columns=['product_category_name_<lambda>']
                            ).value_counts().head(10)
        pemberlian_kategoribarang_di_kota.append([df_customer_city_merged.index[i],df_temp])

    return pemberlian_kategoribarang_di_kota

### Membuat Klaster customer
kumpulan_klaster = ['Klaster I','Klaster II','Klaster III','Klaster IV','Klaster V','Klaster VI','Klaster VII']

def create_klaster_customer(df_customer_merged: pd.DataFrame) -> pd.DataFrame:

    """
    Fungsi ini digunakan untuk menghasilkan klaster customer

    Parameters:
        df_customer_merged (pandas Data Frames): Data Frame df_customer_merged
    
    Returns:
        df_customer_klaster (pandas DataFrame): Data Frame df_customer_klaster
    """

    batas_atas_klaster_I = 70
    batas_atas_klaster_II = 130
    batas_atas_klaster_III = 210
    batas_atas_klaster_IV = 350
    batas_atas_klaster_V = 1000
    batas_atas_klaster_VI = 4000

    batasan_klaster = [0,
                       batas_atas_klaster_I,
                       batas_atas_klaster_II,
                       batas_atas_klaster_III,
                       batas_atas_klaster_IV,
                       batas_atas_klaster_V,
                       batas_atas_klaster_VI,
                       float('inf')
                       ]

    df_customer_merged['Klaster'] = pd.cut(df_customer_merged['payment_value_sum'],
                                           bins=batasan_klaster,
                                           labels=kumpulan_klaster,
                                           include_lowest=True)
    df_customer_klaster = df_customer_merged[['customer_id','payment_value_sum','Klaster']]

    df_customer_klaster = df_customer_klaster.groupby(by='Klaster').agg({'Klaster': 'count'
                                                                        ,'customer_id': lambda x: list(x)
                                                                        ,'payment_value_sum': lambda x: list(x)})

    df_customer_klaster.columns = ['customer_id_count','customer_id','payment_value_sum']

    return df_customer_klaster

### Membuat Klaster seller
def create_klaster_sellers(df_sellers_merged: pd.DataFrame) -> pd.DataFrame:

    """
    Fungsi ini digunakan untuk menghasilkan klaster seller

    Parameters:
        df_sellers_merged (pandas Data Frames): Data Frame df_sellers_merged
    
    Returns:
        df_sellers_klaster (pandas DataFrame): Data Frame df_sellers_klaster
    """

    batas_atas_klaster_I = 300
    batas_atas_klaster_II = 1000
    batas_atas_klaster_III = 2500
    batas_atas_klaster_IV = 5000
    batas_atas_klaster_V = 10000
    batas_atas_klaster_VI = 50000

    batasan_klaster = [0,
                       batas_atas_klaster_I,
                       batas_atas_klaster_II,
                       batas_atas_klaster_III,
                       batas_atas_klaster_IV,
                       batas_atas_klaster_V,
                       batas_atas_klaster_VI,
                       float('inf')
                       ]

    df_sellers_merged['Klaster'] = pd.cut(df_sellers_merged['price_sum'],
                                          bins=batasan_klaster,
                                          labels=kumpulan_klaster,
                                          include_lowest=True)
    
    df_sellers_klaster = df_sellers_merged[['seller_id','price_sum','Klaster']]

    df_sellers_klaster = df_sellers_klaster.groupby(by='Klaster').agg({'Klaster': 'count'
                                                                    ,'seller_id': lambda x: list(x)
                                                                    ,'price_sum': lambda x: list(x)})

    df_sellers_klaster.columns = ['seller_id_count','seller_id','price_sum']

    return df_sellers_klaster

## MEMBUAT FILTER
min_date = df_order["order_purchase_timestamp"].min()
max_date = df_order["order_purchase_timestamp"].max()


## MEMBUAT FILTER
min_date = df_order["order_purchase_timestamp"].min()
max_date = df_order["order_purchase_timestamp"].max()

with st.sidebar:
    st.title('Proyek Data Analisis')
    # Menambahkan logo perusahaan
    st.image("https://learn.g2.com/hubfs/Imported%20sitepage%20images/1ZB5giUShe0gw9a6L69qAgsd7wKTQ60ZRoJC5Xq3BIXS517sL6i6mnkAN9khqnaIGzE6FASAusRr7w=w1439-h786.png")

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )


### Filter diterapkan
df_order_update = df_order[(df_order["order_purchase_timestamp"] >= str(start_date)) &
                           (df_order["order_purchase_timestamp"] <= str(end_date))]
                           
df_order_items_update = df_order_items[(df_order_items["shipping_limit_date"] >= str(start_date)) &
                                       (df_order_items["shipping_limit_date"] <= str(end_date))]

### Data yang telah difilter diterapkan untuk membuat beberapa data frame
pivot_seller, pivot_order = create_pivot_seller_and_order(df_order_items_update,
                                                          df_product,
                                                          df_order_payments,
                                                          df_order_update)

df_sellers_merged, df_customer_merged = create_df_sellers_and_customer_merged(pivot_seller,
                                                                              df_sellers,
                                                                              pivot_order,
                                                                              df_order_update,
                                                                              df_customer)

# brazil_df =  create_df_brazil()

# df_cities =  create_df_cities(brazil_df, colors, state_index)

# df_geo_point_cust =  create_df_geo_point_cust(df_customer_merged)

# df_geo_point_sel =  create_df_geo_point_sel(df_sellers_merged)

# df_product_demand =  create_df_product_demand(product_index, df_geo_point_cust)

# df_product_supply =  create_df_product_supply(product_index, df_geo_point_sel)

df_sellers_city_merged = create_df_sellers_city_merged(df_sellers_merged)

penjualan_kategoribarang_di_kota = return_kategori_di_kota_jual(df_sellers_city_merged)

df_customer_city_merged = create_df_customer_city_merged(df_customer_merged)

pembelian_kategoribarang_di_kota = return_kategori_di_kota_jual(df_customer_city_merged)

df_customer_klaster = create_klaster_customer(df_customer_merged)

df_sellers_klaster = create_klaster_sellers(df_sellers_merged)


## DEPLOYMENT


## OVERVIEW
st.header('OVERVIEW')
col1, col2 = st.columns(border=True, spec=[0.4, 0.6])

with col1:

    # Revenue
    st.text("Total Revenue E-Commerce")
    st.text(f"{df_customer_merged['payment_value_sum'].sum()} BRL")

    # Transaksi
    st.text("Total Transactions")
    st.text(f"{len(df_order_items[df_order_items['order_id'].isin(kelompok_seller)].index)} Transactions")

    # Active Users
    st.text("Active Users")
    st.text(f"{df_customer_merged['customer_id'].iloc[:-1].nunique()} Users")
        
with col2:

    # Total Revenue Graphic
    monthly_summary = create_monthly_summary(df_customer_merged)

    ## Plotting
    ax = monthly_summary['payment_value_sum'].plot(
        kind='line',
        figsize=(8, 3.5),
        title='Monthly Revenue Trend',
        marker='o'
    )

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Revenue (BRL)')
    plt.xlabel('Month-Year')

    ## Menambahkan anotasi untuk setiap titik
    for idx, value in enumerate(monthly_summary['payment_value_sum']):
        ax.annotate(
            f'{value:.0f}',
            xy=(idx, value),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            fontsize=8
        )

    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # Total Transaksi Graphic
    monthly_transactions = create_monthly_transactions(df_order_items)
    
    ax = monthly_transactions['seller_id'].plot(
        kind='line',
        figsize=(8, 3.5),
        title='Monthly Transactions Trend',
        marker='o'
    )
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Transactions')
    plt.xlabel('Month-Year')

    ## Menambahkan anotasi untuk setiap titik
    for idx, value in enumerate(monthly_transactions['seller_id']):
        ax.annotate(
            f'{value:.0f}',
            xy=(idx, value),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            fontsize=8
        )

    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    # Active Sellers
    st.text("Active Sellers")
    st.text(f"{df_sellers_merged['seller_id'].nunique()} Sellers")

## Sales Analysis
st.header('SALES ANALYSIS')
col1, col2 = st.columns(2)

with col1:

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))

    df_0 = df_sellers_merged.groupby(by='seller_state').agg({
                                                        'price_sum': 'sum'
                                                        }).sort_values(by = ('price_sum'), ascending = False).head(5)
    df_0 = df_0.sort_values(by = ('price_sum'), ascending = False)
    
    colors = ["#8F4700", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(y=df_0.index,
                x=df_0['price_sum'],
                data=df_0,
                palette=colors,
                ax=ax
                )
                
    ax.set_xlabel("Total Penghasilan (Juta BRL)")
    ax.set_ylabel("Nama State")
    ax.set_title("Top 5 Total Penghasilan Seluruh Seller di Setiap State")
    ax.tick_params(axis='y', )
    ax.tick_params(axis='x', )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))

    df_0 = df_sellers_merged.groupby(by='seller_city').agg({
                                                        'price_sum': 'sum'
                                                        }).sort_values(by = ('price_sum'), ascending = False).head(5)
    df_0 = df_0.sort_values(by = ('price_sum'), ascending = False)
    
    index_ = [i[:6]+'...' for i in df_0.index]

    sns.barplot(y=index_,
                x=df_0['price_sum'],
                data=df_0,
                palette=colors,
                ax=ax
                )    
    
    ax.set_xlabel("Total Penghasilan (Juta BRL)")
    ax.set_ylabel("Nama City")
    ax.set_title("Top 5 Total Penghasilan Seluruh Seller di Setiap Kota")
    ax.tick_params(axis='y', )
    ax.tick_params(axis='x', )
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))

    df_0 = df_sellers_merged.groupby(by='seller_id').agg({
                                                        'price_sum': 'sum'
                                                        }).sort_values(by = ('price_sum'), ascending = False).head(5)
    df_0 = df_0.sort_values(by = ('price_sum'), ascending = False)
    
    index_ = [i[:3]+'...' for i in df_0.index]

    sns.barplot(y=index_,
                x=df_0['price_sum'],
                data=df_0,
                palette=colors,
                ax=ax
                )   

    ax.set_xlabel("Total Penghasilan (BRL)")
    ax.set_ylabel("ID Seller")
    ax.set_title("Top 5 Total Penghasilan Seller")
    ax.tick_params(axis='y', )
    ax.tick_params(axis='x', )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    
    df_0 = df_sellers_merged.groupby(by='seller_state').agg({
                                'price_sum': 'sum',
                                'product_category_name_<lambda>': 'sum'
                                }).sort_values(by = ('price_sum'), ascending = False).head(8)
    state = st.selectbox(
        label="Choose State:",
        options=df_0.index,
        index=0
    )

    df_monthly_seller_state = create_df_monthly_seller_state(state, df_order_items, df_sellers, df_order)

    ## Plotting
    ax = df_monthly_seller_state['price'].plot(
            kind='line',
            figsize=(8, 3.5),
            title='Monthly Revenue Trend',
            marker='o'
        )

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Revenue (BRL)')
    plt.xlabel('Month-Year')

    ## Menambahkan anotasi untuk setiap titik
    for idx, value in enumerate(df_monthly_seller_state['price']):
            ax.annotate(
                f'{value:.0f}',
                xy=(idx, value),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )
    
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    df_0 = df_sellers_merged.groupby(by='seller_city').agg({
                                'price_sum': 'sum',
                                'product_category_name_<lambda>': 'sum'
                                }).sort_values(by = ('price_sum'), ascending = False).head(8)
    city = st.selectbox(
        label="Choose City:",
        options=df_0.index,
        index=0
    )

    df_monthly_seller_city = create_df_monthly_seller_city(city, df_order_items, df_sellers, df_order)

    ## Plotting
    ax = df_monthly_seller_city['price'].plot(
            kind='line',
            figsize=(8, 3.5),
            title='Monthly Revenue Trend',
            marker='o'
        )

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Revenue (BRL)')
    plt.xlabel('Month-Year')

    ## Menambahkan anotasi untuk setiap titik
    for idx, value in enumerate(df_monthly_seller_city['price']):
            ax.annotate(
                f'{value:.0f}',
                xy=(idx, value),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )
    
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

## Customer Analysis
st.header('CUSTOMER ANALYSIS')
col1, col2 = st.columns(2)

with col1:

    # st.markdown("<p> </p>", unsafe_allow_html=True)
    # st.markdown("<p> </p>", unsafe_allow_html=True)
    # st.markdown("<p> </p>", unsafe_allow_html=True)
    # st.markdown("<p> </p>", unsafe_allow_html=True)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))

    df_0 = df_customer_merged.groupby(by='customer_state').agg({
                                                        'payment_value_sum': 'sum'
                                                        }).sort_values(by = ('payment_value_sum'), ascending = False).head(5)
    df_0 = df_0.sort_values(by = ('payment_value_sum'), ascending = False)
    
    colors = ["#8F4700", "#D3D3D3","#D3D3D3", "#D3D3D3", "#D3D3D3"]

    sns.barplot(y=df_0.index,
                x=df_0['payment_value_sum'],
                data=df_0,
                palette=colors,
                ax=ax
                )
    ax.set_xlabel("Total Pengeluaran (Juta BRL)")
    ax.set_ylabel("Nama State")
    ax.set_title("Top 5 Total Pengeluaran Seluruh Customer di Setiap State")
    ax.tick_params(axis='y',)
    ax.tick_params(axis='x',)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))

    df_0 = df_customer_merged.groupby(by='customer_city').agg({
                                                        'payment_value_sum': 'sum'
                                                        }).sort_values(by = ('payment_value_sum'), ascending = False).head(5)
    df_0 = df_0.sort_values(by = ('payment_value_sum'), ascending = False)
    
    index_ = [i[:6]+'...' for i in df_0.index]

    sns.barplot(y=index_,
                x=df_0['payment_value_sum'],
                data=df_0,
                palette=colors,
                ax=ax
                )    
    
    ax.set_xlabel("Total Pengeluaran (Juta BRL)")
    ax.set_ylabel("Nama City")
    ax.set_title("Top 5 Total Pengeluaran Seluruh Customer di Setiap Kota")
    ax.tick_params(axis='y',)
    ax.tick_params(axis='x',)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))
    
    df_0 = df_customer_merged.groupby(by='customer_id').agg({
                                                        'payment_value_sum': 'sum'
                                                        }).sort_values(by = ('payment_value_sum'), ascending = False).head(5)
    df_0 = df_0.sort_values(by = ('payment_value_sum'), ascending = False)
    
    index_ = [i[:3]+'...' for i in df_0.index]

    sns.barplot(y=index_,
                x=df_0['payment_value_sum'],
                data=df_0,
                palette=colors,
                ax=ax
                )   

    ax.set_xlabel("Total Pengeluaran (BRL)")
    ax.set_ylabel("ID Customer")
    ax.set_title("Top 5 Total Pengeluaran Customer")
    ax.tick_params(axis='y',)
    ax.tick_params(axis='x',)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:

    df_0 = df_customer_merged.groupby(by='customer_state').agg({
                                'payment_value_sum': 'sum',
                                'product_category_name_<lambda>': 'sum'
                                }).sort_values(by = ('payment_value_sum'), ascending = False).head(8)
    state = st.selectbox(
        label="Choose State:",
        options=df_0.index,
        index=0
    )

    df_monthly_customer_state = create_df_monthly_customer_state(state, df_order, df_order_payments, df_customer)
    
    ## Plotting
    ax = df_monthly_customer_state['payment_value'].plot(
            kind='line',
            figsize=(8, 3.5),
            title='Monthly Revenue Trend',
            marker='o'
        )

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Revenue (BRL)')
    plt.xlabel('Month-Year')

        ## Menambahkan anotasi untuk setiap titik
    for idx, value in enumerate(df_monthly_customer_state['payment_value']):
            ax.annotate(
                f'{value:.0f}',
                xy=(idx, value),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )

    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    df_0 = df_customer_merged.groupby(by='customer_city').agg({
                                'payment_value_sum': 'sum',
                                'product_category_name_<lambda>': 'sum'
                                }).sort_values(by = ('payment_value_sum'), ascending = False).head(8)
    city = st.selectbox(
        label="Choose City:",
        options=df_0.index,
        index=0
    )

    df_monthly_customer_city = create_df_monthly_customer_city(city, df_order, df_order_payments, df_customer)
    
    ## Plotting
    ax = df_monthly_customer_city['payment_value'].plot(
            kind='line',
            figsize=(8, 3.5),
            title='Monthly Revenue Trend',
            marker='o'
        )

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.ylabel('Revenue (BRL)')
    plt.xlabel('Month-Year')

        ## Menambahkan anotasi untuk setiap titik
    for idx, value in enumerate(df_monthly_customer_city['payment_value']):
            ax.annotate(
                f'{value:.0f}',
                xy=(idx, value),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=8
            )

    plt.tight_layout()
    st.pyplot(plt)
    plt.close()
