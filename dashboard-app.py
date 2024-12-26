import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rcParams
import seaborn as sns
import streamlit as st
import math
import geopandas as gpd
from matplotlib.colors import ListedColormap
import unicodedata
import requests
import tempfile
from streamlit_option_menu import option_menu


############# STYLING #############
# Font untuk matplotlib
font_url = 'https://reminerva.github.io/Roboto-Medium.ttf'

# Unduh font ke file sementara
response = requests.get(font_url)
with tempfile.NamedTemporaryFile(delete=False, suffix='.ttf') as temp_font_file:
    temp_font_file.write(response.content)
    temp_font_path = temp_font_file.name  # Path ke file sementara

# Load font custom
custom_font = font_manager.FontProperties(fname=temp_font_path)

# Set font sebagai default
rcParams['font.family'] = custom_font.get_name()

st.html("""
        
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
        
        @font-face {
        font-family: "Source Sans Pro";
        src: url('https://reminerva.github.io/Roboto-Medium.ttf') format("truetype");
        }
        @font-face {
        font-family: "Roboto";
        src: url('https://reminerva.github.io/Roboto-Medium.ttf') format("truetype");
        }

        [data-testid="stMetric"] {
            background-color: #FFF;
            text-align: center;
            align-content: auto;
            margin: 0;
            padding: 10px;
        }

        [data-testid="stMetricLabel"] {
            font-size: .7rem;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.25rem;
        }

        [data-testid="stColumn"] p{
            font-size: .9rem;
        }

        [data-testid="stColumn"] li{
            font-size: .9rem;
        }

        h4 {
            font-family: Roboto;
            text-align: center;
            padding: .5rem 0;
            font-size: 1.5rem;
        }

        h4 span {
            color: #7e74f1
        }

        h5 {
            font-family: Roboto;
            text-align: center;
            padding: .5rem 0;
            font-size: 1.25rem;
        }

        h5 span {
            color: #7e74f1
        }

        </style>
        """)

@st.cache_resource
def read_csv():

    # Import File CSV
    df_customer = pd.read_csv('data/df_customer_clean.csv')
    df_order = pd.read_csv('data/df_order_clean.csv')
    df_order_items = pd.read_csv('data/df_order_items_clean.csv')
    df_order_payments = pd.read_csv('data/df_order_payments_clean.csv')
    df_product = pd.read_csv('data/df_product_clean.csv')
    df_sellers = pd.read_csv('data/df_sellers_clean.csv')
    df_geolocation = pd.read_csv('data/df_geolocation_clean.csv')

    # Tertinggal
    df_geolocation.drop(df_geolocation[df_geolocation['geolocation_lat'] <= -35].index, inplace=True)

    ## Convert datetime
    df_order['order_purchase_timestamp'] = pd.to_datetime(df_order['order_purchase_timestamp'])
    df_order_items['shipping_limit_date'] = pd.to_datetime(df_order_items['shipping_limit_date'])

    return df_customer, df_order, df_order_items, df_order_payments, df_product, df_sellers, df_geolocation

df_customer, df_order, df_order_items, df_order_payments, df_product, df_sellers, df_geolocation = read_csv()

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
def find_prod(prod_select:str, x:list):

    if prod_select in x:
      return True
    else:
      return False
    
### Mendapatkan klaster rfm
def assign_klaster_rfm(data_frame):
    
    # Hitung jumlah skor yang lebih dari 3
    count_high_scores = sum([data_frame['score_freq'] >= 2, data_frame['score_rec'] >= 2, data_frame['score_monet'] > 2])

    # Tentukan klaster berdasarkan jumlah skor tinggi
    if count_high_scores >= 2:
        return "1st Priority"
    elif count_high_scores == 1:
        return "2nd Priority"
    else:
        return "3rd Priority"

### Convert number ke teks 
def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'

### Mendapatkan pivot_seller dan pivot_order
@st.cache_data
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
@st.cache_data
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

    # ## Ubah kembali year_month ke string (opsional)
    # monthly_summary['year_month'] = monthly_summary['year_month'].astype(str)
    # monthly_summary['year_month'] = pd.to_datetime(monthly_summary['year_month'])
    # monthly_summary['year_month'] = (monthly_summary['year_month']).dt.to_period('M')

    monthly_summary = monthly_summary.drop(index=monthly_summary[monthly_summary['year_month'] == '2018-09'].index)

    monthly_summary = monthly_summary.set_index('year_month')

    return monthly_summary

### Mendapatkan monthly_transactions
def create_monthly_transactions(df_order_items: pd.DataFrame) -> tuple:

    daily_transactions = df_order_items[df_order_items['order_id'].isin(kelompok_seller)].copy()
    daily_transactions['shipping_limit_date'] = pd.to_datetime(daily_transactions['shipping_limit_date'])

    ## Ekstrak tahun dan bulan
    daily_transactions['year_month_day'] = daily_transactions['shipping_limit_date'].dt.to_period('D')
    daily_transactions = daily_transactions.sort_values(by='year_month_day', ascending=True).reset_index()
    daily_transactions.drop(columns='index', inplace = True)

    daily_transactions = daily_transactions.drop(index=daily_transactions[daily_transactions['year_month_day'] == '2020-02-03'].index)
    daily_transactions = daily_transactions.drop(index=daily_transactions[daily_transactions['year_month_day'] == '2020-04-09'].index)
    daily_transactions = daily_transactions.drop(index=daily_transactions[daily_transactions['year_month_day'] == '2020-04-09'].index)

    monthly_transactions = daily_transactions.copy()

    ## Ekstrak tahun dan bulan
    monthly_transactions['year_month'] = daily_transactions['year_month_day'].dt.to_timestamp().dt.to_period('M')

    ## Kelompokkan berdasarkan year_month dan jumlahkan payment_value_sum
    monthly_transactions = monthly_transactions.groupby('year_month')['seller_id'].count().reset_index()
    monthly_transactions = monthly_transactions.drop(index=monthly_transactions[monthly_transactions['year_month'] == '2018-09'].index)
    monthly_transactions = monthly_transactions.drop(index=monthly_transactions[monthly_transactions['year_month'] == '2020-02'].index)
    monthly_transactions = monthly_transactions.drop(index=monthly_transactions[monthly_transactions['year_month'] == '2020-04'].index)

    monthly_transactions = monthly_transactions.set_index('year_month')

    return monthly_transactions, daily_transactions

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

### Mendapatkan rec_pivot_org
@st.cache_data
def create_rec_pivot_org(period: int, daily_transactions: pd.DataFrame) -> pd.DataFrame:

    rec = daily_transactions[daily_transactions['year_month_day'] >= daily_transactions['year_month_day'].max() - pd.Timedelta(days=period)][daily_transactions['year_month_day'] <= daily_transactions['year_month_day'].max()]['year_month_day']

    a = rec
    b = rec.max()

    rec_pivot_org = pd.merge(daily_transactions['order_id'], (b-a), left_index=True, right_index=True)
    
    return rec_pivot_org

### Mendapatkan freq_pivot_org
@st.cache_data
def create_freq_pivot_org(period: int, daily_transactions: pd.DataFrame) -> pd.DataFrame:

    freq = daily_transactions[daily_transactions['year_month_day'] >= daily_transactions['year_month_day'].max() - pd.Timedelta(days=period)][daily_transactions['year_month_day'] <= daily_transactions['year_month_day'].max()]
    freq_pivot_org = freq.groupby(by='order_id').agg({'order_item_id': 'count'}).sort_values(by='order_item_id', ascending=False)

    return freq_pivot_org

### Mendapatkan monet_pivot_org
@st.cache_data
def create_monet_pivot_org(period: int, df_customer_merged: pd.DataFrame) -> pd.DataFrame:

    monet = df_customer_merged[df_customer_merged['order_purchase_timestamp'] >= df_customer_merged['order_purchase_timestamp'].max() - pd.Timedelta(days=period)][df_customer_merged['order_purchase_timestamp'] <= df_customer_merged['order_purchase_timestamp'].max()][['order_id', 'payment_value_sum','order_purchase_timestamp']].sort_values(by='order_purchase_timestamp', ascending=True)
    
    monet_pivot_org = monet[monet['order_purchase_timestamp'] >= monet['order_purchase_timestamp'].max() - pd.Timedelta(days=period)].groupby('order_id').agg({'payment_value_sum': 'sum'})
    
    return monet_pivot_org

### Mendapatkan df_rfm_clustering
@st.cache_data
def create_df_rfm_clustering(period:int, rec_pivot_org: pd.DataFrame, freq_pivot_org: pd.DataFrame, monet_pivot_org:pd.DataFrame) -> pd.DataFrame:

    period = period/30

    df_rfm = pd.merge(rec_pivot_org, freq_pivot_org, on='order_id')
    df_rfm = pd.merge(df_rfm, monet_pivot_org, on='order_id')

    df_rfm['year_month_day'] = df_rfm['year_month_day'].astype('str')

    df_rfm['year_month_day'] = df_rfm['year_month_day'].str.replace(r"[<*> Days]", "", regex=True)

    df_rfm['year_month_day'] = pd.to_numeric(df_rfm['year_month_day'], errors='coerce').fillna(0).astype('int64')

    df_rfm['score_freq'] = df_rfm['order_item_id'].apply(lambda x: 3 if x >= 3
                                                        else (2 if x >= 2
                                                                else (1 if x >= 0 else 0)))

    df_rfm['score_rec'] = df_rfm['year_month_day'].apply(lambda x: 1 if x >= 26*period
                                                        else (2 if x >= 20*period
                                                                else (3 if x >= 0 else 0)))

    m = df_rfm['payment_value_sum'].max()/3
    df_rfm['score_monet'] = df_rfm['payment_value_sum'].apply(lambda x: 5 if x >= 1.75*(m)*period
                                            else (4 if x >= 1.25*(m)*period
                                                    else (3 if x >= 1*(m)*period
                                                        else (2 if (1/2)*(m)*period
                                                                else (1 if x >= 0 else 0)))))
    
    df_rfm.sort_values(by='order_item_id', ascending=False)
    
    df_rfm['klaster_rfm_score'] = df_rfm.apply(assign_klaster_rfm, axis=1)
    df_rfm = df_rfm.sort_values(by='klaster_rfm_score', ascending=True)

    df_rfm_clustering = df_rfm[['order_id', 'klaster_rfm_score']].drop_duplicates()

    return df_rfm_clustering

### Mendapatkan df_brazil
@st.cache_resource
def create_df_brazil() -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk mengahsilkan data frame peta brazil beserta centroidnya.

    Returns:
        brazil_df: GeoPandas Data Frame brazil_df
    """

    # URL mentah file GeoJSON
    url = "https://raw.githubusercontent.com/luizpedone/municipal-brazilian-geodata/refs/heads/master/data/Brasil.json"
    
    # Membaca file GeoJSON
    _brazil_df = gpd.read_file(url)

    # Konversi Latitude Longitude ke Meter
    _brazil_df['geometry_crs'] = _brazil_df['geometry'].to_crs(epsg=3395)
  
    # Menghitung centroid untuk setiap geometris (polygon atau multipolygon)
    _brazil_df['centroid_crs'] = _brazil_df.geometry_crs.centroid

    # Konversi kembali ke awal
    _brazil_df['centroid'] = _brazil_df['centroid_crs'].to_crs(epsg=4326)

    _brazil_df = _brazil_df.sort_values(by='UF', ascending=True).reset_index(drop=True)

    return _brazil_df

### Mendapatkan df_cities
def create_df_cities(state: str) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_cities yang berisi peta dari kota-kota pada state(negara) yang dipilih.

    Parameters:
        colors (list): list warna yang digunakan pada peta brazil
        state (str): state yang dipilih oleh user lalu

    Returns:
        df_cities (gpd.GeoDataFrame):
        GeoPandas Data Frame df_cities       
    """
    
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

### Mendapatkan prod_demand_counts
def create_prod_demand_counts(df_geo_point_cust: pd.DataFrame) -> pd.Series:

    all_prod = []
    for li in df_geo_point_cust['product_category_name_<lambda>']:
      for i in li:
        all_prod.append(i)
    
    # Membuat Series dari list dan menghitung nilai
    prod_demand_counts = pd.Series(all_prod).value_counts()

    prod_demand_counts = prod_demand_counts.apply(lambda x: str(x))
    prod_demand_counts.index  = prod_demand_counts.index.str.replace('_', ' ').str.title()
    prod_demand_counts['index+count'] = prod_demand_counts.index + ' (' + prod_demand_counts + ' Items)'

    return prod_demand_counts

### Mendapatkan df_product_demand
def create_df_product_demand(prod_cat_demand_select: str, df_geo_point_cust: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_product_demand yang berisi poin-poin (letak) product yang diinginkan di peta brazil.

    Parameters:
        df_geo_point_cust (GeoPandas DataFrame): GeoPandas Data Frame df_geo_point_cust
        prod_cat_demand_select (str): Product yang dipilih

    Returns:
        gpd.GeoDataFrame (df_product_demand):
        GeoPandas Data Frame df_product_demand       
    """

    prod_cat_demand_select = prod_cat_demand_select.split('(')[0].strip()
    prod_cat_demand_select = prod_cat_demand_select.lower().replace(" ", "_")

    df_product_demand = df_geo_point_cust[df_geo_point_cust['product_category_name_<lambda>'].apply(lambda x: find_prod(prod_cat_demand_select, x)) == True]
    
    return df_product_demand

### Mendapatkan prod_demand_counts
def create_prod_supply_counts(df_geo_point_sel: pd.DataFrame) -> pd.Series:

    all_prod = []
    for li in df_geo_point_sel['product_category_name_<lambda>']:
      for i in li:
        all_prod.append(i)
    
    # Membuat Series dari list dan menghitung nilai
    prod_supply_counts = pd.Series(all_prod).value_counts()

    ### Hitung banyaknya seller yang menjual setiap produk
    list_0 = []
    for i in prod_supply_counts.index:
        list_0.append(df_geo_point_sel['seller_id'][
            df_geo_point_sel['product_category_name_<lambda>'].apply(
                lambda x: find_prod(i, x)) == True].count())
    
    data = {'index': prod_supply_counts.index, 'count': list_0}

    prod_supply_counts = pd.DataFrame(data)

    prod_supply_counts['count'] = prod_supply_counts['count'].apply(lambda x: str(x))
    prod_supply_counts['index']  = prod_supply_counts['index'].apply(lambda x: (x).replace('_', ' ').title())

    prod_supply_counts['index+count'] = prod_supply_counts['index'] + ' (' + prod_supply_counts['count'] + ' Sellers)'

    return prod_supply_counts

### Mendapatkan df_product_supply
def create_df_product_supply(prod_cat_supply_select: str, df_geo_point_sel: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    """
    Fungsi ini bertujuan untuk menghasilkan GeoPandas Data Frame df_product_supply yang berisi poin-poin (letak) product yang diinginkan di peta brazil.

    Parameters:
        df_geo_point_sel (GeoPandas DataFrame): GeoPandas Data Frame df_geo_point_sel
        prod_cat_supply_select (str): Index dari product yang dipilih

    Returns:
        gpd.GeoDataFrame (df_product_supply):
        GeoPandas Data Frame df_product_supply       
    """

    prod_cat_supply_select = prod_cat_supply_select.split('(')[0].strip()
    prod_cat_supply_select = prod_cat_supply_select.lower().replace(" ", "_")

    df_product_supply = df_geo_point_sel[df_geo_point_sel['product_category_name_<lambda>'].apply(lambda x: find_prod(prod_cat_supply_select, x)) == True]
    
    return df_product_supply

### Mendapatkan df_sellers_state_merge
@st.cache_data
def create_df_sellers_state_merged(df_sellers_merged: pd.DataFrame) -> pd.DataFrame:

    df_sellers_state_merged = df_sellers_merged.groupby(by='seller_state').agg({
                                'price_sum': 'sum',
                                'product_category_name_<lambda>': 'sum',
                                'seller_id': 'count'
                                }).sort_values(by = ('price_sum'), ascending = False).head(8)

    df_sellers_state_merged.seller_id = df_sellers_state_merged.seller_id.apply(lambda x: str(x))
    df_sellers_state_merged['index+id'] = df_sellers_state_merged.index + ' (' + df_sellers_state_merged.seller_id + ' Sellers)'

    return df_sellers_state_merged

### Mendapatkan df_sellers_city_merged
@st.cache_data
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
@st.cache_data
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

### Mendapatkan df_customer_state_merge
@st.cache_data
def create_df_customer_state_merged(df_customer_merged: pd.DataFrame) -> pd.DataFrame:

    df_customer_state_merged = df_customer_merged.groupby(by='customer_state').agg({
                                'payment_value_sum': 'sum',
                                'product_category_name_<lambda>': 'sum',
                                'customer_id': 'count'
                                }).sort_values(by = ('payment_value_sum'), ascending = False).head(8)

    df_customer_state_merged.customer_id = df_customer_state_merged.customer_id.apply(lambda x: str(x))
    df_customer_state_merged['index+id'] = df_customer_state_merged.index + ' (' + df_customer_state_merged.customer_id + ' Customers)'

    return df_customer_state_merged

### Mendapatkan df_customer_city_merged
@st.cache_data
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
@st.cache_data
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

# ### Membuat Klaster customer
# kumpulan_klaster = ['Klaster I','Klaster II','Klaster III','Klaster IV','Klaster V','Klaster VI','Klaster VII']

# def create_klaster_customer(df_customer_merged: pd.DataFrame) -> pd.DataFrame:

#     """
#     Fungsi ini digunakan untuk menghasilkan klaster customer

#     Parameters:
#         df_customer_merged (pandas Data Frames): Data Frame df_customer_merged
    
#     Returns:
#         df_customer_klaster (pandas DataFrame): Data Frame df_customer_klaster
#     """

#     batas_atas_klaster_I = 70
#     batas_atas_klaster_II = 130
#     batas_atas_klaster_III = 210
#     batas_atas_klaster_IV = 350
#     batas_atas_klaster_V = 1000
#     batas_atas_klaster_VI = 4000

#     batasan_klaster = [0,
#                        batas_atas_klaster_I,
#                        batas_atas_klaster_II,
#                        batas_atas_klaster_III,
#                        batas_atas_klaster_IV,
#                        batas_atas_klaster_V,
#                        batas_atas_klaster_VI,
#                        float('inf')
#                        ]

#     df_customer_merged['Klaster'] = pd.cut(df_customer_merged['payment_value_sum'],
#                                            bins=batasan_klaster,
#                                            labels=kumpulan_klaster,
#                                            include_lowest=True)
#     df_customer_klaster = df_customer_merged[['customer_id','payment_value_sum','Klaster']]

#     df_customer_klaster = df_customer_klaster.groupby(by='Klaster').agg({'Klaster': 'count'
#                                                                         ,'customer_id': lambda x: list(x)
#                                                                         ,'payment_value_sum': lambda x: list(x)})

#     df_customer_klaster.columns = ['customer_id_count','customer_id','payment_value_sum']

#     return df_customer_klaster

# ### Membuat Klaster seller
# def create_klaster_sellers(df_sellers_merged: pd.DataFrame) -> pd.DataFrame:

#     """
#     Fungsi ini digunakan untuk menghasilkan klaster seller

#     Parameters:
#         df_sellers_merged (pandas Data Frames): Data Frame df_sellers_merged
    
#     Returns:
#         df_sellers_klaster (pandas DataFrame): Data Frame df_sellers_klaster
#     """

#     batas_atas_klaster_I = 300
#     batas_atas_klaster_II = 1000
#     batas_atas_klaster_III = 2500
#     batas_atas_klaster_IV = 5000
#     batas_atas_klaster_V = 10000
#     batas_atas_klaster_VI = 50000

#     batasan_klaster = [0,
#                        batas_atas_klaster_I,
#                        batas_atas_klaster_II,
#                        batas_atas_klaster_III,
#                        batas_atas_klaster_IV,
#                        batas_atas_klaster_V,
#                        batas_atas_klaster_VI,
#                        float('inf')
#                        ]

#     df_sellers_merged['Klaster'] = pd.cut(df_sellers_merged['price_sum'],
#                                           bins=batasan_klaster,
#                                           labels=kumpulan_klaster,
#                                           include_lowest=True)
    
#     df_sellers_klaster = df_sellers_merged[['seller_id','price_sum','Klaster']]

#     df_sellers_klaster = df_sellers_klaster.groupby(by='Klaster').agg({'Klaster': 'count'
#                                                                     ,'seller_id': lambda x: list(x)
#                                                                     ,'price_sum': lambda x: list(x)})

#     df_sellers_klaster.columns = ['seller_id_count','seller_id','price_sum']

#     return df_sellers_klaster


#### GRAFIK
##### GRAFIK LINE CHART
@st.cache_resource
def create_line_chart(data_frame: pd.DataFrame, column_: str, title_: str, ylabel_: str):

    ## Plotting
    ax = data_frame[column_].plot(
        kind='line',
        figsize=(8, 3.5),
        marker='o',
        color= '#7e74f1',
    )

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.title(title_, fontproperties=custom_font)
    plt.ylabel(ylabel_, fontproperties=custom_font)
    plt.xlabel('Month-Year', fontproperties=custom_font)
    plt.tick_params(axis='y')
    plt.tick_params(axis='x')

    ## Menambahkan anotasi untuk setiap titik
    for idx, value in enumerate(data_frame[column_]):
        ax.annotate(
            f'{value:.0f}',
            xy=(idx, value),
            xytext=(0, 5),
            textcoords='offset points',
            ha='center',
            fontsize=8,
        )

    # Mengatur font pada tick labels
    for label in ax.get_xmajorticklabels():
        (label.set_fontproperties(custom_font))

    for label in ax.get_xminorticklabels():
        (label.set_fontproperties(custom_font))

    for label in ax.get_yticklabels():
        label.set_fontproperties(custom_font)

    plt.tight_layout()
    return st.pyplot(plt, clear_figure=True)

##### GRAFIK BAR CHART
@st.cache_resource
def create_bar_chart(data_frame: pd.DataFrame, 
                     _index_:list, 
                     xlabel_: str, 
                     ylabel_: str, 
                     title_: str, 
                     column_: str, 
                     colors: list,
                     slicer=3):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))
    

    for i in _index_:

        if len(i) > slicer:

            _index_ = [j[:slicer]+'...' for j in _index_]
            break

    sns.barplot(y=_index_,
                x=data_frame[column_],
                data=data_frame,
                palette=colors,
                ax=ax
                )
                
    ax.set_xlabel(xlabel_, fontproperties=custom_font)
    ax.set_ylabel(ylabel_, fontproperties=custom_font)
    ax.set_title(title_, fontproperties=custom_font)
    ax.tick_params(axis='y', )
    ax.tick_params(axis='x', )

        # Mengatur font pada tick labels
    for label in ax.get_xmajorticklabels():
        (label.set_fontproperties(custom_font))

    for label in ax.get_xminorticklabels():
        (label.set_fontproperties(custom_font))

    for label in ax.get_yticklabels():
        label.set_fontproperties(custom_font)

    plt.gca().spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    return st.pyplot(plt)

##### GRAFIK PIE CHART
@st.cache_resource
def create_pie_chart(df_rfm_clustering: pd.DataFrame, title_):

    klaster = df_rfm_clustering['klaster_rfm_score'].unique()
    count = (df_rfm_clustering[df_rfm_clustering['klaster_rfm_score'] == klaster[0]]['order_id'].count(),
             df_rfm_clustering[df_rfm_clustering['klaster_rfm_score'] == klaster[1]]['order_id'].count(),
             df_rfm_clustering[df_rfm_clustering['klaster_rfm_score'] == klaster[2]]['order_id'].count(),
             )
      
    colors = ("#03DAC6", "#7e74f1", "#5d51e8")
    explode = (0.07, 0.03, 0.05)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))

    ax.pie(
        x=count,
        labels=klaster,
        autopct='%1.1f%%',
        colors=colors,
        explode=explode,
        wedgeprops = {'width': 0.5},
        textprops={'fontproperties': custom_font}
        )

    plt.title(title_, fontproperties=custom_font)
    plt.tight_layout()

    return st.pyplot(plt)

##### GRAFIK PETA BRAZIL
@st.cache_resource
def create_map_brazil(column_, _brazil_df, _df_geo_point, colors_map):

    axis = _brazil_df.plot(color = 'white', edgecolor='black', figsize=(10, 15))
    _df_geo_point.sort_values(by=column_, ascending=True).plot(ax = axis, column=column_,  cmap=ListedColormap(colors_map), markersize = 5, legend=True)

    # Menambah label provinsi
    for city, coords in zip(_brazil_df.UF, _brazil_df.centroid):
        plt.text(coords.x, coords.y, city, fontsize=15, ha='center', color='black', fontproperties=custom_font)

    # Menambahkan judul
    plt.title('Brazil Map', fontsize=25,fontproperties=custom_font)

    plt.tight_layout()

    return st.pyplot(plt, clear_figure=True)

##### GRAFIK PETA STATE CUSTOMER
@st.cache_resource
def create_map_state_customer(state_map_select, _brazil_df, _df_geo_point, colors_map):

    df_cities =  create_df_cities(state_map_select)
    color_ = colors_map[int(_brazil_df.sort_values(by='UF', ascending=True)[_brazil_df['UF'] == state_map_select]['UF'].index[0])]

    # Filter, hanya poin-poin yang ada di state yang dipilih saja
    # _df_geo_point = _df_geo_point[_df_geo_point.geometry.within(_brazil_df[_brazil_df['UF'] == state_map_select].geometry.iloc[0])]

    # Plot map
    axis = df_cities.plot(color = 'white', edgecolor = 'black', figsize = (10, 10))
    _df_geo_point[_df_geo_point.geometry.within(_brazil_df[_brazil_df['UF'] == state_map_select].geometry.iloc[0])].plot(ax = axis, color = color_, markersize = 5)
    
    plt.title(f'{state_map_select} State Map', fontproperties=custom_font, fontsize=15)

    plt.tight_layout()

    return st.pyplot(plt, clear_figure=True)

##### GRAFIK PETA STATE SELLER
@st.cache_resource
def create_map_state_seller(state_map_select, _brazil_df, _df_geo_point, colors_map):

    df_cities =  create_df_cities(state_map_select)
    color_ = colors_map[int(_brazil_df.sort_values(by='UF', ascending=True)[_brazil_df['UF'] == state_map_select]['UF'].index[0])]

    # Filter, hanya poin-poin yang ada di state yang dipilih saja
    # _df_geo_point = _df_geo_point[_df_geo_point.geometry.within(_brazil_df[_brazil_df['UF'] == state_map_select].geometry.iloc[0])]

    # Plot map
    axis = df_cities.plot(color = 'white', edgecolor = 'black', figsize = (10, 10))
    _df_geo_point[_df_geo_point.geometry.within(_brazil_df[_brazil_df['UF'] == state_map_select].geometry.iloc[0])].plot(ax = axis, color = color_, markersize = 5)
    
    plt.title(f'{state_map_select} State Map', fontproperties=custom_font, fontsize=15)

    plt.tight_layout()

    return st.pyplot(plt, clear_figure=True)

##### GRAFIK PETA BRAZIL PRODUCT DEMAND
@st.cache_resource
def create_map_brazil_product_dem(column_, _brazil_df, prod_cat_demand_select, _df_geo_point_cust, colors_map):

    _df_geo_point = create_df_product_demand(prod_cat_demand_select, _df_geo_point_cust)

    axis = _brazil_df.plot(color = 'white', edgecolor='black', figsize=(10, 15))
    _df_geo_point.sort_values(by=column_, ascending=True).plot(ax = axis, column=column_,  cmap=ListedColormap(colors_map), markersize = 5, legend=True)

    # Menambah label provinsi
    for city, coords in zip(_brazil_df.UF, _brazil_df.centroid):
        plt.text(coords.x, coords.y, city, fontsize=15, ha='center', color='black', fontproperties=custom_font)

    # Menambahkan judul
    plt.title('Peta Provinsi Brazil', fontsize=25,fontproperties=custom_font)

    plt.tight_layout()

    return st.pyplot(plt, clear_figure=True)

##### GRAFIK PETA BRAZIL PRODUCT SUPPLY
@st.cache_resource
def create_map_brazil_product_sup(column_, _brazil_df, prod_cat_supply_select, _df_geo_point_cust, colors_map):

    _df_geo_point = create_df_product_supply(prod_cat_supply_select, _df_geo_point_cust)

    axis = _brazil_df.plot(color = 'white', edgecolor='black', figsize=(10, 15))
    _df_geo_point.sort_values(by=column_, ascending=True).plot(ax = axis, column=column_,  cmap=ListedColormap(colors_map), markersize = 5, legend=True)

    # Menambah label provinsi
    for city, coords in zip(_brazil_df.UF, _brazil_df.centroid):
        plt.text(coords.x, coords.y, city, fontsize=15, ha='center', color='black', fontproperties=custom_font)

    # Menambahkan judul
    plt.title('Peta Provinsi Brazil', fontsize=25,fontproperties=custom_font)

    plt.tight_layout()

    return st.pyplot(plt, clear_figure=True)

## MEMBUAT FILTER
min_date = df_order["order_purchase_timestamp"].min()
max_date = df_order["order_purchase_timestamp"].max()


## MEMBUAT FILTER
min_date = df_order["order_purchase_timestamp"].min()
max_date = df_order["order_purchase_timestamp"].max()

with st.sidebar:
    st.html('<h4><span>DASHBOARD</span></h4>')
    # Menambahkan logo perusahaan
    # st.image("https://learn.g2.com/hubfs/Imported%20sitepage%20images/1ZB5giUShe0gw9a6L69qAgsd7wKTQ60ZRoJC5Xq3BIXS517sL6i6mnkAN9khqnaIGzE6FASAusRr7w=w1439-h786.png")

    # Menu navigasi
    selected = option_menu(
        menu_title="Navigation Menu",  # required
        options=["OVERVIEW", "SALES ANALYSIS", "CUSTOMER ANALYSIS",
                 "RFM ANALYSIS", "PRODUCT ANALYSIS", "GEOSPATIAL ANALYSIS",
                 "SHOW ALL"],  # required
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Interval',
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

brazil_df =  create_df_brazil()

df_geo_point_cust =  create_df_geo_point_cust(df_customer_merged)

prod_demand_counts = create_prod_demand_counts(df_geo_point_cust)

df_geo_point_sel =  create_df_geo_point_sel(df_sellers_merged)

df_sellers_state_merged = create_df_sellers_state_merged(df_sellers_merged)

df_sellers_city_merged = create_df_sellers_city_merged(df_sellers_merged)

penjualan_kategoribarang_di_kota = return_kategori_di_kota_jual(df_sellers_city_merged)

df_customer_state_merged = create_df_customer_state_merged(df_customer_merged)

df_customer_city_merged = create_df_customer_city_merged(df_customer_merged)

pembelian_kategoribarang_di_kota = return_kategori_di_kota_jual(df_customer_city_merged)

# df_customer_klaster = create_klaster_customer(df_customer_merged)

# df_sellers_klaster = create_klaster_sellers(df_sellers_merged)


## DEPLOYMENT
# plt.style.use('default')

if selected == "OVERVIEW" or selected == "SHOW ALL":
    ## OVERVIEW
    st.html('<h4><span>OVERVIEW</span></h4>')
    col1, col2 = st.columns(spec=[0.4, 0.6])

    with col1:

        # Revenue
        value_ = format_number(df_customer_merged['payment_value_sum'].sum())
        st.metric(label=("Total Revenue"),
                value=(f"{value_} BRL"),
                border=True)
        
        # Transaksi
        value_ = format_number(len(df_order_items_update[df_order_items_update['order_id'].isin(kelompok_seller)].index))
        st.metric(label="Total Transactions",
                value=(f"{value_} Transactions"),
                border=True)
        
        # Active Users
        value_ = format_number(df_customer_merged['customer_id'].iloc[:-1].nunique())
        st.metric(label="Active Users",
                value=(f"{value_} Users"),
                border=True)
        
        # Active Sellers
        value_ = format_number(df_sellers_merged['seller_id'].nunique())
        st.metric(label="Active Sellers",
                value=(f"{value_} Sellers"),
                border=True)
            
    with col2:

        # Total Revenue Graphic
        monthly_summary = create_monthly_summary(df_customer_merged)

        create_line_chart(monthly_summary, 'payment_value_sum', 'Monthly Revenue Trend', 'Revenue (BRL)')
        
        plt.close('all')

        # Total Transaksi Graphic
        monthly_transactions, daily_transactions = create_monthly_transactions(df_order_items_update)
        
        create_line_chart(monthly_transactions, 'seller_id', 'Monthly Transactions Trend', 'Transactions')
        
        plt.close('all')

if selected == "SALES ANALYSIS" or selected == "SHOW ALL":
    ## Sales Analysis
    st.html('<h4><span>SALES </span>ANALYSIS</h4>')
    col1, col2 = st.columns(2)

    with col1:

        colors = ["#7e74f1", "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE",
                "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE"]

        create_bar_chart(df_sellers_state_merged.head(5),
                        df_sellers_state_merged.head(5).index,
                        "Total Incomes (Million BRL)",
                        "State's Name",
                        "Top 5 Total Incomes by All Sellers in Each State",
                        'price_sum',
                        colors
                        )

        plt.close('all')

        create_bar_chart(df_sellers_city_merged.head(5),
                        df_sellers_state_merged.head(5).index,
                        "Total Incomes (Million BRL)",
                        "City's Name",
                        "Top 5 Total Incomes by All Sellers in Each City",
                        'price_sum',
                        colors
                        )

        plt.close('all')

        df_0 = df_sellers_merged.groupby(by='seller_id').agg({
                                                            'price_sum': 'sum'
                                                            }).sort_values(by = ('price_sum'), ascending = False).head(5)
        
        create_bar_chart(df_0,
                        df_0.index,
                        "Total Incomes (BRL)",
                        "Seller's ID",
                        "Top 5 Seller's Total Incomes",
                        'price_sum',
                        colors
                        )
        
        plt.close('all')

    with col2:
        
        state = st.selectbox(
            label="Choose State:",
            options=df_sellers_state_merged.index,
            index=0
        )

        df_monthly_seller_state = create_df_monthly_seller_state(state, df_order_items_update, df_sellers, df_order)

        create_line_chart(df_monthly_seller_state, 
                        'price', 
                        f'{state} State Monthly Incomes Trend', 
                        'Incomes (BRL)'
                        )

        plt.close('all')

        city = st.selectbox(
            label="Choose City:",
            options=df_sellers_city_merged.index,
            index=0
        )

        df_monthly_seller_city = create_df_monthly_seller_city(city, df_order_items_update, df_sellers, df_order)

        create_line_chart(df_monthly_seller_city, 
                        'price', 
                        f'{city.title()} Monthly Incomes Trend', 
                        'Incomes (BRL)'
                        )
                        
        plt.close('all')

if selected == "CUSTOMER ANALYSIS" or selected == "SHOW ALL":
    ## Customer Analysis
    st.html('<h4><span>CUSTOMER </span>ANALYSIS</h4>')
    col1, col2 = st.columns(2)

    with col1:

        # st.markdown("<p> </p>", unsafe_allow_html=True)
        # st.markdown("<p> </p>", unsafe_allow_html=True)
        # st.markdown("<p> </p>", unsafe_allow_html=True)
        # st.markdown("<p> </p>", unsafe_allow_html=True)

        colors = ["#7e74f1", "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE",
                "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE"]

        create_bar_chart(df_customer_state_merged.head(5),
                        df_customer_state_merged.head(5).index,
                        "Total Expenses (Million BRL)",
                        "State's Name",
                        "Top 5 Total Expenses by All Customers in Each State",
                        "payment_value_sum",
                        colors
                        )

        plt.close('all')
        
        create_bar_chart(df_customer_city_merged.head(5),
                        df_customer_city_merged.head(5).index,
                        "Total Expenses (Million BRL)",
                        "City's Name",
                        "Top 5 Total Expenses by All Customers in Each City",
                        "payment_value_sum",
                        colors
                        )

        plt.close('all')
        
        df_0 = df_customer_merged.groupby(by='customer_id').agg({
                                                            'payment_value_sum': 'sum'
                                                            }).sort_values(by = ('payment_value_sum'), ascending = False).head(5)

        create_bar_chart(df_0,
                        df_0.index,
                        "Total Expenses (BRL)",
                        "Customer's ID",
                        "Top 5 Total Expenses by All Customers in Each City",
                        "payment_value_sum",
                        colors
                        )

        plt.close('all')

    with col2:

        state = st.selectbox(
            label="Choose State:",
            options=df_customer_state_merged.index,
            index=0
        )

        df_monthly_customer_state = create_df_monthly_customer_state(state, df_order_update, df_order_payments, df_customer)
        
        create_line_chart(df_monthly_customer_state, 
                        'payment_value', 
                        f'{state} State Monthly Expenses Trend', 
                        'Expenses (BRL)'
                        )

        plt.close('all')

        city = st.selectbox(
            label="Choose City:",
            options=df_customer_city_merged.index,
            index=0
        )

        df_monthly_customer_city = create_df_monthly_customer_city(city, df_order_update, df_order_payments, df_customer)
        
        create_line_chart(df_monthly_customer_city, 
                        'payment_value', 
                        f'{city.title()} City Monthly Expenses Trend', 
                        'Expenses (BRL)'
                        )

        plt.close('all')

if selected == "RFM ANALYSIS" or selected == "SHOW ALL":
    ## RFM Analysis
    st.html('<h4><span>RFM </span>ANALYSIS</h4>')
    monthly_transactions, daily_transactions = create_monthly_transactions(df_order_items_update)
    period = int((st.selectbox(label=f"Select Period: (today: {daily_transactions['year_month_day'].max()})",
                                options=("30 days ago", "60 days ago", "90 days ago"),
                                index=0))[:2])

    col1, col2, col3 = st.columns(3)

    with col1:

        rec_pivot_org = create_rec_pivot_org(period, daily_transactions)
        
        recency = int(str(rec_pivot_org['year_month_day'].sum())[1:-8])/rec_pivot_org['year_month_day'].count()
        
        value_ = math.floor(recency)
        st.metric(label=("Average Recency"),
                value=(f"{value_} Days ago"),
                border=True)

    with col2:

        freq_pivot_org = create_freq_pivot_org(period, daily_transactions)

        value_ = math.ceil(freq_pivot_org.sum()/freq_pivot_org.count())

        st.metric(label=("Average Frequency"),
                value=(f"{value_} Transactions"),
                border=True)

    with col3:

        monet_pivot_org = create_monet_pivot_org(period, df_customer_merged)

        value_ = math.ceil(monet_pivot_org.sum()/monet_pivot_org.count())

        st.metric(label=("Average Frequency"),
                value=(f"{value_} BRL"),
                border=True)
        
    col1, col2 = st.columns(spec=[0.4,0.6])

    with col1:
        
        df_rfm_clustering = create_df_rfm_clustering(period, rec_pivot_org, freq_pivot_org, monet_pivot_org)

        create_pie_chart(df_rfm_clustering, "Customer Priority Cluster")

        plt.close('all')

    with col2:
        st.html("""<p>This pie chart illustrates the distribution of customers into three priority categories based on the RFM (Recency, Frequency, Monetary) analysis:
                <li>1st Priority - Customers with high values in two or more RFM dimensions.</li>
                <li>2nd Priority - Customers with high values in one RFM dimension.</li>
                <li>3rd Priority - Customers with low values across all RFM dimensions.</li></p>""")

if selected == "PRODUCT ANALYSIS" or selected == "SHOW ALL":
    ## PRODUCT Analysis
    st.html('<h4><span>PRODUCT </span>ANALYSIS</h4>')

    input_kota = st.selectbox(
        label="Berapa kota yang ditampilkan?",
        options=(2, 3, 4, 5, 6, 7, 8),
        index=1
    )

    input_barang = st.selectbox(
        label="Berapa kategori barang yang ditampilkan?",
        options=(2, 3, 4, 5, 6, 7, 8, 9, 10),
        index=4
    )

    col1, col2 = st.columns(2)

    colors = ["#7e74f1", "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE",
            "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE", "#F5F3FE"]

    with col1:
        
        for i in range (input_kota):

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))    
            
            penjualan_kategoribarang_di_kota[i][1] = penjualan_kategoribarang_di_kota[i][1].reset_index().sort_values(by = ('count'), ascending = False)
            penjualan_kategoribarang_di_kota[i][1] = penjualan_kategoribarang_di_kota[i][1].head(input_barang)
            
            penjualan_kategoribarang_di_kota[i][1]['product_category_name_<lambda>'] = [i[:12]+'...' for i in penjualan_kategoribarang_di_kota[i][1]['product_category_name_<lambda>']]
            
            create_bar_chart(penjualan_kategoribarang_di_kota[i][1],
                            penjualan_kategoribarang_di_kota[i][1]['product_category_name_<lambda>'],
                            "Total Sales of Product (Units)",
                            "Product Category",
                            "Top 10 Sales by Product Category in " + penjualan_kategoribarang_di_kota[i][0].title(),
                            'count',
                            colors,
                            slicer=5)

            plt.close('all')

    with col2:
        
        for i in range (input_kota):

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))    
                
            pembelian_kategoribarang_di_kota[i][1] = pembelian_kategoribarang_di_kota[i][1].reset_index().sort_values(by = ('count'), ascending = False)
            pembelian_kategoribarang_di_kota[i][1] = pembelian_kategoribarang_di_kota[i][1].head(input_barang)
            
            pembelian_kategoribarang_di_kota[i][1]['product_category_name_<lambda>'] = [i[:12]+'...' for i in pembelian_kategoribarang_di_kota[i][1]['product_category_name_<lambda>']]

            create_bar_chart(pembelian_kategoribarang_di_kota[i][1],
                            pembelian_kategoribarang_di_kota[i][1]['product_category_name_<lambda>'],
                            "Total Purchases of Product (Units)",
                            "Product Category",
                            "Top 10 Purchases by Product Category in"+ " " + pembelian_kategoribarang_di_kota[i][0].title(),
                            'count',
                            colors,
                            slicer=5)
            
            plt.close('all')

if selected == "GEOSPATIAL ANALYSIS" or selected == "SHOW ALL":
    ## GEOSPATIAL Analysis
    st.html('<h4><span>GEOSPATIAL </span>ANALYSIS</h4>')

    col1, col2 = st.columns(2)

    with col1:

        st.html('<h5><span>CUSTOMER </span>SECTION</h5>')

        colors_map_cust = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",  # Primary Colors
            "#800000", "#808000", "#008000", "#000080", "#800080", "#808080",  # Dark and muted colors
            "#C0C0C0", "#FF6347", "#DDA000", "#FF1493", "#8A2BE2", "#7FFF00",  # Vivid colors
            "#D2691E", "#B22222", "#228B22", "#FF8C00", "#FA8072", "#A52A2A",  # More vibrant shades
            "#FFD700", "#F900B4", "#8B4513"   # Warm and natural colors
        ]

        create_map_brazil('customer_state', brazil_df, df_geo_point_cust, colors_map_cust)

        plt.close('all')
        
        state_map_select_cust = st.selectbox(
            label="Choose Customer State:",
            options=df_customer_state_merged['index+id'],
            index=0
        )[:2]

        create_map_state_customer(state_map_select_cust, brazil_df[['geometry', 'UF']], df_geo_point_cust['geometry'], colors_map_cust)

        plt.close('all')

        st.html('<h5>PRODUCT<span> DEMAND </span>SECTION</h5>')

        prod_cat_demand_select = st.selectbox(
            label="Choose Product Category:",
            options=prod_demand_counts['index+count'],
            index=0
        )

        # df_product_demand = create_df_product_demand(prod_cat_demand_select, df_geo_point_cust)

        create_map_brazil_product_dem('customer_state', brazil_df, prod_cat_demand_select, df_geo_point_cust, colors_map_cust)

        plt.close('all')

    with col2:

        st.html('<h5><span>SELLER </span>SECTION</h5>')

        colors_map_sel = [
            "#FF0000", "#00FF00", "#FF00FF", "#00FFFF",  # Primary Colors
            "#800000", "#808000", "#008000", "#000080", "#800080", "#808080",  # Dark and muted colors
            "#C0C0C0", "#FF6347", "#DDA000", "#FF1493", "#8A2BE2", "#7FFF00",  # Vivid colors
            "#D2691E", "#B22222", "#228B22", "#FF8C00", "#A52A2A",  # More vibrant shades
            "#FFD700", "#F900B4",  # Warm and natural colors
        ]

        create_map_brazil('seller_state', brazil_df, df_geo_point_sel, colors_map_sel)

        plt.close('all')

        state_map_select_sel = st.selectbox(
            label="Choose Seller State:",
            options=df_sellers_state_merged['index+id'],
            index=0
        )[:2]

        create_map_state_seller(state_map_select_sel, brazil_df[['geometry', 'UF']], df_geo_point_sel['geometry'], colors_map_cust)

        plt.close('all')

        st.html('<h5>PRODUCT<span> SUPPLY </span>SECTION</h5>')

        prod_supply_counts = create_prod_supply_counts(df_geo_point_sel)

        prod_cat_supply_select = st.selectbox(
            label="Choose Product Category:",
            options=prod_supply_counts['index+count'],
            index=0
        )

        create_map_brazil_product_sup('seller_state', brazil_df, prod_cat_supply_select, df_geo_point_sel, colors_map_cust)

        plt.close('all')









## CUSTOMER & SELLER Analysis (CLUSTERING)
# st.markdown('#### CUSTOMER & SELLER ANALYSIS')

# col1, col2 = st.columns(2)

# with col1:
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3.5))

#     colors = ('#C46100', '#EF9234', '#F4B678', '#F9E0A2', '#F4B678', '#EF9234')
#     explode = (0.05, 0.07, 0.09, 0.11, 0.13, 0.17)

#     klaster = kumpulan_klaster[:5]
#     klaster.append("Klaster VI dan VII")
#     count = (df_customer_klaster['customer_id_count'][0],
#             df_customer_klaster['customer_id_count'][1],
#             df_customer_klaster['customer_id_count'][2],
#             df_customer_klaster['customer_id_count'][3],
#             df_customer_klaster['customer_id_count'][4],
#             df_customer_klaster['customer_id_count'][5]+df_customer_klaster['customer_id_count'][6]
#             )

#     ax.pie(
#         x=count,
#         labels=klaster,
#         autopct='%1.1f%%',
#         colors=colors,
#         explode=explode,
#         wedgeprops = {'width': 0.5}
#         )
#     ax.set_title('Klaster Customer')

#     # plt.tight_layout()
#     st.pyplot(plt)
#     plt.close('all')

# with col2:

#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 2.5))

#     klaster = kumpulan_klaster
#     count = (df_sellers_klaster['seller_id_count'][0],
#             df_sellers_klaster['seller_id_count'][1],
#             df_sellers_klaster['seller_id_count'][2],
#             df_sellers_klaster['seller_id_count'][3],
#             df_sellers_klaster['seller_id_count'][4],
#             df_sellers_klaster['seller_id_count'][5],
#             df_sellers_klaster['seller_id_count'][6]
#             )
#     colors = ('#8F4700', '#C46100', '#EF9234', '#F4B678', '#F9E0A2', '#F4B678', '#C46100')
#     explode = (0.02, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13)
#     ax.pie(
#         x=count,
#         labels=klaster,
#         autopct='%1.1f%%',
#         colors=colors,
#         explode=explode,
#         wedgeprops = {'width': 0.5}
#         )
#     ax.set_title("Klaster Seller",)

#     # plt.tight_layout()
#     st.pyplot(fig)
#     plt.close('all')
