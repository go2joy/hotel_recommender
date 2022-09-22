
#!pip install pycountry
from operator import index
import os
os.environ["MODIN_ENGINE"] = "dask" 
import sys
from os.path import dirname, join, realpath
C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)

sys.path.insert(0,P_DIR)

import numpy as np
import pandas as pd
# Importing dask dataframe
# import dask
# import dask.dataframe as pd

# import ray
# ray.init(num_cpus=4, ignore_reinit_error=True)
# import modin.pandas as pd  #pip install modin[ray]

import matplotlib.pyplot as plt
import seaborn as sns
from .util import logger
from os.path import join
from sklearn import preprocessing
from math import sin, cos, sqrt, atan2, radians
import plotly.express as px
from db.database import MyDatabase
from db.sql_query import lst_query
from sklearn.cluster import KMeans
# import the libraries
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import plotly.express as px
import datetime

data_path = join(P_DIR, 'data')

def get_data(n_samples = 1000):
    db = MyDatabase()
    results = []
    for query in lst_query:
        data = None      
        data = db.ses_execute_query(query)
        results.append(data)
        
    user_booking = results[0]
    if n_samples> 0:
        user_booking = user_booking[-n_samples:]
    data_hotel = results[1]
    user_booking.to_csv("user_booking.csv", index=False)
    data_hotel.to_csv("data_hotel.csv", index=False)
    return user_booking, data_hotel


def combine_booking_time(row):
    try:
        if int(row['TYPE']) == 1:
            return pd.Timestamp(row['CHECK_IN_DATE_PLAN']) + pd.Timedelta(row['START_TIME'])
        else:
            return pd.Timestamp(row['CHECK_IN_TIME'])        
    except:
        return pd.Timestamp('2017-01-01 01:01:01')


def combine_end_time(row):
    try:
        if int(row['TYPE']) == 1:
            return pd.Timestamp(row['CHECK_IN_DATE_PLAN']) + pd.Timedelta(row['END_TIME'])
        else:
            return pd.Timestamp(row['CHECK_IN_DATE_PLAN']) + pd.Timedelta(hours = row['CHECKIN_TIME'])     #       
    except:
        return pd.Timestamp('2017-01-01 01:01:01')

def count_x(series):
    return series.count()        
           
def split_booking_time(data_user_booking): #only for hourly
    try:
        df = data_user_booking.query('TYPE == 1')
        df = df.dropna(how = 'any', subset=['CHECK_IN_DATE_PLAN']) #'START_TIME'
        df['b_time'] = df.apply(combine_booking_time, axis=1)
        df['duration'] = df.apply(lambda row: round((pd.Timedelta(row['END_TIME']) - pd.Timedelta(row['START_TIME'])).total_seconds()/3600), axis=1)
        
        df['waiting_time'] = df.apply(lambda row: round((pd.Timestamp(row['b_time']) - \
                                                pd.Timestamp(row['CREATE_TIME'])).total_seconds()), axis=1)
        
    except Exception as ex:
        logger.exception(ex)
        pass
    try:
        
        df_others = data_user_booking.query('TYPE != 1')
        df_others.CHECK_IN_DATE_PLAN = df_others.CHECK_IN_DATE_PLAN.apply(lambda x: pd.to_datetime(x,  errors = 'coerce'))
        df_others = df_others.dropna(how = 'any', subset=['CHECK_IN_DATE_PLAN']) #'START_TIME'
        df_others['b_time'] = df_others['CHECK_IN_DATE_PLAN']#.apply(lambda x: pd.Timedelta(x))

    except Exception as ex:
        logger.exception(ex)
    return pd.concat([df,df_others], axis=0, ignore_index=True)


def get_hour_weekday(_datetime=pd.Timestamp.today(tz=None)):
    return int(_datetime.hour), _datetime.ctime()[:3]

def split_datetime(df):
    try:      
        df['b_year'] = df.b_time.apply(lambda x: int(x.year))
        df['b_month'] = df.b_time.apply(lambda x: int(x.month))
        df['b_day'] = df.b_time.apply(lambda x: int(x.day))
        df['b_hour'] = df.b_time.apply(lambda x: int(x.hour))
        df['b_minute'] = df.b_time.apply(lambda x: int(x.minute))
        df['b_weekday'] = df.b_time.apply(lambda x: x.ctime()[:3])
    except Exception as ex:
        logger.exception(ex)
    return df
    
def clean_data(df):
    df = df.dropna(how = 'any', subset=['LONGITUDE', 'LATITUDE']) #['longitude', 'latitude']
    df = df.drop_duplicates(keep='last')
    return df


# remove data by percentile
def remove_noise_by_percentile(df, q_min, q_max, lst_col=['waiting_time']):
    for x in lst_col:
        q75,q25 = np.percentile(df.loc[:,x],[q_max,q_min])
        intr_qr = q75-q25

        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)

        df.loc[df[x] < min,x] = np.nan
        df.loc[df[x] > max,x] = np.nan
        df = df.dropna()
    return df


def get_data_user_booking(user_booking, lst_hotel, booking_type = [1, 2, 3], booking_status = [0, 1, 2, 3,4,5]):
    # user_booking = user_booking.loc[(user_booking['CREATE_TIME'] >= '2020-01-01')]
    # tbs_user_booking = user_booking.merge(v_hotel_setting, how = 'left', left_on='HOTEL_SN', right_on='HOTEL_SN')
    tbs_user_booking = user_booking.query('TYPE in @booking_type and BOOKING_STATUS in @booking_status')
    # tbs_user_booking = split_booking_time(tbs_user_booking)

    # tbs_user_booking = tbs_user_booking[(tbs_user_booking["HOTEL_SN"]!=467)]
    # tbs_user_booking = tbs_user_booking.dropna(how = 'any', subset=['b_time'])
    # tbs_user_booking = tbs_user_booking.merge(lst_hotel, how = 'left', left_on='HOTEL_SN', right_on='SN', suffixes=('', '_HOTEL'))
    # tbs_user_booking = tbs_user_booking[[c for c in tbs_user_booking.columns if not c.endswith('_delme')]]
    # tbs_user_booking = split_datetime(tbs_user_booking)
    tbs_user_booking = tbs_user_booking.dropna(how='any', subset=['APP_USER_SN'])
    return tbs_user_booking


def get_df_groupby(df, key = 'HOTEL_SN'):
    data = split_datetime(df)
    data = data.groupby([key]).size().to_frame('total_booking_by_{}'.format(key.lower())).reset_index()
    return data



def preprocess_data_booking(df):
    _weekday = df.groupby(['b_weekday']).size().to_frame('count_').reset_index()
    _hour = df.groupby(['b_hour']).size().to_frame('count_').reset_index()
    _year = df.groupby(['b_year']).size().to_frame('count_').reset_index()
    _month = df.groupby(['b_month']).size().to_frame('count_').reset_index()
    _duration = df.groupby(['duration']).size().to_frame('count_').reset_index()
    return _year, _month, _weekday, _hour, _duration

def process_data_hotel(data_user_booking, lst_hotel):
    group_hotel_booking = data_user_booking.groupby(['HOTEL_SN']).size().to_frame('total_booking_hotel').reset_index()
    group_hotel_booking = group_hotel_booking.sort_values(by=['total_booking_hotel'], ascending=False)
    data_hotel = group_hotel_booking.merge(lst_hotel, how = 'left', left_on='HOTEL_SN', right_on='SN', suffixes=('', '_delme'))
    # Discard the columns that acquired a suffix
    data_hotel = data_hotel[[c for c in data_hotel.columns if not c.endswith('_delme')]]
    data_hotel.total_booking_hotel = data_hotel.total_booking_hotel.fillna(0)

    # data_hotel['log_total'] =  data_hotel.total_booking_hotel.apply(lambda x: np.log2(x) if x > 0 else 0)
    data_hotel['log_total'] =  data_hotel.total_booking_hotel.apply(lambda x: x/1000 if x > 0 else 0)
    return data_hotel

        
def scale_df(df):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)
    
    return df_normalized


def calculate_distance(lon1 = 105.800766, lat1 = 21.012758, lon2=105.865092, lat2 = 21.010260):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance

# room_type.query('NUM_OF_ROOM > 1')
def get_average_price(room_type, lst_hotel_hourly):
    avg_hour_price_hotel = room_type.groupby(['HOTEL_SN'])['PRICE_FIRST_HOURS'].mean().to_frame('avg_hour_price_by_hotel').reset_index()
    total_room_per_hotel = room_type.groupby(['HOTEL_SN']).agg({'NUM_OF_ROOM':sum}).reset_index()
    avg_hour_price_hotel = avg_hour_price_hotel.query('avg_hour_price_by_hotel > 30000')
    total_room_per_hotel = total_room_per_hotel.query('NUM_OF_ROOM > 0')

    lst_hotel_hourly_2 = lst_hotel_hourly.merge(avg_hour_price_hotel, how='left', left_on='HOTEL_SN', right_on='HOTEL_SN', suffixes=('', '_delme'))
    lst_hotel_hourly_2 = lst_hotel_hourly_2.merge(total_room_per_hotel, how='left', left_on='HOTEL_SN', right_on='HOTEL_SN', suffixes=('', '_delme'))
    lst_hotel_hourly_2 = lst_hotel_hourly_2[[c for c in lst_hotel_hourly_2.columns if not c.endswith('_delme')]]    
    
    return lst_hotel_hourly_2   

        
def hotel_clustering(df, k = 4, max_iter = 1000):
    # finding the clusters based on input matrix "x"
    model = KMeans(n_clusters = k, init = "k-means++", max_iter = max_iter, n_init = 10, random_state = 0)
    y_clusters = model.fit_predict(df)
    return model, y_clusters        


def read_data_from_file():
    user_booking = pd.read_csv(join(data_path,'user_booking.csv'), low_memory=False)
    data_hotel = pd.read_csv(join(data_path,'data_hotel.csv'), low_memory=False)
    
    return user_booking, data_hotel