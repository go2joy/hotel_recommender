
#!pip install pycountry
import sys
from os.path import dirname, join, realpath
C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)

sys.path.insert(0,P_DIR)

import numpy as np
import pandas as pd
import pandas
# import ray
# ray.init(num_cpus=6, ignore_reinit_error=True)
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


def get_data():
    db = MyDatabase()
    results = []
    for query in lst_query:
        data = None      
        data = db.ses_execute_query(query)
        results.append(data)
        
    user_booking = results[0]
    data_hotel = results[1]
    v_hotel_setting = results[2]
    province= results[3]
    district= results[4]
    app_user= results[5]
    room_type=results[6]
    return user_booking, data_hotel, v_hotel_setting,province, district, app_user, room_type


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


def get_data_user_booking(user_booking, v_hotel_setting, lst_hotel, province, district, booking_type = [1, 2, 3], booking_status = [0, 1, 2, 3,4,5]):
    user_booking = user_booking.loc[(user_booking['CREATE_TIME'] >= '2020-01-01')]
    tbs_user_booking = user_booking.merge(v_hotel_setting, how = 'left', left_on='HOTEL_SN', right_on='HOTEL_SN')
    tbs_user_booking = tbs_user_booking.query('TYPE in @booking_type and BOOKING_STATUS in @booking_status')
    tbs_user_booking = split_booking_time(tbs_user_booking)

    tbs_user_booking = tbs_user_booking[(tbs_user_booking["HOTEL_SN"]!=467)]

    tbs_user_booking = tbs_user_booking.dropna(how = 'any', subset=['b_time'])
    tbs_user_booking = tbs_user_booking.merge(lst_hotel, how = 'left', left_on='HOTEL_SN', right_on='SN', suffixes=('', '_HOTEL'))

    tbs_user_booking = tbs_user_booking.merge(province.loc[:, ['SN', 'NAME']], how = 'left', left_on='PROVINCE_SN', right_on ='SN', suffixes = ('', '_PROVINCE'))
    tbs_user_booking = tbs_user_booking.merge(district.loc[:, ['SN', 'NAME']], how = 'left', left_on='DISTRICT_SN', right_on ='SN', suffixes = ('', '_DISTRICT'))
    # Discard the columns that acquired a suffix
    tbs_user_booking = tbs_user_booking[[c for c in tbs_user_booking.columns if not c.endswith('_delme')]]

    tbs_user_booking = split_datetime(tbs_user_booking)
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
    # group_hotel_booking = group_hotel_booking[group_hotel_booking['total_booking_hotel'] > 50]

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


def calculate_distance(latitude1 = 21.012758, longitude1 = 105.800766, latitude2 = 21.010260, longitude2=105.865092):
    # approximate radius of earth in km
    R = 6373.0
    lat1 = radians(latitude1)
    lon1 = radians(longitude1)
    lat2 = radians(latitude2)
    lon2 = radians(longitude2)

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
   

def hotel_distribution(lst_hotel, export = False):
    # carto-positron , stamen-terrain
    fig = px.density_mapbox(lst_hotel, lat='LATITUDE', lon='LONGITUDE', radius=3,center=dict(lat=16, lon=106), \
        zoom=5,mapbox_style="carto-positron", width= 1000, height=900, hover_name = "NAME", \
            hover_data ={'LATITUDE':False # remove from hover data
                                         ,'LONGITUDE':False # remove from hover data
                                        })
    fig.show()
    if export:
        fig.write_html("heapmap - distribution hotels - zoom to detail.html")
      
        
def hotel_clustering(df, k = 4, max_iter = 1000):
    # finding the clusters based on input matrix "x"
    model = KMeans(n_clusters = k, init = "k-means++", max_iter = max_iter, n_init = 10, random_state = 0)
    y_clusters = model.fit_predict(df)
    return model, y_clusters        



def get_hotel_price(data_room_type):
    """
    return: dataframe hotel avg_price by booking type (PRICE_ONE_DAY, PRICE_OVERNIGHT, PRICE_PER_HOUR)
    """
    try:
        g_room_type = data_room_type.groupby(['HOTEL_SN','FIRST_HOURS', 'PRICE_FIRST_HOURS', \
                            'PRICE_ADDITIONAL_HOURS', 'PRICE_OVERNIGHT', 'PRICE_ONE_DAY'])['HOTEL_SN'].count().to_frame('num_of_room_hotel')
        #                     .filter(lambda x: df['HOTEL_SN'].count()> 1)#agg({'SN':sum}) #['FIRST_HOURS']

        df_room_type = pd.DataFrame(g_room_type).reset_index()
        df_room_type = df_room_type.set_index(df_room_type['HOTEL_SN'])
        df_room_type['PRICE_PER_HOUR'] = df_room_type.apply(lambda row: row['PRICE_FIRST_HOURS']/row['FIRST_HOURS'], axis=1)
        df_room_type.sort_values(by='FIRST_HOURS', ascending=True)
        df_room_type= df_room_type.query('PRICE_PER_HOUR > 10000 & PRICE_ONE_DAY > 50000 & PRICE_OVERNIGHT> 50000 & FIRST_HOURS > 0')

        Data = {'x': df_room_type['PRICE_ONE_DAY'],
                'y': df_room_type['PRICE_OVERNIGHT'],
                'z': df_room_type['PRICE_PER_HOUR']
               }

        df = pd.DataFrame(Data,columns=['x','y', 'z'])
        df = df.groupby(['HOTEL_SN']).agg({'x': 'mean', 'y': 'mean', 'z': 'mean'})
        df = remove_noise_by_percentile(df, 5, 95, lst_col=['x']).dropna()    
        return df
    except Exception as ex:
        logger.exception(ex)
        return None


def agv_price(row):
    if row['FIRST_HOURS'] == 1:
        return row['PRICE_FIRST_HOURS']
    elif row['ADDITIONAL_HOURS'] == 1:
        tmp = row['PRICE_FIRST_HOURS'] -  ((row['FIRST_HOURS'] -1) * row['PRICE_ADDITIONAL_HOURS'])
        return tmp
    else:
        return 0
    
def preprocess_room_type_history(data_room_type_history, data_hotel, province, district): #data_room_type_history, recommender.data_hotel
    df = data_room_type_history.merge(data_hotel.loc[:, ['SN', 'NAME', 'PROVINCE_SN', 'DISTRICT_SN', 'LATITUDE', 'LONGITUDE']], how = 'left', left_on='HOTEL_SN', right_on ='SN', suffixes = ('', '_hotel'))
    df = df.merge(province.loc[:, ['SN', 'NAME']], how = 'left', left_on='PROVINCE_SN', right_on ='SN', suffixes = ('', '_province'))
    df = df.merge(district.loc[:, ['SN', 'NAME']], how = 'left', left_on='DISTRICT_SN', right_on ='SN', suffixes = ('', '_district'))
    df['DATE_CHANGE'] = df['CREATE_TIME'].apply(lambda x: str(x)[:10])
    df = df.query('PRICE_ONE_DAY > 50000 & PRICE_OVERNIGHT> 50000 & PRICE_FIRST_HOURS> 20000 & \
    PRICE_FIRST_HOURS< 5000000 & PRICE_ADDITIONAL_HOURS < 1000000')
    df = df.dropna(subset=['PROVINCE_SN', 'DISTRICT_SN'])
    df = df.query('DISTRICT_SN != 121')
    df = df.query('HOTEL_SN not in [467]')
    df['PRICE_FIRST_HOURS'] = df.apply(agv_price, axis=1) # calculate avg price each hotel (hourly booking)
    return df