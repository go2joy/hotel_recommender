#!/usr/bin/env python
# coding: utf-8
import sys
from os.path import dirname, join, realpath, isfile, basename
C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)

sys.path.insert(0,C_DIR)
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()


import numpy as np
import pandas as pd
# import ray
# ray.init(num_cpus=6, ignore_reinit_error=True)
# import modin.pandas as pd  #pip install modin[ray]
import matplotlib.pyplot as plt
from utils.df_utils import *
from utils.util import logger, timeit
from db.database import MyDatabase
from db.sql_query import get_app_user, get_booking, get_hotel, get_lst_hotel, get_user_booking

import time
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from math import radians
import plotly.express as px
from pandas.io.pytables import *
import os
from memory_profiler import profile
pd.set_option('io.hdf.default_format','table')
pd.options.display.max_columns = None
from os.path import join
from config.config import Cfg



class HotelRecommender():
    def __init__(self,config_file='./config/config.yml'):
        super().__init__()
        self.config = Cfg.load_config_from_file(config_file)
        try:
            self.hdfs_name = 'hdfs_tmp.h5'
            self.store_hdfs = pd.HDFStore(self.hdfs_name)
            self.user_booking, self.data_hotel = get_data()
            logger.info("get data successful...")
        except Exception as ex:
            logger.warning(ex)
                
        self.lst_hotel= clean_data(self.data_hotel)
        self.data_user_booking = get_data_user_booking(self.user_booking,self.lst_hotel, \
            booking_type = [1, 2], booking_status = [2])      #booking_type= [1, 2, 3], booking_status = [0, 1, 2, 3,4,5]   

        self.datetime = datetime.now()
        self.dis_matrix = self.build_distance_metric()
        logger.info("init model...")
        

    def hotel_cluster_by_location(self,cleaned_data):
        kmeans = KMeans(n_clusters = 2, max_iter=1000, init ='k-means++')
        lat_long = cleaned_data[['LONGITUDE', 'LATITUDE']]
        # lot_size = X_weighted[X_weighted.columns[3]]
        weighted_kmeans_clusters = kmeans.fit(lat_long) # Compute k-means clustering.
        cleaned_data['CLUSTER'] = kmeans.predict(lat_long)
        return cleaned_data
 
        
    def do_pivot(self,):
        user_booking = self.data_user_booking.groupby(['APP_USER_SN']).agg(total_booking= ('SN', 'count'), \
            count_unique_hotel= ('HOTEL_SN', pd.Series.nunique), \
            booked_hotels = ('HOTEL_SN', lambda x: ', '.join(str(v) for v in x))).reset_index()

        user_booking = user_booking.query('total_booking > 0').sort_values(by='total_booking', ascending=False)
        lst_app_user_sn = user_booking[user_booking['total_booking'] > 0]['APP_USER_SN']
        cleaned_data = self.data_user_booking.query('APP_USER_SN in @lst_app_user_sn')
        cleaned_data = cleaned_data[['SN', 'APP_USER_SN', 'HOTEL_SN', 'LONGITUDE', 'LATITUDE']]
        user_booking_pivot = self.pivot_big_df(cleaned_data, num_row=50000, is_save = True)

        user_booking_pivot['APP_USER_SN'] = user_booking_pivot.index
        X = user_booking_pivot.drop(['APP_USER_SN'], axis=1)

        similars = cosine_similarity(X)

        for item in similars:
            sort_index= np.argsort(item.reshape(1, -1), axis=1)
        #     print("max score: ", item[sort_index.flatten()[-2]], "user: ", (X.iloc[sort_index.flatten()[-1], :].name, X.iloc[sort_index.flatten()[-7: -1], :].index))
            
            
            
    def pivot_segment(self, segmentNumber, segmentSize,passedFrame):
        frame = passedFrame[(segmentNumber*segmentSize):(segmentNumber*segmentSize + segmentSize)] #slice the input DF

    #     # ensure that all chunks are identically formatted after the pivot by appending a dummy DF with all possible category values
        span = pd.DataFrame() 
        span['HOTEL_SN'] = passedFrame['HOTEL_SN'].unique()
        span['APP_USER_SN']=999999999
        span['SN']=0

        frame = frame.append(span)
        return frame.pivot_table('SN',index='APP_USER_SN',columns='HOTEL_SN', \
                                aggfunc='count',fill_value=0).reset_index()


    # @profile
    def pivot_big_df(self, dataframe, num_row=50000, is_save = True):
        segMax = dataframe.shape[0]/num_row
        for i in range(int(segMax) + 1):
            segment = self.pivot_segment(i, num_row, dataframe)
        #     store.append('data',frame[(i*num_row):(i*num_row + num_row)])
            self.store_hdfs.append('pivotedData',segment)
        df = self.store_hdfs['pivotedData'].set_index('APP_USER_SN').groupby('APP_USER_SN',level=0).sum()
        df = df[df.index != 999999999]
        self.store_hdfs.close()
        os.remove(self.hdfs_name)
        if is_save:
            df.to_csv("df_recommender.csv", index=True)
        return df


    def build_distance_metric(self):
        hotels_df = self.data_hotel.loc[:, ['SN', 'LATITUDE', 'LONGITUDE']]
        hotels_df = hotels_df.dropna(how='any')
        hotels_df = hotels_df.drop_duplicates(keep='last')

        hotels_df['lat'] = np.radians(hotels_df['LATITUDE'])
        hotels_df['lon'] = np.radians(hotels_df['LONGITUDE'])

        dist = DistanceMetric.get_metric('haversine')
        dis_matrix = pd.DataFrame(dist.pairwise(hotels_df[['lat','lon']].to_numpy())*6373,  columns=hotels_df.SN.unique(), index=hotels_df.SN.unique())
        dis_matrix.columns = dis_matrix.columns.map(lambda x: "_"+str(x))
        return dis_matrix

    def get_hotels_by_radius(self, hotel_sn, radius=3):
        try:
            hotel_sn_1 = self.dis_matrix.loc[:, [hotel_sn]]
            return list(hotel_sn_1.query('{} <= @radius'.format(hotel_sn)).sort_values(by=hotel_sn, ascending=True).index)
        except Exception as ex:
            logger.error(ex)
            return []
        
    def get_n_hotels_nearest(self, hotel_sn, hotel_num = 50):
        try:
            hotel_sn_1 = self.dis_matrix.loc[:, [hotel_sn]]
            return list(hotel_sn_1.sort_values(by=hotel_sn, ascending=True).index)[1:hotel_num]
        except Exception as ex:
            logger.error(ex)
            return []




if __name__=="__main__":

    recommender = HotelRecommender()
