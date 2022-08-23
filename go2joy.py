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
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()


import numpy as np
import pandas as pd
# import ray
# ray.init(num_cpus=6, ignore_reinit_error=True)
# import modin.pandas as pd  #pip install modin[ray]
import matplotlib.pyplot as plt
import seaborn as sns
from utils.df_utils import *
from utils.util import logger, timeit
from db.database import MyDatabase
from db.sql_query import get_app_user, get_booking, get_hotel, get_lst_hotel, get_user_booking
pd.options.display.max_columns = None
import time
from sklearn.neighbors import DistanceMetric
from math import radians
import plotly.express as px

from os.path import join
import os
from config.config import Cfg



class HotelRecommender():
    def __init__(self,config_file='./config/config.yml'):
        super().__init__()
        self.config = Cfg.load_config_from_file(config_file)
        try:

            self.user_booking, self.data_hotel, self.v_hotel_setting, self.province, self.district, self.app_user, self.room_type = get_data()
            logger.info("get data successful...")
        except Exception as ex:
            logger.warning(ex)
                
        self.lst_hotel= clean_data(self.data_hotel)
        self.data_user_booking = get_data_user_booking(self.user_booking, self.v_hotel_setting, self.lst_hotel, self.province, \
            self.district, booking_type = [1], booking_status = [2])      #booking_type= [1, 2, 3], booking_status = [0, 1, 2, 3,4,5]   
        
        df_tmp = get_hotel_price(self.room_type)
        self.k_model, self.y_clusters = hotel_clustering(df_tmp, k = 4, max_iter = 1000) 
        self.datetime = datetime.now()
        self.dis_matrix = self.build_distance_metric()
        logger.info("init model...")
        

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
