#!/usr/bin/env python
# coding: utf-8
import imp
import re
import sys
import os
from os.path import dirname, join, realpath, isfile, basename

C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)

os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"]="1"
os.environ["RAY_DISABLE_MEMORY_MONITOR"]="1"
os.environ["MODIN_ENGINE"] = "dask" 

sys.path.insert(0,C_DIR)
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

from multiprocessing import cpu_count
import numpy as np
import pandas as pd
# Importing dask dataframe
# import dask
# import dask.dataframe as pd

# import ray
# ray.init(num_cpus=4, ignore_reinit_error=True)
# import modin.pandas as pd  #pip install modin[ray]

import matplotlib.pyplot as plt
from utils.df_utils import *
from utils.util import logger, timeit, parallel
from utils.apriori import Apriori
from db.database import MyDatabase
from db.sql_query import get_app_user, get_booking, get_hotel, get_lst_hotel, get_user_booking

import time
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN

from math import radians
import plotly.express as px
from pandas.io.pytables import *

from memory_profiler import profile

from os.path import join
from config.config import Cfg
from itertools import chain
import hdbscan
from functools import reduce


class HotelRecommender():
    @timeit
    def __init__(self,config_file='./config/config.yml', n_samples = 1000):
        super().__init__()
        self.config = Cfg.load_config_from_file(config_file)
        try:
            self.hdfs_name = 'hdfs_tmp.h5'
            os.remove(self.hdfs_name)
            self.store_hdfs = pd.HDFStore(self.hdfs_name)
        except Exception as ex:
            logger.warning(ex)
        self.apriori = None
        self.user_booking, self.data_hotel = get_data(n_samples)
        logger.info("get data successful...")                     
        self.lst_hotel= clean_data(self.data_hotel)
        self.data_user_booking = get_data_user_booking(self.user_booking,self.lst_hotel, \
            booking_type = [1, 2], booking_status = [2])      #booking_type= [1, 2, 3], booking_status = [0, 1, 2, 3,4,5]   

        self.dis_matrix = self.build_distance_metric()
        
        self.user_lst_hotel = self.data_user_booking.groupby(['APP_USER_SN']).agg(total_booking= ('SN', 'count'), \
            count_unique_hotel= ('HOTEL_SN', pd.Series.nunique), \
            booked_hotels = ('HOTEL_SN', lambda x: list(set([int(v) for v in x]))) \
                ).reset_index()
     
        self.hotel_lst_user = self.data_user_booking.groupby(['HOTEL_SN']).agg(count_unique_user= ('APP_USER_SN', pd.Series.nunique), \
                total_booked= ('SN', 'count'), \
                booked_users = ('APP_USER_SN', lambda x: [int(v) for v in x])).reset_index()
                
        self.assoc_rule = self.run_apriori(self.user_lst_hotel)
        
        self.model = dict()        
        logger.info("init model...")

    @timeit
    def merge_dicts(self, dict_):
        for key in dict_:
            try:
                self.model[key].append(dict_[key])
            except KeyError:
                self.model[key] = [dict_[key]]

    @timeit
    def get_hotels_of_user(self, lst_user_sn):
        try: 
            lst_hotel = self.user_lst_hotel.query('APP_USER_SN in @lst_user_sn')['booked_hotels'].values.tolist()
            lst_hotel = [item for sublist in lst_hotel for item in sublist]
            return list(set(sorted(lst_hotel, key = lst_hotel.count, reverse = True)))
        except Exception as ex:
            return []  
        


    # @parallel(merge_func=lambda li: sorted(set(chain(*li))))
    @timeit
    def build_user_centroid(self, row):
        lst_hotel = self.get_hotels_of_user([row['APP_USER_SN']])
        centroid = self.data_hotel.query("SN in @lst_hotel")[['LONGITUDE', 'LATITUDE']].mean()
        return centroid

    @timeit
    def clustering_dbscan(self, lst_hotel, eps = 0.1, min_samples = 3): #to remove noise 
        df = self.data_hotel.query("SN in @lst_hotel")[['LONGITUDE', 'LATITUDE']]
        geo_df = df.dropna(subset=['LONGITUDE', 'LATITUDE'], how='any')
        X = geo_df[['LONGITUDE', 'LATITUDE']] #.as_matrix(columns=['LONGITUDE', 'LATITUDE'])
        db = DBSCAN(eps= eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(np.radians(X))

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)   


    @timeit
    def clustering_hdbscan(self, df, min_cluster_size=2, min_samples=2, cluster_selection_epsilon=0.01):
        X = np.array(df[['LONGITUDE', 'LATITUDE']], dtype='float64')
        model = hdbscan.HDBSCAN(min_cluster_size= min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, )
        class_predictions = model.fit_predict(X)
        df['CLUSTER_HDBSCAN'] = class_predictions 
        # remove all the observations with cluster -1 (outliers)
        df = df[df['CLUSTER_HDBSCAN'] >= 0]
        medians_of_user = df.groupby(['CLUSTER_HDBSCAN'])[['LONGITUDE', 'LATITUDE']].median().reset_index(drop=False)
       
        return medians_of_user


    # @parallel(merge_func=lambda li: sorted(set(chain(*li))))
    @timeit
    def get_users_in_hotel(self, lst_hotel_sn):
        try:
            lst_user = self.hotel_lst_user.query('HOTEL_SN in @lst_hotel_sn')['booked_users'].values.tolist()[0]
            return sorted(lst_user, key = lst_user.count, reverse = True)
        except Exception as ex:
            return []    

    @timeit
    def hotel_cluster_by_location(self,cleaned_data):
        kmeans = KMeans(n_clusters = 2, max_iter=1000, init ='k-means++')
        lat_long = cleaned_data[['LONGITUDE', 'LATITUDE']]
        # lot_size = X_weighted[X_weighted.columns[3]]
        weighted_kmeans_clusters = kmeans.fit(lat_long) # Compute k-means clustering.
        cleaned_data['CLUSTER'] = kmeans.predict(lat_long)
        return cleaned_data
    
    # @parallel(merge_func=lambda li: sorted(set(chain(*li))))     
    @timeit  
    def build_model(self, num_clusters = 2):
        user_lst_hotel = self.user_lst_hotel.query('total_booking > 0').sort_values(by='total_booking', ascending=False)
        lst_app_user_sn = user_lst_hotel[user_lst_hotel['total_booking'] > 0]['APP_USER_SN']
        cleaned_data = self.data_user_booking.query('APP_USER_SN in @lst_app_user_sn')
        cleaned_data = cleaned_data[['SN', 'APP_USER_SN', 'HOTEL_SN', 'LONGITUDE', 'LATITUDE']]
        cleaned_data = cleaned_data.astype({'SN':'uint32', 'APP_USER_SN': 'uint32', 'HOTEL_SN': 'uint32', 'LONGITUDE':'float16', 'LATITUDE':'float16'})
        
        # user_booking_pivot = self.pivot_big_df(cleaned_data, num_row=50000, is_save = True)
        # user_booking_pivot['APP_USER_SN'] = user_booking_pivot.index
        # X = user_booking_pivot.drop(['APP_USER_SN'], axis=1)
        # similars = cosine_similarity(X)

        kmeans = KMeans(n_clusters = num_clusters, max_iter=1000, init ='k-means++')
        lat_long = cleaned_data[cleaned_data.columns[3:5]]
        weighted_kmeans_clusters = kmeans.fit(lat_long) # Compute k-means clustering.
        cleaned_data['CLUSTER'] = kmeans.predict(lat_long)        

        for cluster_id in range(num_clusters):
            tmp_data = cleaned_data.query('CLUSTER == @cluster_id')
            X = tmp_data.pivot_table('SN', index = 'APP_USER_SN', columns='HOTEL_SN', aggfunc='count', ).fillna(0)
            
            dict_ = dict()
            similars = cosine_similarity(X)
            user_index = X.index

            for idx, item in enumerate(similars):
                sort_index= np.argsort(item.reshape(1, -1), axis=1).flatten()[-10: ][::-1]
                # users = list(X.iloc[sort_index, :].index)
                users = list(user_index[sort_index])
                score = list(item[sort_index])
                # dict_[X.iloc[idx, :].name] = {"idx": X.iloc[idx, :].name, "users": users, "score": score}
                dict_[user_index[idx]] = {"idx": user_index[idx], "users": users, "score": score}

                del item, users, score
            del X, tmp_data, similars
            self.merge_dicts(dict_)
        del lat_long, cleaned_data 
        return None

    @timeit
    def predict(self, user_sn, min_distance=2):
        lst_res = []
        lst_user = []
        df_median = None
        hotels_of_user = self.get_hotels_of_user([user_sn])
        df_hotels = self.data_hotel.query('SN in @hotels_of_user')[['LONGITUDE', 'LATITUDE']]
        
        if df_hotels.shape[0] > 3:
            df_median = self.clustering_hdbscan(df_hotels,min_cluster_size= 2, min_samples= max(2,int(0.5 * df_hotels.shape[0])))
        if df_median is None or df_median.shape[0] == 0:
            df_median = [df_hotels[['LONGITUDE', 'LATITUDE']].median().reset_index(drop=True).values.tolist()]    

        if len(self.model[user_sn]) > 1:
            for item in self.model[user_sn]: 
                lst_user.extend(item['users'])
            lst_user = list(set(lst_user))
        else:
            lst_user = self.model[user_sn][0]['users']
        
            
        # lst_hotel = self.user_lst_hotel.query('APP_USER_SN in @lst_user')['booked_hotels'].values.tolist()
        # # sorting on basis of frequency of elements
        # result = sorted(lst_hotel, key = lst_hotel.count, reverse = True)
        hotel_of_neighbors = self.get_hotels_of_user(lst_user)
        
        for hotel_sn in hotel_of_neighbors:
            point = self.data_hotel.query('SN == @hotel_sn')[['LONGITUDE', 'LATITUDE']].values.tolist()[0]
            for centroi in df_median:
                if calculate_distance(float(point[0]), float(point[1]), float(centroi[0]), float(centroi[1])) < min_distance:
                    lst_res.append(hotel_sn)
                    continue
        return lst_res #self.get_hotels_of_user(lst_user)


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

    @timeit
    def build_distance_metric(self):
        hotels_df = self.data_hotel.loc[:, ['SN', 'LONGITUDE', 'LATITUDE']]
        hotels_df = hotels_df.dropna(how='any')
        hotels_df = hotels_df.drop_duplicates(keep='last')

        hotels_df['lat'] = np.radians(hotels_df['LATITUDE'])
        hotels_df['lon'] = np.radians(hotels_df['LONGITUDE'])

        dist = DistanceMetric.get_metric('haversine')
        dis_matrix = pd.DataFrame(dist.pairwise(hotels_df[['lat','lon']].to_numpy())*6373,  columns=hotels_df.SN.unique(), index=hotels_df.SN.unique())
        dis_matrix.columns = dis_matrix.columns.map(lambda x: "_"+str(x))
        return dis_matrix

    @timeit
    def get_hotels_by_radius(self, hotel_sn, radius=3):
        try:
            hotel_sn_1 = self.dis_matrix.loc[:, [hotel_sn]]
            return list(hotel_sn_1.query('{} <= @radius'.format(hotel_sn)).sort_values(by=hotel_sn, ascending=True).index)
        except Exception as ex:
            logger.error(ex)
            return []
    @timeit        
    def get_n_hotels_nearest(self, hotel_sn, hotel_num = 50):
        try:
            hotel_sn_1 = self.dis_matrix.loc[:, [hotel_sn]]
            return list(hotel_sn_1.sort_values(by=hotel_sn, ascending=True).index)[1:hotel_num]
        except Exception as ex:
            logger.error(ex)
            return []

    def get_results_powerset(self, list_hotel):   
    #     list_cate = list(set(map(lambda x: int(x), list_product.split())))
        recommend = []
        for item_set in reduce(lambda result, x: result + [subset + [x] for subset in result], list_hotel, [[]]): #, [[]]
            for index, row in self.assoc_rule.iterrows():
                if set(item_set) == set(row.A):
                    recommend.extend(row.B)
        return recommend

    # @parallel(merge_func=lambda li: sorted(set(chain(*li))))
    def run_apriori(self, user_lst_hotel):
        self.apriori = Apriori(0.0005, 0.1)  
        inFile = self.apriori.dataFromDataFrame(user_lst_hotel)
        items, rules = self.apriori.runApriori(inFile, self.apriori.minSupport, self.apriori.minConfidence)
        self.apriori.saveResults(items, rules)
        
        tmp = pd.DataFrame(rules)
        tmp1 = pd.DataFrame(list((tmp[0].values)), columns=['A', 'B'])
        tmp1['confidence'] = tmp[1].values
        df_rule = pd.pivot_table(tmp1, index='A', aggfunc=lambda x: ' '.join(str(v) for v in x)).reset_index()
        df_rule.B = df_rule['B'].apply(lambda row: [int(float(x)) for x in re.findall(r"\d+\.?\d*", str(row))])
        # add
        df_rule.B = df_rule['B'].apply(lambda row: list(set(row)))
        df_rule.A = df_rule['A'].apply(lambda row: sorted(list(int(float(x)) for x in row)))
        df_rule.to_csv("association_rules.csv", index=False)
        return df_rule


if __name__=="__main__":

    recommender = HotelRecommender(n_samples=-1)
    # recommender.build_model()
    # res = recommender.predict(172015) #
    list_product = [830, 2812, 1462]
    res = recommender.get_results_powerset(list_product)
    print(res)
    logger.info(res)

