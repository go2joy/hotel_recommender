import os
os.environ["MODIN_ENGINE"] = "dask" 
import json 
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime
import sqlalchemy
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

# Importing dask dataframe
# import dask
# import dask.dataframe as pd
# import ray
# ray.init(num_cpus=6, ignore_reinit_error=True)
# import modin.pandas as pd  #pip install modin[ray]

import pandas as pd
import urllib.parse
urllib.parse.quote_plus("Mysql@12345!")

# from os import getenv
# # import pymssql
# import mysql.connector
# import pandas as pd

# # query = "use eApplication select pro.ProposalID, ProposalNo, AgentID, ConfirmDate, URL \
# # from eApp_tbProposalDocument doc \
# # inner join eApp_tbProposal pro on doc.ProposalID = pro.ProposalID \
# # where URL like '%10.166.1.114%2021/06%DigitalConfirmation%'"

# query ="select * \
# from HOTEL \
# "

# def connectDB(host="118.69.235.218",  user="admin",  password="Mysql@12345!",  database="hotel_dev_180"):
#     conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
# #     cursor = conn.cursor()  #as_dict = True
#     return conn
# # select tracking data

# def get_data(query):
#     try:
#         conn = connectDB()
#         cursor = conn.cursor()
#         cursor.execute(query)
#         data = cursor.fetchall()
#         data = pd.DataFrame(data)
#         cursor.close()
        
#     except Exception as ex:
#         print(ex)
#     return data

class JsonEncodedDict(sqlalchemy.TypeDecorator):
    """Enables JSON storage by encoding and decoding on the fly."""
    impl = sqlalchemy.Text

    def process_bind_param(self, value, dialect):
        if value is None:
            return '{}'
        else:
            return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return {}
        else:
            return json.loads(value)
        

class MyDatabase:
    def __init__(self, dbtype='MYSQL_PRO', dbname='go2joy', isolation_level='READ COMMITTED'): #staging
        '''
        isolation_level =  READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, and SERIALIZABLE
        '''
        dbtype = dbtype.upper()
        # DB_ENGINE = {"SQLITE":"sqlite:////media/anhlbt/DATA/nlp_2020/flask_fullstack/MadBlog/back-end/{DB}"}
        DB_ENGINE = {"POSTGRESQL_DEV":"postgresql+psycopg2://airflow:airflow@localhost/{DB}",
                    "MYSQL_DEV":"mysql+pymysql://admin:Mysql%4012345%21@118.69.235.218/{DB}",
                    "MYSQL_DEV_GO2JOY":"mysql+pymysql://root:lbtanh@localhost/{DB}", #docker
                    "MYSQL_STAGING":"mysql+pymysql://anh.tuan:Go2joy2wsx#EDC@go2joy-staging.c56ujmeumc6j.ap-southeast-1.rds.amazonaws.com/{DB}",
                    "MYSQL_PRO":"mysql+pymysql://anh.tuan:Go2joy2wsx#EDC@go2joy-replica.c56ujmeumc6j.ap-southeast-1.rds.amazonaws.com/{DB}",
                    "MYSQL_PMS_PRO":"mysql+pymysql://tuan.anh:tuananh%40gopms%21@pms-production.cuiciqs3fjdo.ap-southeast-1.rds.amazonaws.com/{DB}",
                    "MYSQL_PRO_2":"mysql+pymysql://anh.tuan:Go2joy2wsx#EDC@go2joy.c56ujmeumc6j.ap-southeast-1.rds.amazonaws.com/{DB}",
                    "MARIADB_DEV":"mariadb+mariadbconnector://root:lbtanh@127.0.0.1:3307/{DB}", #docker
                    } 
        #pms-production.cuiciqs3fjdo.ap-southeast-1.rds.amazonaws.com   tuananh%40gopms%21    Go2joy2wsx%23EDC

        if dbtype in DB_ENGINE.keys():
            self.engine_url = DB_ENGINE[dbtype].format(DB=dbname)
            self.db_engine = sqlalchemy.create_engine(self.engine_url, pool_recycle=3600)
            print(self.db_engine) # we have 2 way to connect to db, self.db_engine.connect() or use Session
            Session = sessionmaker(bind=self.db_engine)
            # When we make a new Session, either using the constructor directly or when we call upon the callable produced by a sessionmaker, we can pass the bind argument directly, overriding the pre-existing bind.
            self.session = Session(bind=self.db_engine.execution_options(isolation_level=isolation_level))
            
        else:
            print("DBType is not found in DB_ENGINE")

    # Insert, Update, Delete
    def execute_query(self, query=''):
        if query == '' : return
        # print(query)
        with self.db_engine.connect() as connection:
            try:
                data = connection.execute(query)
                return pd.DataFrame(data)
            except Exception as e:
                print(e)
                return None

    def ses_execute_query(self, query=''):
        if query == '' : return

        try:
            data = self.session.execute(query)
            self.session.commit()
            return pd.DataFrame(data)
        except Exception as e:
            print(e)
            return None 
                
