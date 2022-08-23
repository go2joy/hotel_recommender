from .config import BROKER_CONN_URI , BACKEND_CONN_URI
from os.path import dirname, realpath, join, basename
import sys, os, stat
import uuid
C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)

sys.path.insert(0, P_DIR)

from io import BytesIO
from sys import exit
from time import sleep
from typing import (
    Any,
    Dict
)
from urllib import request
import traceback
from PIL import Image
from celery import Celery, states, Task
from celery.exceptions import Ignore, MaxRetriesExceededError
from .app_worker import celery_app
# import extractor
# from extractor import Extractor, detect_table, is_box_null
from go2joy import HotelRecommender
from utils.util import exception_logger

logger = exception_logger("worker", join(P_DIR, "./logs/{}_workers.log".format(os.environ.get('WORKERNAME', 'default'))))

UPLOAD_FOLDER = '/usr/src/app/uploads'


class PredictTask(Task):
    def __init__(self):
        super().__init__()
        self.model = None

    def __call__(self, *args, **kwargs):
        if not self.model:
            logger.info('Loading Model...')
            self.model = HotelRecommender('./config.yml')
            logger.info('Model loaded')
        return self.run(*args, **kwargs)



@celery_app.task(ignore_result=False, name='worker.tasks.weight_matrix', bind=True, acks_late=True, base=PredictTask)
def get_weight_matrix(self, recommender):
    try:
        res=[]
        # print(file_name)
        # format_img = format_checking(img)
        # image = Image.open(BytesIO(base64.b64decode(img)))

        # name = str(uuid.uuid4()).split('-')[0]
        # file_name = f'{UPLOAD_FOLDER}/{name}.{format_img}'
        # image.save(file_name)  

        # detected_img, source= self.model.detect(file_name) #local
        
        # detected_img, source= self.model.detect(join(UPLOAD_FOLDER, basename(file_name)))
        
        # result, _ =  self.model.recognize(detected_img)

        # _, table = detect_table(detected_img['ICR_TABLE'])
        # res_table,_ = is_box_null(table)
        
        # loaded_img = self.model.finger_model.img2tensor(source)
        # res, _ = self.model.finger_model.predict(loaded_img)
        # result['FINGER_COUNT']= len(res)
        # result['ICR_TABLE']=res_table
        # res.append(result)
        weight_matrix = recommender.get_weight_matrix(recommender.data_user_booking, 1141, alpha = .5, threshold = .96, visual = True)
        return weight_matrix

    except Exception as ex:
        logger.exception(ex)
        try:
            self.retry(countdown=2)
        except MaxRetriesExceededError as ex:
            return res

