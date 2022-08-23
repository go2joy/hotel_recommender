
from celery import Celery, states, Task
from .config import BROKER_CONN_URI , BACKEND_CONN_URI

celery_app = Celery('data_hprs', backend=BACKEND_CONN_URI, broker=BROKER_CONN_URI, include=['worker.tasks']) #
celery_app.conf.task_default_queue = 'data_hprs'
celery_app.conf.task_routes = {
                               "worker.tasks.weight_matrix": "data_hprs"
                               }
celery_app.conf.update(task_track_started=True, result_expires=3*24*3600)