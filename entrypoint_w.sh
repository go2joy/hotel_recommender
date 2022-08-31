#!/bin/bash


echo Starting workers.
# celery -A worker.tasks.celery_app worker --loglevel=INFO --pool=prefork --concurrency=$NUM_CONCURRENT --hostname=anhlbt@%h --queues CDForm -E --config=worker.celery_config --autoscale=$NUM_CONCURRENT,1
celery -A worker.tasks.celery_app worker --loglevel=INFO --pool=$POOL_TYPE --concurrency=$NUM_CONCURRENT --hostname=anhlbt@%h --queues data_hprs -E --autoscale=10,1