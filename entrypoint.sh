#!/bin/bash


echo Starting server.
# python ./api/server.py
gunicorn -w $NUM_WORKERS -k uvicorn.workers.UvicornWorker api.server:app --bind 0.0.0.0:5005 --timeout 300

# chmod -R 777 /usr/src/app/logs/
# echo Starting workers.
# celery -A worker.tasks.celery_app worker --loglevel=INFO --pool=prefork --concurrency=24 --hostname=anhlbt@%h --queues CDForm -E 