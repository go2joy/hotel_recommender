import os

# REDIS_HOST = os.environ.get('REDIS_HOST', '0.0.0.0') #
# REDIS_PORT = os.environ.get('REDIS_PORT', 6379)
# REDIS_CELERY_DB_INDEX = os.environ.get('REDIS_CELERY_DB_INDEX', 10)
# REDIS_STORE_DB_INDEX = os.environ.get('REDIS_STORE_DB_INDEX', 0)

# RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', '10.166.2.208')
# RABBITMQ_USERNAME = os.environ.get('RABBITMQ_USERNAME', 'newadmin')
# RABBITMQ_PASSWORD = os.environ.get('RABBITMQ_PASSWORD', 'P@ssw0rd1')
# RABBITMQ_PORT = os.environ.get('RABBITMQ_PORT', 5672)
# RABBITMQ_HTTP_PORT = os.environ.get('RABBITMQ_PORT', 15672)

# BROKER_CONN_URI = f"amqp://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_PORT}"
# BACKEND_CONN_URI = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_CELERY_DB_INDEX}"
# REDIS_STORE_CONN_URI = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_STORE_DB_INDEX}"

# BROKER_API = f"http://{RABBITMQ_USERNAME}:{RABBITMQ_PASSWORD}@{RABBITMQ_HOST}:{RABBITMQ_HTTP_PORT}/api/"


BROKER_CONN_URI=os.environ.get('BROKER_CONN_URI', 'amqp://guest:guest@krabbitmq:5672')
BACKEND_CONN_URI = os.environ.get('BACKEND_CONN_URI',"redis://kredis:6379/10")