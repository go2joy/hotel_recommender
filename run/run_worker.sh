name='extractformsdlvn_myworkers';
tag='latest'
worker_name='106'
n_concurrent=1000
device=0
pool='gevent'
logs=/media/anhlbt/dataset/forms_extractor/logs:/usr/src/app/logs
uploads=/media/anhlbt/dataset/forms_extractor/uploads:/usr/src/app/uploads
broker_conn_uri='amqp://newadmin:P@ssw0rd1@10.166.2.208:5672' #rabbitMQ, change here
backend_conn_uri='redis://10.166.2.140:6379/10' #on API SERVICES

docker stop $name;
docker rm $name;
echo --- Starting container $name with $device;

docker run \
    -it\
    -e WORKERNAME=$worker_name\
    -e BACKEND_CONN_URI=$backend_conn_uri\
    -e BROKER_CONN_URI=$broker_conn_uri\
    -e NUM_CONCURRENT=$n_concurrent\
    -e DEVICE=$device\
    -e POOL_TYPE=$pool\
    -u root\
    -v $logs\
    -v $uploads\
    --name=$name\
    --restart always\
    --gpus all\
    $name:$tag