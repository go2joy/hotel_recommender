n_workers=2
name='go2joy_hprs_api';
tag='latest'
start_port=5006
#share folder between workers and api service, change here
logs=~/docker/hotel_price_recommender/logs:/usr/src/app/logs
uploads=~/docker/hotel_price_recommender/uploads:/usr/src/app/uploads

sudo docker stop $name;
sudo docker rm $name;
echo "Starting $n_workers workers on CPU";
echo --- Starting container $name  with CPU  at port $start_port;

sudo docker run -p $start_port:5005\
    -d\
    -e PORT=5005\
    -e NUM_WORKERS=$n_workers\
    -v $logs\
    -v $uploads\
    --health-cmd='curl -f http://0.0.0.0:5005/info || exit 1'\
    --health-interval=2m\
    --health-timeout=10s\
    --health-retries=3\
    --name=$name\
    --restart always\
    $name:$tag