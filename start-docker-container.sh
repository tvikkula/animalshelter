docker-machine start default
eval $(docker-machine env)
docker build -t "purplerain" .