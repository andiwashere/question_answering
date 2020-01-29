#!/bin/bash
# loosely following http://wiki.ros.org/docker/Tutorials/GUI and https://leimao.github.io/blog/TensorBoard-On-Docker/

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
X_WINDOW_OPTS_PAP="--volume=$XSOCK:$XSOCK --volume=$XAUTH:$XAUTH --env=XAUTHORITY=${XAUTH} --env=DISPLAY=${DISPLAY} --env=QT_X11_NO_MITSHM=1"

# execute in command line: ssh -L 127.0.0.1:16006:0.0.0.0:5001 username@server
docker run --gpus all $X_WINDOW_OPTS_PAP -it -d \
                  -u $(id -u):$(id -g) -p 5000:8888 -p 5001:6006 \
                  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
                  -v ~/workspace/ws1920/seminar:/seminar -w="/seminar" --name=seminar_con \
                  --network=host seminar:tf

export containerId=$(docker ps -l -q)
xhost +local:"docker inspect --format='{{ .Config.Hostname }}' $containerId"
docker exec -it seminar_con bash