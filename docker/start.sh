#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

export ARCH=`uname -m`

cd "$(dirname "$0")"
root_dir=$PWD 
cd $root_dir

echo "Starting ${orange}deep_rl_course${reset_color} container"

if [ "$ARCH" == "x86_64" ] 
then
    ARGS="--ipc host --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
else
    echo "Arch ${ARCH} not supported"
    exit
fi

docker run -it -d --rm \
        $ARGS \
        --privileged \
        --name deep_rl_course \
        --net host \
        -v $root_dir/../:/home/docker_rl/deep-rl-class:rw \
        deeprlcourse

docker exec --user root \
    deep_rl_course bash -c "/etc/init.d/ssh start"