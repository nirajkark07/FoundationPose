docker rm -f foundationpose
DIR=$(pwd)/../
xhost + && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --privileged --name foundationpose --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp -v /dev/:/dev/--ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE foundationpose:latest bash -c "cd $DIR && bash"
