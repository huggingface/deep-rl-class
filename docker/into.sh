#!/bin/bash

docker exec --user "docker_rl" -it deep_rl_course \
    /bin/bash -c "cd /home/docker_rl; echo deep_rl_course container; echo ; /bin/bash"