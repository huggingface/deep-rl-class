# Docker image for Deep RL course

This image is based on `nvcr.io/nvidia/pytorch:22.04-py3` and contains all necessary dependencies to execute course's notebooks.

## Requirements

- x86-64 architecture;
- Nvidia GPU compatible with CUDA.

## How to use

1. Configure image name in `build.sh` script. Build image with command:

    ```
    bash build.sh
    ```

2. Configure image name (if you changed it) and container name in `start.sh` script. Create and run container with command:

    ```
    bash start.sh
    ```

3. Configure container name (if you changed it) in `into.sh` script. To enter the container execute it with command:

    ```
    bash into.sh
    ```