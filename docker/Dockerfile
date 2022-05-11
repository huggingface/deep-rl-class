FROM nvcr.io/nvidia/pytorch:22.04-py3

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    sudo \
    tar \
    unzip \
    openssh-server \
    python3-pip \
    build-essential \
    python-opengl \
    ffmpeg \
    xvfb \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglfw3 \
    swig \
    && rm -rf /var/lib/apt/lists/*

# install git-lfs
RUN wget -q https://github.com/git-lfs/git-lfs/releases/download/v3.1.4/git-lfs-linux-amd64-v3.1.4.tar.gz && \
    mkdir git-lfs && \
    tar -xzf git-lfs-linux-amd64-v3.1.4.tar.gz -C git-lfs && \
    bash git-lfs/install.sh && \
    rm -rf git-lfs git-lfs-linux-amd64-v3.1.4.tar.gz

# add user and his password
ARG USER=docker_rl
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}

RUN mkdir deep-rl-class && chown -R ${UID}:${GID} /home/${USER}

USER ${UID}:${GID}

# Install MuJoCo v1.5.0 library as it is required by some of gym deps
RUN wget -q https://www.roboti.us/download/mjpro150_linux.zip && \
    unzip -q mjpro150_linux.zip -d .mujoco/ && \
    rm mjpro150_linux.zip && \
    wget -q https://www.roboti.us/file/mjkey.txt -P .mujoco/ && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/docker_rl/.mujoco/mjpro150/bin' >> .bashrc && \
    echo 'export PATH=$PATH:$HOME/.local/bin' >> .bashrc

# we need to manually export LD_LIBRARY_PATH and PATH because of non-interactive mode while building image
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/docker_rl/.mujoco/mjpro150/bin && \
    export PATH=$PATH:$HOME/.local/bin && \
    pip install pyglet==1.5.1 lockfile glfw imageio && \
    pip install stable-baselines3[extra]>=1.5.0 && \
    pip install \
        pyvirtualdisplay \
        gym[all] \
        huggingface_sb3 \
        ale-py==0.7.4 \
        ipywidgets

# See https://github.com/DLR-RM/stable-baselines3/issues/298 
RUN pip uninstall -y opencv-python && pip install opencv-python-headless