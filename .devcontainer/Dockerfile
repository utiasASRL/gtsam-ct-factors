FROM ubuntu:22.04

# Preset the timezone information to prevent hang when installing tzdata
ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install essential tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    vim \
    gdb \
    lldb \
    valgrind \
    iputils-ping \
    libeigen3-dev \
    ffmpeg\
    libboost-all-dev \
    libsm6 \
    libxext6 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Boost env vars
ENV BOOST_ROOT=/usr/local
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CPLUS_INCLUDE_PATH=/usr/local/include:$CPLUS_INCLUDE_PATH

# Set up shared memory
RUN mkdir -p /dev/shm && chmod 1777 /dev/shm

# Set up python environment
RUN apt-get update && apt-get -y install python3-pip
RUN pip install uv

# Install PyQt5 Dependencies
RUN apt-get update && apt-get install -y \
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev

# tmux
RUN apt-get update && apt-get install -y tmux

# Clone and build yaml-cpp
RUN git clone https://github.com/jbeder/yaml-cpp.git /opt/yaml-cpp \
    && cd /opt/yaml-cpp \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF -DYAML_CPP_BUILD_TESTS=OFF \
    && make -j$(nproc) \
    && make install

# Clone and build lgmath (for steam), install to /usr/local
RUN mkdir -p /opt/lgmath \
    && git clone https://github.com/utiasASRL/lgmath.git /opt/lgmath \
    && cd /opt/lgmath \
    && mkdir -p build && cd build \
    && cmake .. \
    && cmake --build . \
    && cmake --install . 

# Clone and build STEAM
RUN mkdir -p /opt/steam \
    && git clone --branch lostinthewoods-ex https://github.com/utiasASRL/steam.git /opt/steam \
    && cd /opt/steam \
    && mkdir -p build && cd build \
    && cmake .. -DUSE_AMENT=off \
    && cmake --build . -j \
    && cmake --install . 

# VNC SETUP
RUN apt-get update && apt-get install -y \
    x11vnc xvfb xfce4 xfce4-terminal

# CCMAKE
RUN apt-get update && apt-get install -y cmake-curses-gui

# kcachegrind
RUN apt-get update && apt-get install -y kcachegrind

# Set up user inside container to avoid issues with permissions.
ARG USERNAME=cho
ARG USER_UID=1094
ARG USER_GID=1094

RUN groupadd --gid $USER_GID $USERNAME \
&& useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 

# Change any ownerships here...

# Move to working dir
WORKDIR /${HOMEDIR}/gtsam-ct-factors

# Switch user
USER ${USERNAME}

# Default command
CMD ["/bin/bash"]