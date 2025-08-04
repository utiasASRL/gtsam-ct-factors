FROM ubuntu:22.04

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
    && git clone https://github.com/utiasASRL/steam.git /opt/steam \
    && cd /opt/steam \
    && mkdir -p build && cd build \
    && cmake .. -DUSE_AMENT=off \
    && cmake --build . -j \
    && cmake --install . 

# Entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT [ "/entrypoint.sh" ]


# User setup args
ARG USERID=0
ARG GROUPID=0
ARG USERNAME=root
ARG HOMEDIR=/root
# Setup user 
RUN if [ ${GROUPID} -ne 0 ]; then addgroup --gid ${GROUPID} ${USERNAME}; fi \
  && if [ ${USERID} -ne 0 ]; then adduser --disabled-password --gecos '' --uid ${USERID} --gid ${GROUPID} ${USERNAME}; fi

# Move to working dir
WORKDIR /${HOMEDIR}/gtsam-ct-factors

# Switch user
USER ${USERNAME}

# Default command
CMD ["/bin/bash"]