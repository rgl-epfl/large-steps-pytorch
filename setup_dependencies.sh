#!/bin/bash

# Make sure all dependencies are installed
sudo apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl \
    dvipng \
    texlive-latex-extra \
    texlive-fonts-recommended \
    cm-super \
    libeigen3-dev

# Make sure submodules are checked out
git submodule update --init --recursive

# Botsch-Kobbelt remesher
cd ext/botsch-kobbelt-remesher-libigl
mkdir -p build
cd build
cmake ..
make -j
cd ../../..

# nvdiffrast
cd ext/nvdiffrast
pip install .
