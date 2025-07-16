# Dockerfile for SageMaker Processing and Training (GPU, PyTorch, Geospatial)

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py39-cu118-ubuntu20.04

# Install system dependencies for geospatial libraries and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy

# Create conda environment and install dependencies
COPY requirements.full.txt /tmp/requirements.full.txt
RUN conda install -y python=3.9 && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.full.txt

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.0

# Set up the entrypoint for SageMaker Processing/Training
ENV PYTHONUNBUFFERED=TRUE
WORKDIR /opt/ml/code

ENTRYPOINT ["python3"]
