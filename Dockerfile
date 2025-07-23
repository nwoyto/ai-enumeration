# Dockerfile for SageMaker Processing and Training (GPU, PyTorch, Geospatial)

FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker-v1.12-2025-07-15-19-53-45

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

# Create conda environment and install dependencies
COPY requirements.full.txt /tmp/requirements.full.txt
# Diagnostics: show conda info and config
RUN conda info && conda config --show

# Install mamba for faster conda installs and use conda-forge as the highest-priority channel
RUN conda install -c conda-forge -y mamba && \
    mamba install -c conda-forge -y \
        gdal \
        rasterio \
        geopandas \
        shapely \
        fiona \
        pyproj && \
    conda clean -afy

# Upgrade pip and install only non-geospatial requirements
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.full.txt

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.0

# Set up the entrypoint for SageMaker Processing/Training
ENV PYTHONUNBUFFERED=TRUE
WORKDIR /opt/ml/code


