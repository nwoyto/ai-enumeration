# Dockerfile for SageMaker Processing with geospatial stack

FROM continuumio/miniconda3:latest

# Install system dependencies for geospatial libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.0

# Install Python dependencies
COPY preprocessing/requirements.txt /opt/ml/processing/code/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /opt/ml/processing/code/requirements.txt

# Set up the entrypoint for SageMaker Processing
ENV PYTHONUNBUFFERED=TRUE

# SageMaker will copy your script to /opt/ml/processing/input/code/
WORKDIR /opt/ml/processing/input/code

ENTRYPOINT ["python3"]
