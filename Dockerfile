FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker-v1.12-2025-07-15-19-53-45

# Install system dependencies for geospatial libraries and Python development tools
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
    # Clean up APT cache to reduce image size
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and mamba for efficient package management
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH /opt/conda/bin:$PATH

# Create conda environment and install core geospatial dependencies via mamba
# Using mamba and conda-forge for these often ensures better compilation and dependency resolution
RUN conda install -c conda-forge -y mamba && \
    mamba install -c conda-forge -y \
        gdal \
        rasterio \
        geopandas \
        shapely \
        fiona \
        pyproj && \
    conda clean -afy

# --- PyTorch-Geometric and related dependencies ---
# Install torch-scatter, torch-sparse, torch-geometric, etc. from PyG's custom index
# Based on your base image: PyTorch 2.5.1 and CUDA 12.4
# We use the torch-2.5.0+cu124 wheel index as it's the closest official PyG build for PyTorch 2.5.1 with CUDA 12.4.
# It's important to use --no-cache-dir to keep the image size down.
RUN pip install --no-cache-dir \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    torch_geometric \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

# --- Other Python package installations ---
# Upgrade pip before installing remaining packages
RUN pip install --upgrade pip

# Create a temporary requirements file for the remaining packages
# This ensures a clean pip install for these specific ones
# Note: torch, torchvision are generally already provided by the base DLC image,
# but listing them here for completeness if you ever change base image.
# If they are already installed, pip will usually skip them or re-install if versions differ.
# It's safer to not include 'torch' and 'torchvision' here as they are part of the base image
# and can cause conflicts if pip tries to replace them.
# The base image is public.ecr.aws/deep-learning-containers/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker-v1.12-2025-07-15-19-53-45
# So, torch, torchvision, and possibly numpy, pandas are already present.
# We will create a requirements.txt with the *remaining* packages.

# Create a temporary file for the non-geospatial, non-PyG specific requirements
RUN echo "numpy" > /tmp/requirements_additional.txt && \
    echo "pandas" >> /tmp/requirements_additional.txt && \
    echo "Pillow==9.5.0" >> /tmp/requirements_additional.txt && \
    echo "sagemaker" >> /tmp/requirements_additional.txt && \
    echo "scikit-learn" >> /tmp/requirements_additional.txt && \
    echo "timm" >> /tmp/requirements_additional.txt && \
    echo "tqdm" >> /tmp/requirements_additional.txt && \
    echo "albumentations" >> /tmp/requirements_additional.txt && \
    # Install these additional packages
    pip install --no-cache-dir -r /tmp/requirements_additional.txt && \
    # Clean up the temporary file
    rm /tmp/requirements_additional.txt

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_VERSION=3.6.0

# Set up the entrypoint for SageMaker Processing/Training
ENV PYTHONUNBUFFERED=TRUE
WORKDIR /opt/ml/code