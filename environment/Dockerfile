# hash:sha256:095d4b382fe7ae2a98904ed95b1ed7cf3a265a0e8cd0574679df9753f892b28f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip3 install -U --no-cache-dir \
    dask-image==2024.5.3 \
    ipython==8.18.1 \
    ipywidgets==8.1.5 \
    matplotlib==3.9.4 \
    numpy==2.0.2 \
    plotly==6.0.1 \
    zarr==2.16.0
