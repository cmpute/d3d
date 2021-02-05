FROM continuumio/miniconda

RUN conda install -y conda-build
RUN apt update
RUN apt install -y build-essential

COPY . /d3d
WORKDIR /d3d/conda/cpu
RUN conda build -c conda-forge --output-folder . .
