FROM continuumio/miniconda

RUN conda install -y conda-build
RUN apt update
RUN apt install -y build-essential

COPY . /d3d
RUN conda build /d3d/conda -c pytorch -c conda-forge
