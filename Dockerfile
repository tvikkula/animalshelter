FROM continuumio/miniconda

RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN mkdir /code
ADD ./requirements.txt /code/
RUN conda create -n data-tommi --file /code/requirements.txt