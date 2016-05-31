FROM continuumio/miniconda
RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN mkdir /code
ADD ./requirements.txt /code/
RUN apt-get update
RUN apt-get install -y g++ make
RUN pip install --pre xgboost
WORKDIR /code
