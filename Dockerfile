FROM continuumio/miniconda
RUN export LC_ALL=en_US.UTF-8
RUN export LANG=en_US.UTF-8
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN mkdir /code
ADD ./requirements.txt /code/
RUN conda create -n purplerain --file /code/requirements.txt
RUN source activate purplerain