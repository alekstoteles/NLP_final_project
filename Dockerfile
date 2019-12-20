# FROM tensorflow/tensorflow

FROM ubuntu:latest

MAINTAINER kevinstone@berkeley.edu

ENV ARCH=aarch64

WORKDIR /code
ENV FLASK_APP run_keras_server.py
ENV FLASK_RUN_HOST 0.0.0.0

# Set the locale
RUN apt-get update && apt-get install -y apt-utils

RUN apt-get install -y software-properties-common

RUN apt-get update && apt-get install -y locales && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    grep \
    python3.6 \
    python3-pip

RUN python3 -m pip install --upgrade pip

RUN pip install --upgrade setuptools
RUN pip install keras gunicorn jinja2 ninja numpy pandas requests urllib3 wget wtforms flask numpy
# RUN pip install https://files.pythonhosted.org/packages/3f/98/5a99af92fb911d7a88a0005ad55005f35b4c1ba8d75fba02df726cd936e6/tensorflow-1.15.0-cp36-cp36m-manylinux2010_x86_64.whl
RUN pip install tensorflow

COPY . .

CMD ["flask", "run"]
# CMD ["python3", "run_keras_server.py"]