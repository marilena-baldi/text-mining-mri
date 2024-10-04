FROM python:3.8 AS base

ARG PROJECT_PATH=/sdn-ml
ENV PYTHONPATH=$PROJECT_PATH

WORKDIR $PROJECT_PATH

COPY ./src/requirements.txt ./src/requirements.txt

RUN pip install -r ./src/requirements.txt