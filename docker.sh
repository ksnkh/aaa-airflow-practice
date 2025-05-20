#!/bin/bash

IMAGE_NAME=kosnah/aaa-airflow

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME