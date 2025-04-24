#!/bin/bash

IMAGE_NAME=nik_name/docker_tag

docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME