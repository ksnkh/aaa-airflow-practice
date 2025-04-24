#!/bin/bash

mkdir data

wget https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/Avitotechcomp2025/data_competition_1/clickstream.pq -O data/clickstream.pq >> _
wget https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/Avitotechcomp2025/data_competition_1/test_users.pq -O data/test_users.pq >> _
wget https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/Avitotechcomp2025/data_competition_1/events.pq -O data/events.pq >> _
