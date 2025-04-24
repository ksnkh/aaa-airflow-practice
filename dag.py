from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import pandas as pd

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}



DEFAULT_CONFIG = {
    'command': "ls",
    'MLFLOW_TRACKING_URI': 'http://51.250.35.156:5000/',
    'image': 'mastryukov1990/aaa_2025',
}


class Key:
    TASK_INSTANCE_KEY = 'ti'  # xcom object
    # https://airflow.apache.org/docs/apache-airflow/stable/concepts/operators.html#reserved-params-keyword
    PARAMS = 'params'
    DAG_RUN = 'dag_run'


def get_custom_dags_df() -> pd.DataFrame:
    SHEET_ID = '1cAMLY10DY8gz2UWPk1EkRJDDMtDW2cafHONCPzoNYi4'
    SHEET_NAME = 'aaa_2025'
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    return pd.read_csv(url)[['image', 'name', 'command']]


def get_config_value(task_id: str, key: str):
    return f'{{{{task_instance.xcom_pull(task_ids="{task_id}", key="{key}")}}}}'


class ConfigPusher:
    DEFAULT_CONFIG = DEFAULT_CONFIG

    def prepare_default(self, dag_run) -> dict:
        return {}

    def finalize_config(self, config) -> dict:
        return {}

    def __call__(self, **kwargs):
        task_instance = kwargs[Key.TASK_INSTANCE_KEY]
        dag_config = kwargs.get(Key.PARAMS, {})

        default_config = self.DEFAULT_CONFIG.copy()
        config = {
            key: dag_config[key] if key in dag_config else default_value
            for key, default_value in default_config.items()
        }
        for key, value in sorted(config.items()):
            task_instance.xcom_push(key=key, value=value)


def get_debug_dag():
    dag = DAG(
        'docker_with_mount_dag_v2',
        default_args=default_args,
        description='Run a Docker container with volume mount',
        schedule_interval=timedelta(days=1),  # Adjust as needed
        start_date=datetime(2023, 1, 1),
        catchup=False,
        params=DEFAULT_CONFIG,
    )

    config_operator = PythonOperator(
        task_id='config',
        python_callable=ConfigPusher(),
        do_xcom_push=True,
        dag=dag
    )

    docker_operator = DockerOperator(
        task_id='train_model',
        image=get_config_value('config', 'image'),
        command=get_config_value('config', 'command'),
        api_version='auto',
        auto_remove='success',
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(
                source="/home/tolkkk/repos/airflow/data",
                target="/app/data",
                type="bind",
            ),
        ],
        network_mode='bridge',
        dag=dag,
        environment={
            'MLFLOW_TRACKING_URI': get_config_value('config', 'MLFLOW_TRACKING_URI'),
        }
    )
    config_operator >> docker_operator

    return dag


def get_dag(image: str, dag_id: str, command: str):
    print(image)
    dag = DAG(
        f'student-{dag_id}',
        default_args=default_args,
        description='Run a Docker container with volume mount',
        schedule_interval=timedelta(days=1),  # Adjust as needed
        start_date=datetime(2023, 1, 1),
        catchup=False,
        params={
            'command': command,
            'MLFLOW_TRACKING_URI': 'http://51.250.35.156:5000/',
            'image': image,
        },
    )

    config_operator = PythonOperator(
        task_id='config',
        python_callable=ConfigPusher(),
        do_xcom_push=True,
        dag=dag
    )

    docker_operator = DockerOperator(
        task_id='train_model',
        image=get_config_value('config', 'image'),
        command=get_config_value('config', 'command'),
        api_version='auto',
        auto_remove='success',
        docker_url="unix://var/run/docker.sock",
        mounts=[
            Mount(
                source="/home/tolkkk/repos/airflow/data",
                target="/app/data",
                type="bind",
            ),
        ],
        network_mode='bridge',
        dag=dag,
        environment={
            'MLFLOW_TRACKING_URI': get_config_value('config', 'MLFLOW_TRACKING_URI'),
        }
    )
    config_operator >> docker_operator

    return dag


df = get_custom_dags_df()

dags = []

for row in df.iterrows():
    row = row[1]
    globals()[row['name']] = get_dag(image=row['image'], dag_id=row['name'], command=row['command'])


if __name__ == "__main__":
    for d in dags:
        d.cli()
