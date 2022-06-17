from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount
from airflow.models import Variable


default_args = {
    'owner': 'airflow',
    'email': ['agorlenko1@gmail.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

model_path = Variable.get('model_path', '/data/models/{{ ds }}/lr_model.pkl')
host_data_dir = Variable.get('data_dir')

with DAG(
        'predict_dag',
        default_args=default_args,
        schedule_interval='@daily',
        start_date=days_ago(5),
) as dag:
    predict = DockerOperator(
        image='airflow-predict',
        command='--data-dir /data/raw/{{ ds }} --model-path ' + model_path + ' --output-dir /data/predictions/{{ ds }}',
        network_mode='bridge',
        task_id='docker-airflow-predict',
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[Mount(source=host_data_dir, target='/data', type='bind')]
    )

    predict
