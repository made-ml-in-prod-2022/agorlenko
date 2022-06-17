import json
import os
import pickle

import pandas as pd
import click
from sklearn.metrics import f1_score, roc_auc_score


@click.command('evaluate')
@click.option('--input-dir')
@click.option('--models-dir')
@click.option('--output-dir')
def evaluate(input_dir: str, models_dir: str, output_dir: str):
    with open(os.path.join(models_dir, 'lr_model.pkl'), mode='rb') as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(os.path.join(input_dir, 'test_data.csv'))

    X = df[[col for col in df.columns if col != 'target']]
    y = df['target']

    predicts = pipeline.predict(X)

    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "f1_score": f1_score(y, predicts),
        "roc_auc_score": roc_auc_score(y, predicts),
    }

    with open(os.path.join(output_dir, 'lr_metrics.json'), mode='w') as metric_file:
        json.dump(metrics, metric_file)


if __name__ == '__main__':
    evaluate()
