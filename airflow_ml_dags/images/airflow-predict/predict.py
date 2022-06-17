import os
import pickle

import pandas as pd
import click


@click.command('train')
@click.option('--data-dir')
@click.option('--model-path')
@click.option('--output-dir')
def predict(data_dir: str, model_path: str, output_dir: str):

    X = pd.read_csv(os.path.join(data_dir, 'data.csv'))

    with open(model_path, mode='rb') as f:
        pipeline = pickle.load(f)


    predicts = pipeline.predict(X)

    os.makedirs(output_dir, exist_ok=True)

    predicts_df = pd.DataFrame(predicts)
    predicts_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


if __name__ == '__main__':
    predict()
