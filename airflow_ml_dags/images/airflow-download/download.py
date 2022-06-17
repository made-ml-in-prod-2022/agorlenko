import os
import random

import click
import pandas as pd


SOURCE_DATA_PATH = 'source_data.csv'
TARGET_COLUMN = 'condition'


@click.command('download')
@click.argument('output_dir')
def download(output_dir: str):
    seed = hash(output_dir) % ((1 << 32) - 2)
    random.seed(seed)

    df = pd.read_csv(SOURCE_DATA_PATH)
    max_size = df.shape[0]
    n_rows = random.randint(max_size // 2, max_size)
    df = df.sample(n=n_rows, random_state=seed, ignore_index=True)
    data_df = df[[column for column in df.columns if column != TARGET_COLUMN]]
    target_df = df[[TARGET_COLUMN]]

    os.makedirs(output_dir, exist_ok=True)
    data_df.to_csv(os.path.join(output_dir, 'data.csv'), index=False)
    target_df.to_csv(os.path.join(output_dir, 'target.csv'), index=False)


if __name__ == '__main__':
    download()
