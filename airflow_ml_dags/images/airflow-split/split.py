import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command('split')
@click.option('--input-dir')
@click.option('--output-dir')
@click.option('--val_size', type=float)
def split(input_dir: str, output_dir: str, val_size: float):

    random_state = hash(output_dir) % ((1 << 32) - 2)

    data = pd.read_csv(os.path.join(input_dir, 'train_data.csv'))

    train_data, test_data = train_test_split(
        data, test_size=val_size, random_state=random_state
    )

    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
    test_data.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)


if __name__ == '__main__':
    split()
