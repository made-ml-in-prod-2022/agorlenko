import os.path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities.split_params import SplittingParams


def read_data(path: str, root_path: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(root_path, path))


def split_train_test_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, test_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, test_data
