from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from ml_project.entities.train_params import TrainingParams

FACTORIES = {
    'LogisticRegression': LogisticRegression,
    'RandomForestClassifier': RandomForestClassifier,
}


def train_model(features: pd.DataFrame, target: pd.Series, transformer, train_params: TrainingParams) -> Pipeline:
    if train_params.model_type not in FACTORIES:
        raise ValueError(f'unknown model type: {train_params.model_type}')
    model = FACTORIES[train_params.model_type](**train_params.options)
    pipeline = Pipeline([
        ('preprocessing', transformer),
        ('classifier', model)
    ])
    pipeline.fit(features, target)
    return pipeline


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "f1_score": f1_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predicts),
    }
