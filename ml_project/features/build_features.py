import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml_project.entities.feature_params import FeatureParams


def get_numeric_features_pipline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ('scaler', StandardScaler())
    ])


def get_categorical_features_pipline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])


def build_transformer(feature_params: FeatureParams) -> ColumnTransformer:
    return ColumnTransformer([
        ('numeric_features', get_numeric_features_pipline(), feature_params.numeric_features),
        ('categorical_features', get_categorical_features_pipline(), feature_params.categorical_features)
    ])


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    return df[params.target_col]
