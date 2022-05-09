import yaml
from dataclasses import dataclass


@dataclass()
class PredictionParams:
    output_model_path: str


def read_prediction_params(file: str) -> PredictionParams:
    with open(file, mode='r') as f:
        return PredictionParams(yaml.safe_load(f)['output_model_path'])
