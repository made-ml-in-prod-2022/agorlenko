from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema

from ml_project.entities.feature_params import FeatureParams
from ml_project.entities.split_params import SplittingParams
from ml_project.entities.train_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(file: str) -> TrainingPipelineParams:
    with open(file, mode='r') as f:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(f))
