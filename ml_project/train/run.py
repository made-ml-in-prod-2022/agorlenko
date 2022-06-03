import json
import logging.config
import os
from typing import Tuple

from argparse import ArgumentParser, Namespace

from ml_project import ROOT_DIR
from ml_project.data.make_dataset import read_data, split_train_test_data
from ml_project.entities.train_pipeline_params import read_training_pipeline_params
from ml_project.features.build_features import build_transformer, extract_target
from ml_project.log_utils import init_loggers
from ml_project.models.accessor import Accessor
from ml_project.models.trainer import train_model, predict_model, evaluate_model


logger = logging.getLogger('app')


def train(config_file: str, root_path: str = ROOT_DIR) -> Tuple[str, dict]:
    logger.info('start training model')
    training_pipeline_params = read_training_pipeline_params(config_file)
    logger.info('train params: %s', training_pipeline_params)

    data = read_data(training_pipeline_params.input_data_path, root_path)
    logger.info('read data for train with shape: %s', data.shape)

    train_df, test_df = split_train_test_data(data, training_pipeline_params.splitting_params)
    logger.info('train data with shape: %s, test data with shape: %s', train_df.shape, test_df.shape)

    transformer = build_transformer(training_pipeline_params.feature_params)
    train_target = extract_target(train_df, training_pipeline_params.feature_params)

    logger.info('start training %s model', training_pipeline_params.train_params.model_type)
    model = train_model(
        train_df, train_target, transformer, training_pipeline_params.train_params
    )
    logger.info('finish training %s model', training_pipeline_params.train_params.model_type)

    test_target = extract_target(test_df, training_pipeline_params.feature_params)
    logger.info('test features shape: %s', test_df.shape)

    logger.info('start predict %s model', training_pipeline_params.train_params.model_type)
    predicts = predict_model(model, test_df)
    logger.info('finish predict %s model', training_pipeline_params.train_params.model_type)

    metrics = evaluate_model(predicts, test_target)
    logger.info('metrics: %s', metrics)
    with open(os.path.join(root_path, training_pipeline_params.metric_path), mode='w') as metric_file:
        json.dump(metrics, metric_file)
    logger.info('metrics saved to %s', training_pipeline_params.metric_path)

    path_to_model = os.path.join(root_path, training_pipeline_params.output_model_path)
    model_accessor = Accessor(path_to_model)
    model_accessor.save(model)
    logger.info('model saved to %s', path_to_model)

    return path_to_model, metrics


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog='train',
        description='Heart Disease Cleveland train model',
    )
    parser.add_argument('-c', '--config', required=True, type=str)
    return parser.parse_args()


def main(args: Namespace):
    train(args.config)


if __name__ == '__main__':
    init_loggers()
    args = parse_args()
    main(args)
