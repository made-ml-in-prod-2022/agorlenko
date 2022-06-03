import logging
import os
from argparse import Namespace, ArgumentParser

from ml_project import ROOT_DIR
from ml_project.data.make_dataset import read_data
from ml_project.entities.predict_params import read_prediction_params
from ml_project.log_utils import init_loggers
from ml_project.models.accessor import Accessor
from ml_project.models.trainer import predict_model

logger = logging.getLogger('app')


def predict(config_file: str, test_file_path: str, output_file_path: str, root_path: str = ROOT_DIR):
    logger.info('start prediction')
    prediction_params = read_prediction_params(config_file)
    logger.info('prediction params: %s', prediction_params)

    path_to_model = os.path.join(root_path, prediction_params.output_model_path)
    model_accessor = Accessor(path_to_model)
    model = model_accessor.load()
    logger.info('loaded model from %s', path_to_model)

    data = read_data(test_file_path, root_path)
    logger.info('read data for predict with shape: %s', data.shape)

    logger.info('start prediction')
    predicts = predict_model(model, data)
    logger.info('finish prediction')

    if output_file_path is None:
        print(predicts)
    else:
        with open(output_file_path, mode='w') as f:
            f.write(str(predicts))
        print(f'write file to {output_file_path}')


def parse_args() -> Namespace:
    parser = ArgumentParser(
        prog='train',
        description='Heart Disease Cleveland train model',
    )
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-t', '--test_df', required=True, type=str)
    parser.add_argument('-o', '--output_file', required=False, type=str, default=None)
    return parser.parse_args()


def main(args: Namespace):
    predict(args.config, args.test_df, args.output_file)


if __name__ == '__main__':
    init_loggers()
    args = parse_args()
    main(args)
