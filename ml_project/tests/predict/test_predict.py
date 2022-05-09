import os.path

from ml_project.predict.run import predict
from ml_project.tests import ROOT_DIR
from ml_project.train.run import train


def test_predict_model(tmp_path):
    output_file = os.path.join(tmp_path, 'out')
    config_path = 'configs/lr_config.yaml'
    train(config_path, ROOT_DIR)
    predict(config_path, 'data/raw/test_heart_cleveland_upload.csv', output_file, ROOT_DIR)
    with open(output_file) as f:
        assert len(f.read()) > 0
