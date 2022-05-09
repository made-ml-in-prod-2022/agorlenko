import os

from ml_project.tests import ROOT_DIR
from ml_project.train.run import train


def test_train_model():
    path_to_model, metrics = train('configs/lr_config.yaml', ROOT_DIR)
    assert metrics['f1_score'] > 0.5
    assert metrics['roc_auc_score'] > 0.5
    assert os.path.exists(path_to_model)
