from ml_project.data.make_dataset import read_data
from ml_project.tests import ROOT_DIR


def test_read_data():
    df = read_data('data/raw/data.csv', ROOT_DIR)
    assert df.shape[0] > 0 and df.shape[1] > 0
