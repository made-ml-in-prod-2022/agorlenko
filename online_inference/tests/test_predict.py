import os
from unittest.mock import patch

from starlette.testclient import TestClient

from ml_project.models.accessor import Accessor
from online_inference.app import app


def _write_model_data(model_url, output):
    test_data_accessor = Accessor('data/lr_model.pkl')
    accessor = Accessor(output)
    accessor.save(test_data_accessor.load())


@patch('gdown.download')
def test_predict(mock_gdown_download, monkeypatch):
    envs = {
        'MODEL_URL': 'test_model_url'
    }
    monkeypatch.setattr(os, 'environ', envs)
    mock_gdown_download.side_effect = _write_model_data
    request_data = {
        'age': [61],
        'sex': [1],
        'cp': [0],
        'trestbps': [134],
        'chol': [234],
        'fbs': [0],
        'restecg': [0],
        'thalach': [145],
        'exang': [0],
        'oldpeak': [2.6],
        'slope': [1],
        'ca': [2],
        'thal': [0]
    }
    with TestClient(app) as client:
        response = client.post('/predict', json={'data': request_data})
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]['condition'] in (0, 1)
