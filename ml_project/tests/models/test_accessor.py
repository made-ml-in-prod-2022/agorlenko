import os.path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_project.models.accessor import Accessor


def test_read_and_write_data(tmp_path):
    pipeline = Pipeline([
        ('preprocessing', StandardScaler()),
        ('estimator', LogisticRegression())
    ])
    file_path = os.path.join(tmp_path, 'model')
    accessor = Accessor(file_path)
    accessor.save(pipeline)
    assert os.path.exists(file_path)
    loaded_pipeline = accessor.load()
    print(loaded_pipeline)
    assert pipeline.steps[0][0] == loaded_pipeline.steps[0][0]
    assert isinstance(pipeline.steps[0][1], type(loaded_pipeline.steps[0][1]))
    assert pipeline.steps[1][0] == loaded_pipeline.steps[1][0]
    assert isinstance(pipeline.steps[1][1], type(loaded_pipeline.steps[1][1]))
