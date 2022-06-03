import pickle

from sklearn.pipeline import Pipeline


class Accessor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save(self, model: Pipeline) -> None:
        with open(self.file_path, mode='wb') as f:
            pickle.dump(model, f)

    def load(self) -> Pipeline:
        with open(self.file_path, mode='rb') as f:
            return pickle.load(f)
