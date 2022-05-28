from typing import Union, List, Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from .validation import DiseasePredictResponse


def make_predict(data: Dict[str, List[Union[int, float]]], model: Pipeline,) -> List[DiseasePredictResponse]:
    df = pd.DataFrame(data)
    predict = model.predict(df)
    return [DiseasePredictResponse(condition=x) for x in predict]
