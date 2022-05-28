from typing import List, Union, Dict

from pydantic import BaseModel


class HeartDiseasesModel(BaseModel):
    data: Dict[str, List[Union[int, float]]]


class DiseasePredictResponse(BaseModel):
    condition: int
