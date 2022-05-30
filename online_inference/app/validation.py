from typing import List

from pydantic import BaseModel, validator


def _check_value_in_list(field_name, v, values):
    if v not in values:
        raise ValueError(f'{field_name} must be in {values}')
    return v


def _check_value_in_intreval(field_name, v, lower_bound, upper_bound):
    if v < lower_bound or v > upper_bound:
        raise ValueError(f'{field_name} must belong interval [{lower_bound}, {upper_bound}])')
    return v


class HeartDiseasesModel(BaseModel):
    age: List[int]
    sex: List[int]
    cp: List[int]
    trestbps: List[int]
    chol: List[int]
    fbs: List[int]
    restecg: List[int]
    thalach: List[int]
    exang: List[int]
    oldpeak: List[int]
    slope: List[int]
    ca: List[int]
    thal: List[int]

    @validator('age', each_item=True)
    def age_must_belong_correct_interval(cls, v):
        return _check_value_in_intreval('age', v, 0, 150)

    @validator('sex', each_item=True)
    def sex_must_be_correct(cls, v):
        return _check_value_in_list('sex', v, (0, 1))

    @validator('cp', each_item=True)
    def cp_must_be_correct(cls, v):
        return _check_value_in_list('cp', v, (0, 1, 2, 3))

    @validator('trestbps', each_item=True)
    def trestbps_must_belong_correct_interval(cls, v):
        return _check_value_in_intreval('trestbps', v, 0, 300)

    @validator('fbs', each_item=True)
    def fbs_must_be_correct(cls, v):
        return _check_value_in_list('fbs', v, (0, 1))

    @validator('restecg', each_item=True)
    def restecg_must_be_correct(cls, v):
        return _check_value_in_list('restecg', v, (0, 1, 2))

    @validator('thalach', each_item=True)
    def thalach_must_belong_correct_interval(cls, v):
        return _check_value_in_intreval('thalach', v, 0, 300)

    @validator('exang', each_item=True)
    def exang_must_be_correct(cls, v):
        return _check_value_in_list('exang', v, (0, 1))

    @validator('slope', each_item=True)
    def slope_must_be_correct(cls, v):
        return _check_value_in_list('slope', v, (0, 1, 2))

    @validator('ca', each_item=True)
    def ca_must_be_correct(cls, v):
        return _check_value_in_list('ca', v, (0, 1, 2, 3))

    @validator('thal', each_item=True)
    def thal_must_be_correct(cls, v):
        return _check_value_in_list('thal', v, (0, 1, 2))


class DiseasePredictResponse(BaseModel):
    condition: int

    @validator('condition')
    def condition_must_be_correct(cls, v):
        if v not in (0, 1):
            raise ValueError('condition must be 0 or 1')
        return v
