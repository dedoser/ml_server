from typing import Optional, List, Dict, Any
from pydantic import BaseModel, constr, PositiveInt
from enum import Enum


class ModelType(str, Enum):
    LOGISTIC_REGRESSION = 'lr'
    KNN = 'knn'
    RANDOM_FOREST = 'rf'


class FeatureType(str, Enum):
    TFID = "tfid"
    COUNT = "cnt"


class BaseConfig(BaseModel):
    model_path: constr(min_length=1)


class FitConfig(BaseConfig):
    feature_type: FeatureType
    model_type: ModelType
    params: Optional[Dict[str, Any]] = None


class PredictConfig(BaseConfig):
    limit: Optional[PositiveInt] = None


class Texts(BaseModel):
    texts: List[str]


class Labels(BaseModel):
    labels: List[str]


class ReturnValue(BaseModel):
    success: bool
    message: Optional[str] = None
    traceback: Optional[List[str]] = None


class PredictReturnValue(ReturnValue):
    prediction: Optional[List[float]]
