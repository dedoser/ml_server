import joblib
import os
import traceback
import logging
import src.mp as mp

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.Config import BaseConfig, FitConfig, PredictConfig
from src.Config import Texts, Labels
from src.Config import ReturnValue, PredictReturnValue
from src.Config import FeatureType, ModelType
from src.Logging import LogConfig
from logging.config import dictConfig

FEATURE_TYPE_MAPPING = {
    FeatureType.COUNT: CountVectorizer,
    FeatureType.TFID: TfidfVectorizer,
}

MODEL_MAPPING = {
    ModelType.LOGISTIC_REGRESSION: LogisticRegression,
    ModelType.KNN: KNeighborsClassifier,
    ModelType.RANDOM_FOREST: RandomForestClassifier,
}

dictConfig(LogConfig().dict())
logger = logging.getLogger("server")


save_dir = os.environ["SAVE_DIR"]

class Classifier:

    @staticmethod
    def fit(texts: Texts, labels: Labels, config: FitConfig, count) -> ReturnValue:
        try:
            mp.inc_process_count(count)
            vectorizer = FEATURE_TYPE_MAPPING[config.feature_type]()

            data = vectorizer.fit_transform(texts.texts)
            params = {}
            if config.params is not None:
                params = config.params
            model = MODEL_MAPPING[config.model_type](**params)
            logger.info(f"Start fitting model {config.model_path}")
            model.fit(data, labels.labels)
            logger.info(f"Stop fitting model {config.model_path}")
            save_file = {"model": model, "featureType": vectorizer}
            Classifier.unload(save_file, config)
            mp.dec_process_count(count)
        except Exception as e:
            mp.dec_process_count(count)
            return ReturnValue(success=False,
                               message="Cannot fit model",
                               traceback=traceback.format_exception(e))
        return ReturnValue(success=True, message="Success fitting")

    @staticmethod
    def predict(save_model, texts: Texts, config: PredictConfig) -> ReturnValue:
        try:
            model = save_model["model"]
            vectorizer = save_model["featureType"]
            logger.info(f"Start predict model {config.model_path}")
            data = vectorizer.fit_transform(texts.texts)
            logger.info(f"Finish predict model {config.model_path}")
            return PredictReturnValue(success=True,
                                      prediction=model.predict(data))
        except Exception as e:
            return ReturnValue(success=False,
                               message=str(e.args),
                               traceback=traceback.format_exception(e))

    @staticmethod
    def load(config: BaseConfig):
        path = os.path.join(save_dir, f"{config.model_path}.joblib")

        if not os.path.exists(path):
            raise ValueError(f"Cannot find model with such name - {config.model_path}")
        logger.info(f"Unpack model {config.model_path} from disk to inference")
        return joblib.load(path)

    @staticmethod
    def unload(model, config: BaseConfig):
        path = os.path.join(save_dir, f"{config.model_path}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved model {config.model_path} on disk")

    @staticmethod
    def remove(config: BaseConfig):
        path = os.path.join(save_dir, f"{config.model_path}.joblib")
        os.remove(path)

    @staticmethod
    def remove_all():
        for file in os.listdir(save_dir):
            path = os.path.join(save_dir, file)
            os.remove(path)

    @staticmethod
    def status(config: BaseConfig):
        is_saved = False
        file_name = f"{config.model_path}.joblib"
        if file_name in os.listdir(save_dir):
            is_saved = True
        is_inferenced = False
        if mp.contains_model(config):
            is_inferenced = True
        
        return {'saved': is_saved, 'inference': is_inferenced}