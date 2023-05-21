import traceback
import src.mp as mp

from fastapi import APIRouter
from src.Config import BaseConfig, FitConfig, PredictConfig, ReturnValue, Texts, Labels
from src.Classifier import Classifier
from multiprocessing import Process

conrtollers = APIRouter()


@conrtollers.post("/fit")
async def fit(x: Texts, y: Labels, config: FitConfig) -> ReturnValue:
    if mp.has_free_process(mp.COUNT):
        process = Process(target=Classifier.fit,
                          args=(x, y, config, mp.COUNT))
        process.start()
        return ReturnValue(success=True)
    else:
        return ReturnValue(success=False,
                           message="No free process")


@conrtollers.post("/predict")
async def predict(x: Texts, config: PredictConfig) -> ReturnValue:
    if mp.contains_model(config):
        model = mp.get_model(config)
        return Classifier.predict(model, x, config)
    return ReturnValue(success=False,
                       message=f"Model with path {config.model_path} is not loaded")


@conrtollers.post("/load")
async def load(config: BaseConfig) -> ReturnValue:
    try:
        if mp.contains_model(config):
            return ReturnValue(success=True, message=f"Model {config.model_path} is already loaded")
        if mp.is_cache_full():
            return ReturnValue(success=True, message=f"Max amount of loaded models")
        model = Classifier.load(config)
        mp.update_cache(model, config)
        return ReturnValue(success=True)
    except Exception as e:
        return ReturnValue(success=False,
                           message="Unexpected error during load", traceback=traceback.format_exception(e))


@conrtollers.post("/unload")
async def unload(config: BaseConfig) -> ReturnValue:
    if not mp.contains_model(config):
        return ReturnValue(success=False,
                           message=f"Model with path {config.model_path} is not loaded. Cannot unload")
    model = mp.get_model(config)
    Classifier.unload(model, config)
    mp.delete_from_cache(config)
    return ReturnValue(success=True)


@conrtollers.post("/remove")
async def remove(config: BaseConfig) -> ReturnValue:
    try:
        Classifier.remove(config)
        return ReturnValue(success=True)
    except Exception as e:
        return ReturnValue(success=False,
                           message=f"Failed to remove model {config.model_path}",
                           traceback=traceback.format_exception(e))



@conrtollers.post("/remove_all")
async def remove_all() -> ReturnValue:
    try:
        Classifier.remove_all()
        return ReturnValue(success=True)
    except Exception as e:
        return ReturnValue(success=False,
                           message="Failed to remove all",
                           traceback=traceback.format_exception(e))


@conrtollers.get("/")
async def hello():
    return {'data': 'ml_server'}

@conrtollers.get("/process")
async def process_count():
    (count, max_proc) = mp.process_count(mp.COUNT)

    return  {'count': count, 'max_count': max_proc}

@conrtollers.get("/status/{model_path}")
async def status(model_path: str):
    return Classifier.status(BaseConfig(model_path=model_path))
