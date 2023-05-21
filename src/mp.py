import os

import multiprocessing as mp
from src.Config import BaseConfig
from dotenv import load_dotenv


load_dotenv()

MAX_MODELS = int(os.environ["MAX_MODELS"])
MAX_PROCESS = int(os.environ["MAX_PROCESS"])
MODEL_CACHE = dict()

LOCK = mp.Lock()

COUNT = mp.Value('i', 0)


def is_cache_full():
    return len(MODEL_CACHE) >= MAX_MODELS


def update_cache(model, config: BaseConfig):
    MODEL_CACHE.update({config.model_path: model})


def delete_from_cache(config: BaseConfig):
    del MODEL_CACHE[config.model_path]


def contains_model(config: BaseConfig):
    return config.model_path in MODEL_CACHE


def get_model(config: BaseConfig):
    return MODEL_CACHE[config.model_path]


def inc_process_count(count):
    with count.get_lock():
        count.value = count.value + 1


def dec_process_count(count):
    with count.get_lock():
        count.value = count.value - 1


def has_free_process(count):
    with count.get_lock():
        return count.value < MAX_PROCESS

def process_count(count):
    return (count.value, MAX_PROCESS)
