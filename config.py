from os.path import join
from pathlib import Path
from typing import Union

import numpy as np
from pydantic import BaseModel, Field


class DataArgs(BaseModel):
    pad_to_max_length: bool = True
    max_seq_length: int = 128
    overwrite_cache: bool = False
    max_train_samples: int = None
    max_predict_samples: int = None
    max_eval_samples: int = None
    # max_train_samples: int = 10000
    # max_predict_samples: int = 2000
    # max_eval_samples: int = 1000

    class Config:
        frozen = True


class Config(BaseModel):
    EPOCHS: int = 10
    BATCH_SIZE: int = 32
    LR: float = 0.05
    METRIC: str = 'accuracy'
    MODEL_NAME: str = "prajjwal1/bert-tiny"
    DATASET: str = Field(alias='dataset')
    TASK: str = Field(alias='task')
    MODEL_OUTPUT_DIR: Union[str, Path] = None
    RESULTS_PATH: Union[str, Path] = None
    PUSH_TO_HUB: bool = True
    MAX_LENGTH: int = 512
    LOGGER_STEP: int = 5000

    def __init__(self, **data):
        super().__init__(**data)
        self.MODEL_OUTPUT_DIR = f"out/{self.MODEL_NAME.split('/')[-1]}-{self.TASK}-{self.DATASET}"
        self.RESULTS_PATH = join('results', 'evaluation_results.json')


class PrivateConfig(Config):
    DELTA: float = 10e-6
    EPSILON: float = np.inf
    MAX_GRAD_NORM: float = 1.0
    NOISE_SCALE: float = 1.0
    SAMPLING_RATE: float = 0.1

    def __init__(self, **data):
        super().__init__(**data)
        self.RESULTS_PATH = join('results', 'evaluation_results_dp.json')
        self.MODEL_OUTPUT_DIR = f"out/{self.MODEL_NAME.split('/')[-1]}-{self.TASK}-{self.DATASET}_DP_{self.DELTA}_{self.EPSILON}"