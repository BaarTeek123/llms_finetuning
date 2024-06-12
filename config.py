from pathlib import Path
from typing import Union

from pydantic import BaseModel, Field, AliasPath


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
    EPOCHS: int = 3
    BATCH_SIZE: int = 32
    LR: float = 2e-5
    METRIC: str = 'accuracy'
    MODEL_NAME: str = "prajjwal1/bert-tiny"
    DATASET: str = "qqp"
    TASK: str = Field(alias='task')
    OUTPUT_DIR: Union[str, Path] = None
    PUSH_TO_HUB: bool = True
    MAX_LENGTH: int = 512

    def __init__(self, **data):
        super().__init__(**data)
        self.OUTPUT_DIR = f"out/{self.MODEL_NAME.split('/')[-1]}-{self.TASK}-{self.DATASET}"


