from dataclasses import dataclass

from .base_config import BaseConfig
from .settings import *


@dataclass
class RunConfig(BaseConfig, RunSettings):
    pass


@dataclass
class ColBERTConfig(RunSettings, ResourceSettings, IndexingSettings,
                    SearchSettings, BaseConfig):
    pass
