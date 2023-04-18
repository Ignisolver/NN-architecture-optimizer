from dataclasses import dataclass
from pathlib import Path
from typing import NewType, List

PROJECT_PATH = Path(__file__).parent
RAW_DATA_PATH = PROJECT_PATH.joinpath("raw_data")
READY_DATA_PATH = PROJECT_PATH.joinpath("ready_data")


@dataclass
class Layer:
    pass


NetworkData = NewType("NetworkData", List[Layer])
Population = NewType("Population", List[NetworkData])