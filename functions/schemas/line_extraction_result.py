from dataclasses import dataclass
from typing import Tuple, List

Box = Tuple[int, int, int, int]


@dataclass
class LineExtractionResult:
    line_number: int
    text: str
    field: str
    boxes: List[Box]