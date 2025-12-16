from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class LayoutFingerprint:
    num_lines: int = 0

    avg_line_height: float = 0.0
    std_line_height: float = 0.0

    avg_line_width: float = 0.0
    std_line_width: float = 0.0

    avg_line_gap: float = 0.0
    std_line_gap: float = 0.0

    alignment_variance: float = 0.0

    zone_distribution: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width_distribution: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    line_y_positions: List[float] = field(default_factory=list)