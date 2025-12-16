from dataclasses import dataclass

@dataclass
class LineCountValidationResult:
    is_valid: bool
    detected_lines: int
    expected_lines: int
    reason: str
