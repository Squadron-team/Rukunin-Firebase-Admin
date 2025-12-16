from dataclasses import dataclass, field
from typing import Optional, Dict
from schemas.line_count_validation import LineCountValidationResult
from schemas.layout_validation_result import LayoutValidationResult

@dataclass
class FieldVerification:
    expected: str
    actual: str
    is_match: bool


@dataclass
class VerificationSummary:
    passed: bool
    final_verdict: str

@dataclass
class VerificationResult:
    summary: VerificationSummary = field(
        default_factory=lambda: VerificationSummary(
            passed=False,
            final_verdict="POTENTIALLY FAKE",
        )
    )
    field_verification: Optional[Dict[str, FieldVerification]] = None
    line_validation: Optional[LineCountValidationResult] = None
    layout_validation: Optional[LayoutValidationResult] = None
    early_detection: bool = True