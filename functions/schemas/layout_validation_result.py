from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class LayoutValidationResult:
    is_valid: bool
    similarity_score: float
    threshold: float
    detected_fingerprint: Any
    template_fingerprint: Any
    detailed_scores: Dict[str, float]
    violations: List[str]
    reason: str