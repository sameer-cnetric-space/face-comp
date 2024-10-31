from dataclasses import dataclass
from typing import Dict

@dataclass
class LivenessCheckResult:
    """Data class to store check results"""
    passed: bool
    score: float
    details: Dict
    duration: float