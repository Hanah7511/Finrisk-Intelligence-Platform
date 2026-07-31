from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ApiConfig:
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retries: int = 3
    backoff_factor: float = 0.3
    data_key: Optional[str] = None  # key inside JSON response e.g. "data", "records"