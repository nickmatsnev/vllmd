from dataclasses import dataclass, field
from time import time


@dataclass
class Worker:
    id: str
    host: str
    port: int

    gpu_count: int
    gpu_memory_gb: int

    active_requests: int = 0
    healthy: bool = True
    last_seen: float = field(default_factory=time)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
