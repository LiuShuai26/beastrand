# algorithms/base.py
from typing import Dict, Any, Protocol

class Algorithm(Protocol):
    def prepare_batch(self, slot_view, ctx) -> Dict[str, Any]:
        """Turn one (or more) slot views into a trainable batch dict."""
        ...

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Run one optimization step/epoch on the given batch and return stats."""
        ...
