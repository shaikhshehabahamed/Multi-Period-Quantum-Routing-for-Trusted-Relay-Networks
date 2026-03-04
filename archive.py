from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """Return True if `a` dominates `b` (minimization)."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


@dataclass
class ArchiveItem:
    objectives: Tuple[float, ...]
    payload: Dict[str, Any]


class ParetoArchive:
    """Pareto archive for minimization objectives with optional diversity truncation."""

    def __init__(self, max_size: int | None = None) -> None:
        self.max_size = max_size
        self.items: List[ArchiveItem] = []

    def update(self, obj: Tuple[float, ...], payload: Dict[str, Any]) -> None:
        # If dominated by any existing item, do nothing.
        for it in self.items:
            if dominates(it.objectives, obj):
                return

        # Remove all items dominated by the new point.
        self.items = [it for it in self.items if not dominates(obj, it.objectives)]
        self.items.append(ArchiveItem(obj, payload))

        if self.max_size is not None and len(self.items) > self.max_size:
            self._truncate()

    def _truncate(self) -> None:
        """Truncate archive to `max_size` using crowding distance (NSGA-II style)."""
        if self.max_size is None or len(self.items) <= self.max_size:
            return

        n = len(self.items)
        m = len(self.items[0].objectives)
        dist = [0.0] * n

        # Compute crowding distance along each objective.
        for k in range(m):
            order = sorted(range(n), key=lambda i: self.items[i].objectives[k])

            # Always keep boundary points for each objective.
            dist[order[0]] = float("inf")
            dist[order[-1]] = float("inf")

            minv = float(self.items[order[0]].objectives[k])
            maxv = float(self.items[order[-1]].objectives[k])
            rng = maxv - minv
            if rng <= 0.0:
                continue

            for t in range(1, n - 1):
                prevv = float(self.items[order[t - 1]].objectives[k])
                nextv = float(self.items[order[t + 1]].objectives[k])
                dist[order[t]] += (nextv - prevv) / rng

        keep = sorted(range(n), key=lambda i: (-dist[i], i))[: self.max_size]
        self.items = [self.items[i] for i in keep]

    def as_points(self) -> List[Tuple[float, ...]]:
        return [it.objectives for it in self.items]

    def __len__(self) -> int:
        return len(self.items)