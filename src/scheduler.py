from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass


@dataclass
class _Partition:
    loads: list[float]    
    items: list[list[int]]  

    @property
    def key(self) -> float:
        return self.loads[0]


def ldm_partition(weights: list[float], k: int) -> list[list[int]]:

    n = len(weights)
    if k <= 0:
        raise ValueError("k must be positive")
    if n == 0:
        return [[] for _ in range(k)]
    if k == 1:
        return [list(range(n))]

    counter = itertools.count()
    heap: list[tuple[float, int, _Partition]] = []
    for i, w in enumerate(weights):
        loads = [float(w)] + [0.0] * (k - 1)
        items: list[list[int]] = [[i]] + [[] for _ in range(k - 1)]
        heapq.heappush(heap, (-loads[0], next(counter), _Partition(loads, items)))

    while len(heap) > 1:
        _, _, a = heapq.heappop(heap)
        _, _, b = heapq.heappop(heap)

        b_asc_order = sorted(range(k), key=lambda idx: b.loads[idx])
        merged_loads = [0.0] * k
        merged_items: list[list[int]] = [[] for _ in range(k)]
        for slot, b_idx in enumerate(b_asc_order):
            merged_loads[slot] = a.loads[slot] + b.loads[b_idx]
            merged_items[slot] = a.items[slot] + b.items[b_idx]

        order = sorted(range(k), key=lambda idx: -merged_loads[idx])
        merged = _Partition(
            [merged_loads[idx] for idx in order],
            [merged_items[idx] for idx in order],
        )
        heapq.heappush(heap, (-merged.key, next(counter), merged))

    _, _, final = heap[0]
    return final.items
