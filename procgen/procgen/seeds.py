"""
Seed: Reproducible randomness with hierarchical sub-seeds.

Every generator gets its own child seed so results are fully reproducible
from a single top-level integer, yet each sub-system is independent.
"""

from __future__ import annotations
import random
import hashlib


class Seed:
    """
    Hierarchical seed: a root seed can produce named child seeds,
    all deterministically derived from the original value.

    Example:
        root = Seed(42)
        melody_seed = root.child("melody")   # always same value for seed 42
        rhythm_seed = root.child("rhythm")
        rng = melody_seed.rng()              # stdlib Random instance
    """

    def __init__(self, value: int | str | None = None):
        if value is None:
            value = random.randint(0, 2**32 - 1)
        if isinstance(value, str):
            value = int(hashlib.sha256(value.encode()).hexdigest(), 16) % (2**32)
        self._value: int = int(value)
        self._children: dict[str, "Seed"] = {}

    @property
    def value(self) -> int:
        return self._value

    def child(self, name: str) -> "Seed":
        """Return (or create) a deterministic child seed for the given name."""
        if name not in self._children:
            combined = f"{self._value}:{name}"
            child_val = int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**32)
            self._children[name] = Seed(child_val)
        return self._children[name]

    def rng(self) -> random.Random:
        """Return a seeded Random instance."""
        r = random.Random()
        r.seed(self._value)
        return r

    def __repr__(self) -> str:
        return f"Seed({self._value})"
