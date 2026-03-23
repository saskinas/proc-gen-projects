"""
procgen.random — Richer randomness primitives for generators.

WeightedTable   — pick items by weight
MarkovChain     — next-state by transition probabilities
GaussianSampler — bounded normal distribution sampling
CurveMapper     — map a 0..1 float through a named curve
"""

from __future__ import annotations
import random
import math
from typing import Any, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# WeightedTable
# ---------------------------------------------------------------------------

class WeightedTable:
    """
    A discrete probability table. Pick items by relative weight.

    Example::

        table = WeightedTable([("sword", 10), ("bow", 5), ("staff", 2)])
        item = table.pick(rng)          # "sword" most likely
        items = table.pick_many(rng, 3) # 3 samples without replacement
    """

    def __init__(self, items: list[tuple[Any, float]]):
        if not items:
            raise ValueError("WeightedTable cannot be empty")
        self._items = [item for item, _ in items]
        self._weights = [float(w) for _, w in items]
        total = sum(self._weights)
        self._normalized = [w / total for w in self._weights]

    def pick(self, rng: random.Random) -> Any:
        return rng.choices(self._items, weights=self._weights, k=1)[0]

    def pick_many(self, rng: random.Random, n: int, replace: bool = True) -> list:
        if replace:
            return rng.choices(self._items, weights=self._weights, k=n)
        # Without replacement: use weighted shuffle
        indexed = list(zip(self._normalized, self._items))
        result = []
        for _ in range(min(n, len(self._items))):
            total = sum(w for w, _ in indexed)
            r = rng.random() * total
            cumulative = 0
            for i, (w, item) in enumerate(indexed):
                cumulative += w
                if r <= cumulative:
                    result.append(item)
                    indexed.pop(i)
                    break
        return result

    def with_bias(self, key: Any, multiplier: float) -> "WeightedTable":
        """Return a new table with one item's weight multiplied."""
        new_items = []
        for item, weight in zip(self._items, self._weights):
            w = weight * multiplier if item == key else weight
            new_items.append((item, w))
        return WeightedTable(new_items)

    def __repr__(self) -> str:
        top = sorted(zip(self._weights, self._items), reverse=True)[:3]
        summary = ", ".join(f"{i}({w:.1f})" for w, i in top)
        return f"WeightedTable([{summary}, ...])"


# ---------------------------------------------------------------------------
# MarkovChain
# ---------------------------------------------------------------------------

class MarkovChain:
    """
    A first-order Markov chain for sequence generation.

    States transition probabilistically. Useful for:
    - chord progressions
    - room type sequences
    - weather pattern sequences
    - narrative beat flows

    Example::

        chain = MarkovChain({
            "I":  [("IV", 0.4), ("V", 0.4), ("vi", 0.2)],
            "IV": [("V", 0.5), ("I", 0.3), ("ii", 0.2)],
            "V":  [("I", 0.7), ("vi", 0.3)],
            "vi": [("IV", 0.5), ("ii", 0.3), ("V", 0.2)],
        })
        progression = chain.walk(rng, start="I", steps=8)
    """

    def __init__(self, transitions: dict[str, list[tuple[str, float]]]):
        self._transitions: dict[str, WeightedTable] = {}
        for state, edges in transitions.items():
            self._transitions[state] = WeightedTable(edges)

    def next(self, rng: random.Random, state: str) -> str:
        if state not in self._transitions:
            raise KeyError(f"Unknown state '{state}'")
        return self._transitions[state].pick(rng)

    def walk(self, rng: random.Random, start: str, steps: int) -> list[str]:
        """Generate a sequence of states starting from `start`."""
        seq = [start]
        current = start
        for _ in range(steps - 1):
            current = self.next(rng, current)
            seq.append(current)
        return seq

    def states(self) -> list[str]:
        return list(self._transitions)

    def __repr__(self) -> str:
        return f"MarkovChain(states={self.states()})"


# ---------------------------------------------------------------------------
# GaussianSampler
# ---------------------------------------------------------------------------

class GaussianSampler:
    """
    Sample from a normal distribution, clamped to [lo, hi].

    Useful for: velocity variation, room sizing, tempo jitter, elevation noise.

    Example::

        sampler = GaussianSampler(mean=0.5, std=0.15, lo=0.0, hi=1.0)
        value = sampler.sample(rng)
    """

    def __init__(self, mean: float, std: float, lo: float = 0.0, hi: float = 1.0):
        self.mean = mean
        self.std = std
        self.lo = lo
        self.hi = hi

    def sample(self, rng: random.Random) -> float:
        val = rng.gauss(self.mean, self.std)
        return max(self.lo, min(self.hi, val))

    def sample_int(self, rng: random.Random) -> int:
        return int(round(self.sample(rng)))

    def __repr__(self) -> str:
        return f"GaussianSampler(μ={self.mean}, σ={self.std}, [{self.lo}, {self.hi}])"


# ---------------------------------------------------------------------------
# CurveMapper
# ---------------------------------------------------------------------------

class CurveMapper:
    """
    Map a normalised float [0, 1] through a named non-linear curve.

    Useful for: energy → tempo (exponential), danger → enemy count (power),
    complexity → note density (s-curve).

    Curves:
        "linear"      — identity
        "ease_in"     — slow start, fast end  (quadratic)
        "ease_out"    — fast start, slow end
        "ease_inout"  — smooth S  (cubic Hermite)
        "power2"      — x²
        "power3"      — x³
        "sqrt"        — √x
        "log"         — logarithmic
        "step"        — 0 below 0.5, 1 above
    """

    _CURVES = {
        "linear":     lambda x: x,
        "ease_in":    lambda x: x * x,
        "ease_out":   lambda x: 1 - (1 - x) ** 2,
        "ease_inout": lambda x: x * x * (3 - 2 * x),
        "power2":     lambda x: x ** 2,
        "power3":     lambda x: x ** 3,
        "sqrt":       lambda x: math.sqrt(x),
        "log":        lambda x: math.log1p(x * (math.e - 1)),
        "step":       lambda x: 0.0 if x < 0.5 else 1.0,
    }

    def __init__(self, curve: str = "linear", out_lo: float = 0.0, out_hi: float = 1.0):
        if curve not in self._CURVES:
            raise ValueError(f"Unknown curve '{curve}'. Options: {list(self._CURVES)}")
        self.curve = curve
        self.out_lo = out_lo
        self.out_hi = out_hi
        self._fn = self._CURVES[curve]

    def map(self, x: float) -> float:
        """Map x ∈ [0,1] through the curve into [out_lo, out_hi]."""
        x = max(0.0, min(1.0, x))
        curved = self._fn(x)
        return self.out_lo + curved * (self.out_hi - self.out_lo)

    def map_int(self, x: float) -> int:
        return int(round(self.map(x)))

    @classmethod
    def curves(cls) -> list[str]:
        return list(cls._CURVES)

    def __repr__(self) -> str:
        return f"CurveMapper(curve='{self.curve}', out=[{self.out_lo}, {self.out_hi}])"
