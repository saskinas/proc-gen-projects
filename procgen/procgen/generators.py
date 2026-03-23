"""
GeneratorBase and CompositeGenerator.

Generators are leaf workers: they take structured params + context
and return domain-specific output objects.

To create a generator:
    class MyGenerator(GeneratorBase):
        def generate(self, params, context):
            rng = self.rng()   # seeded Random instance
            return {"value": rng.randint(0, 100)}
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .seeds import Seed


class GeneratorBase(ABC):
    """
    Abstract base for all generators.

    Subclasses implement `generate(params, context) -> Any`.

    Attributes:
        seed    — child Seed assigned by the execution engine
    """

    def __init__(self, seed: "Seed"):
        self.seed = seed

    def rng(self) -> random.Random:
        """Return a seeded Random instance. Call once per generate() for reproducibility."""
        return self.seed.rng()

    def child_seed(self, name: str) -> "Seed":
        """Get a named child seed for sub-components."""
        return self.seed.child(name)

    @abstractmethod
    def generate(self, params: dict, context: dict) -> Any:
        """
        Produce output.

        Args:
            params  — resolved parameters from the GenerationPlan
            context — accumulated outputs from prior steps + shared context

        Returns:
            Any domain-specific object.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(seed={self.seed})"


class CompositeGenerator(GeneratorBase):
    """
    A generator that delegates to an ordered list of child generators
    and merges their outputs into a dict.

    Useful for building compound generators without registering each
    sub-generator individually.

    Example::

        class SceneGenerator(CompositeGenerator):
            def build_children(self, params, context):
                return [
                    ("terrain",   TerrainGenerator(self.child_seed("terrain"))),
                    ("weather",   WeatherGenerator(self.child_seed("weather"))),
                ]
    """

    def build_children(self, params: dict, context: dict) -> list[tuple[str, GeneratorBase]]:
        """Return (name, generator) pairs. Override in subclass."""
        return []

    def generate(self, params: dict, context: dict) -> dict[str, Any]:
        children = self.build_children(params, context)
        result: dict[str, Any] = {}
        running_context = dict(context)
        for name, gen in children:
            out = gen.generate(params, running_context)
            result[name] = out
            running_context[name] = out
        return result
