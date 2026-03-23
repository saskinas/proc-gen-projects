# procgen — A Generic Procedural Generation Framework
"""
procgen: Designer-first procedural generation framework.

Flow:
    Designer → Intent → domain.build_plan() → domain.execute() → Output

The domain owns the full generation pipeline:
    - validate_intent  — check intent parameters against the domain's schema
    - build_plan       — translate intent into ordered generation steps
    - generators       — declare { name → GeneratorClass } for each step
    - execute          — run the plan (sequential by default, overridable)
    - assemble         — combine outputs into the final domain object

Usage:
    from procgen import Designer
    from procgen.domains import MusicDomain, DungeonDomain

    designer = Designer(domain=MusicDomain())
    designer.set("mood", "melancholic")
    designer.set("energy", 0.3)
    designer.set("complexity", "medium")

    result = designer.generate()
"""

from .core import Designer, Registry, GenerationPlan, Intent
from .engine import DecisionEngine
from .generators import GeneratorBase, CompositeGenerator
from .constraints import ConstraintSet, Rule
from .seeds import Seed
from . import export
from . import random as rnd
from . import grammar
from . import wfc

__all__ = [
    "Designer",
    "Registry",
    "GenerationPlan",
    "Intent",
    "DecisionEngine",
    "GeneratorBase",
    "CompositeGenerator",
    "ConstraintSet",
    "Rule",
    "Seed",
    "export",
    "rnd",
    "grammar",
    "wfc",
]

__version__ = "0.1.0"
