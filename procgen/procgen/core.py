"""
Core types for the procgen framework.

Intent          — The designer's high-level decisions (key→value).
GenerationPlan  — Resolved, structured instructions ready for generators.
Registry        — Optional named registry for plugin/dynamic generator binding.
Designer        — The main human-facing entry point.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .generators import GeneratorBase
    from .seeds import Seed
    from .domains.base import DomainBase


# ---------------------------------------------------------------------------
# Intent: raw designer decisions
# ---------------------------------------------------------------------------

class Intent:
    """
    A flat, typed bag of designer decisions.

    Designers set high-level knobs here; the domain reads these
    and converts them into a structured GenerationPlan.

    Supported value types:
        str         — categorical choices  ("mood": "dark")
        float       — continuous [0, 1]   ("energy": 0.7)
        int         — discrete count      ("rooms": 5)
        bool        — toggle              ("looping": True)
        list        — multi-select        ("instruments": ["piano", "cello"])
    """

    def __init__(self):
        self._values: dict[str, Any] = {}
        self._meta: dict[str, dict] = {}   # optional: hints, ranges, descriptions

    # --- set / get ---------------------------------------------------------

    def set(self, key: str, value: Any, *, description: str = "", choices: list = None, range: tuple = None) -> "Intent":
        self._values[key] = value
        self._meta[key] = {
            "description": description,
            "choices": choices,
            "range": range,
        }
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._values:
            raise KeyError(f"Intent is missing required key '{key}'")
        return self._values[key]

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def as_dict(self) -> dict:
        return dict(self._values)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"Intent({pairs})"


# ---------------------------------------------------------------------------
# GenerationPlan: structured instructions produced by build_plan()
# ---------------------------------------------------------------------------

@dataclass
class GenerationPlan:
    """
    Structured instructions produced by a domain's build_plan() from designer Intent.

    Generators read from this plan; they never see the raw Intent directly.
    This separation lets the domain normalise, validate, and enrich inputs.

    Fields:
        steps       — ordered list of (generator_name, params) to execute
        context     — shared read-only data available to all generators
        seed        — the root Seed for this generation run
        metadata    — free-form annotations (domain, intent summary, etc.)
    """
    steps: list[tuple[str, dict]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    seed: "Seed | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, generator_name: str, params: dict | None = None) -> "GenerationPlan":
        self.steps.append((generator_name, params or {}))
        return self

    def __repr__(self) -> str:
        step_names = [name for name, _ in self.steps]
        return f"GenerationPlan(steps={step_names}, seed={self.seed})"


# ---------------------------------------------------------------------------
# Registry: optional utility for plugin / dynamic generator binding
# ---------------------------------------------------------------------------

class Registry:
    """
    Optional named registry mapping generator names to factory callables.

    Not required by the core pipeline — domains declare their generators
    via the `generators` property. Use Registry for plugin systems or
    scenarios where generators are bound late (loaded from external packages,
    user-supplied overrides, etc.).

    Factory signature: factory(seed: Seed) -> GeneratorBase
    """

    def __init__(self):
        self._factories: dict[str, Callable] = {}

    def register(self, name: str, factory: Callable | None = None):
        """
        Register a generator factory by name.

        Can be used as a decorator::

            @registry.register("melody")
            def make_melody(seed):
                return MelodyGenerator(seed)
        """
        if factory is not None:
            self._factories[name] = factory
            return factory

        def decorator(fn):
            self._factories[name] = fn
            return fn
        return decorator

    def build(self, name: str, seed: "Seed") -> "GeneratorBase":
        if name not in self._factories:
            raise KeyError(f"No generator registered under '{name}'. "
                           f"Available: {list(self._factories)}")
        return self._factories[name](seed)

    def names(self) -> list[str]:
        return list(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __repr__(self) -> str:
        return f"Registry(generators={self.names()})"


# ---------------------------------------------------------------------------
# Designer: the main human-facing API
# ---------------------------------------------------------------------------

class Designer:
    """
    The designer's entry point.

    A Designer holds:
    - A domain object that owns the full generation pipeline
    - An Intent (high-level decisions)
    - Optionally a seed for reproducibility

    Typical workflow::

        designer = Designer(domain=MusicDomain())
        designer.set("mood", "dark")
        designer.set("energy", 0.8)
        result = designer.generate()

    The domain controls everything domain-specific:
    - What intent parameters are valid (schema)
    - How intent maps to generation steps (build_plan)
    - What generators exist (generators property)
    - How steps are executed (execute — sequential by default, overridable)
    - How outputs combine into the final object (assemble)
    """

    def __init__(self, domain: "DomainBase", seed: int | None = None):
        from .seeds import Seed
        self.domain = domain
        self.intent = Intent()
        self.seed = Seed(seed)

    # --- fluent API --------------------------------------------------------

    def set(self, key: str, value: Any, **meta) -> "Designer":
        """Set a designer intent value."""
        self.intent.set(key, value, **meta)
        return self

    def set_seed(self, seed: int) -> "Designer":
        from .seeds import Seed
        self.seed = Seed(seed)
        return self

    # --- generate ----------------------------------------------------------

    def _prepare_plan(self) -> GenerationPlan:
        """Validate intent, build plan, inject seed and metadata."""
        self.domain.validate_intent(self.intent)
        plan = self.domain.build_plan(self.intent)
        plan.seed = self.seed
        plan.metadata["intent_summary"] = dict(self.intent.items())
        plan.metadata["domain"] = self.domain.__class__.__name__
        if hasattr(self, "_overrides") and self._overrides:
            plan.metadata["_generator_overrides"] = dict(self._overrides)
        return plan

    def generate(self) -> Any:
        """
        Full pipeline: Intent → GenerationPlan → domain.execute() → Output

        The domain controls how execution happens — sequential by default,
        but overridable per domain for parallel, conditional, or custom flows.
        """
        plan = self._prepare_plan()
        return self.domain.execute(plan)

    def trace(self) -> tuple:
        """
        Like generate(), but returns (output, trace_dict) where trace_dict
        maps each step name to its raw generator output.

        Useful for pipeline introspection, debugging, and demos::

            score, trace = designer.trace()
            print(trace["harmonic_plan"])   # list[HarmonicEvent]
            print(trace["theme"])           # motif dict
        """
        plan = self._prepare_plan()
        return self.domain.traced_execute(plan)

    def explain(self) -> str:
        """
        Return a human-readable description of what this Designer would generate.

        Resolves intent → plan but does NOT run generators.  Shows all steps,
        their params, and the resolved intent values.

        Example::

            print(designer.explain())
            # Domain    : MusicDomain
            # Seed      : Seed(42)
            # Intent    : {'key': 'D', 'mode': 'minor', ...}
            # Steps (5):
            #   1. harmonic_plan  <- {key: D, ...}
            #   2. theme          <- {melodic_theme: fugue_subject, ...}
            #   ...
        """
        from .engine import DecisionEngine
        plan = self._prepare_plan()
        engine = DecisionEngine(self.domain)
        return engine.explain(self.intent, self.seed)

    def override(self, step_name: str, generator_class: Any) -> "Designer":
        """
        Replace the generator for a named step with a different class.

        The override applies only to this Designer instance and does not
        mutate the domain.  Multiple overrides may be chained::

            designer.override("harmonic_plan", MyHarmonicGenerator)
                    .override("theme", MyThemeGenerator)
            score = designer.generate()

        Args:
            step_name       — name matching a step in the domain's generators
            generator_class — a GeneratorBase subclass (not an instance)
        """
        if not hasattr(self, "_overrides"):
            self._overrides: dict[str, Any] = {}
        self._overrides[step_name] = generator_class
        return self

    def dry_run(self) -> GenerationPlan:
        """Resolve the intent to a plan without executing generators."""
        return self._prepare_plan()

    def __repr__(self) -> str:
        return f"Designer(domain={self.domain.__class__.__name__}, intent={self.intent})"
