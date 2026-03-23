"""
DecisionEngine: Optional utility for inspecting or transforming GenerationPlans.

The core pipeline no longer requires this — Designer.generate() handles
validation + plan building + execution directly through the domain.

Use DecisionEngine when you need to:
  - Inspect a plan before executing it (editors, preview tools, debuggers)
  - Transform or enrich a plan between build_plan() and execute()
  - Build tooling that separates planning from execution explicitly

For most use cases, prefer: designer.generate() or designer.dry_run()
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Intent, GenerationPlan
    from .domains.base import DomainBase
    from .seeds import Seed


class DecisionEngine:
    """
    Stateless utility that resolves Intent → GenerationPlan via a domain.

    Wraps the same validation + plan-building logic as Designer._prepare_plan(),
    but exposes it as a standalone object for tooling and introspection use cases.
    """

    def __init__(self, domain: "DomainBase"):
        self.domain = domain

    def resolve(self, intent: "Intent", seed: "Seed") -> "GenerationPlan":
        """Resolve an Intent into a concrete GenerationPlan (without executing it)."""
        self.domain.validate_intent(intent)
        plan = self.domain.build_plan(intent)
        plan.seed = seed
        plan.metadata["intent_summary"] = dict(intent.items())
        plan.metadata["domain"] = self.domain.__class__.__name__
        return plan

    def explain(self, intent: "Intent", seed: "Seed") -> str:
        """Return a human-readable explanation of what would be generated."""
        plan = self.resolve(intent, seed)
        lines = [
            f"Domain    : {plan.metadata['domain']}",
            f"Seed      : {plan.seed}",
            f"Intent    : {plan.metadata['intent_summary']}",
            f"Steps ({len(plan.steps)}):",
        ]
        for i, (name, params) in enumerate(plan.steps, 1):
            lines.append(f"  {i}. {name}  <- {params}")
        if plan.context:
            lines.append(f"Context   : {plan.context}")
        return "\n".join(lines)
