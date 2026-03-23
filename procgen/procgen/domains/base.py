"""
DomainBase: Abstract interface every domain must implement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import Intent, GenerationPlan
    from ..generators import GeneratorBase
    from ..seeds import Seed


class DomainBase(ABC):

    def validate_intent(self, intent: "Intent") -> None:
        if hasattr(self, "schema"):
            self.schema.validate(intent)

    @abstractmethod
    def build_plan(self, intent: "Intent") -> "GenerationPlan":
        """Translate intent into an ordered generation plan."""
        ...

    @property
    @abstractmethod
    def generators(self) -> dict[str, "Callable[[Seed], GeneratorBase]"]:
        """
        Declare this domain's generator factories.

        Returns a mapping of generator_name → factory(seed) → GeneratorBase.
        The domain owns these; the framework reads them during execute().

        Example::

            @property
            def generators(self):
                return {
                    "layout": RoomLayoutGenerator,
                    "populate": RoomPopulationGenerator,
                }
        """
        ...

    @abstractmethod
    def assemble(self, outputs: dict[str, Any], plan: "GenerationPlan") -> Any:
        """Combine generator outputs into the final domain object."""
        ...

    def execute(self, plan: "GenerationPlan") -> Any:
        """
        Run a generation plan and return the assembled output.

        The default implementation executes steps sequentially, accumulating
        outputs into a context dict passed to each subsequent generator.

        Supports generator overrides injected by Designer.override():
        any entry in plan.metadata["_generator_overrides"] replaces the
        corresponding entry in self.generators for this run only.

        Override this method to implement domain-specific execution strategies:
        parallel steps, conditional branching, retries, custom error handling, etc.

        Example override (parallel independent steps)::

            def execute(self, plan):
                from concurrent.futures import ThreadPoolExecutor
                independent = ["heightmap", "moisture"]
                outputs = {}
                with ThreadPoolExecutor() as pool:
                    futs = {name: pool.submit(self._run_step, plan, name, {}, outputs)
                            for name in independent}
                    for name, fut in futs.items():
                        outputs[name] = fut.result()
                # run dependent steps sequentially after
                for step_name, params in plan.steps:
                    if step_name not in outputs:
                        outputs[step_name] = self._run_step(plan, step_name, params, outputs)
                return self.assemble(outputs, plan)
        """
        outputs: dict[str, Any] = {}
        gens = dict(self.generators)  # mutable copy so overrides don't mutate domain
        gens.update(plan.metadata.get("_generator_overrides", {}))
        for step_name, params in plan.steps:
            if step_name not in gens:
                raise KeyError(
                    f"Domain '{self.__class__.__name__}' has no generator '{step_name}'. "
                    f"Available: {list(gens)}"
                )
            gen = gens[step_name](plan.seed.child(step_name))
            result = gen.generate(params, {**plan.context, **outputs})
            outputs[step_name] = result
        return self.assemble(outputs, plan)

    def traced_execute(self, plan: "GenerationPlan") -> tuple:
        """
        Like execute(), but also returns all intermediate step outputs.

        Returns:
            (final_output, trace)  where trace is dict[step_name → raw output].

        Useful for debugging, demos, and understanding the pipeline::

            score, trace = domain.traced_execute(plan)
            print(trace["harmonic_plan"])   # list of HarmonicEvent
            print(trace["theme"])           # motif dict
            print(trace["voice_lines"])     # per-voice pitch sequences
        """
        outputs: dict[str, Any] = {}
        gens = dict(self.generators)
        gens.update(plan.metadata.get("_generator_overrides", {}))
        for step_name, params in plan.steps:
            if step_name not in gens:
                raise KeyError(
                    f"Domain '{self.__class__.__name__}' has no generator '{step_name}'. "
                    f"Available: {list(gens)}"
                )
            gen = gens[step_name](plan.seed.child(step_name))
            result = gen.generate(params, {**plan.context, **outputs})
            outputs[step_name] = result
        return self.assemble(outputs, plan), outputs
