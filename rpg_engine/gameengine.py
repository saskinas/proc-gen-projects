"""
rpg_engine — Public API.

Single entry point for generating and running JRPG-style games.

Usage::

    from gameengine import GameDesigner, GameEngine

    world = (GameDesigner()
        .set("genre",      "jrpg")
        .set("tone",       "dark")
        .set("protagonist","mage")
        .set("antagonist", "ancient_evil")
        .set("world_size", "medium")
        .set("difficulty", "hard")
        .set("title",      "Echoes of the Void")
        .set("protagonist_name", "Kira")
        .generate(seed=42))

    print(world.describe())

    # Run (requires: pip install pygame)
    from gameengine import GameEngine
    GameEngine(world).run()

    # Or one-liner:
    GameDesigner().set("tone","dark").run(seed=42)

    # Load a saved game:
    engine = GameEngine.load_game("saves/save_01.json")
    engine.run()

Available parameters (call GameDesigner().options() for full list):
    genre           jrpg | action_rpg | dungeon_crawler | visual_novel
    tone            heroic | dark | comedic | mysterious | tragic
    world_size      tiny | small | medium | large | epic
    difficulty      easy | normal | hard | expert
    protagonist     warrior | mage | rogue | scholar | outcast
    antagonist      empire | demon_lord | corruption | ancient_evil | rival
    chapters        1–6
    party_size      1–4
    title           str  (auto-generated if empty)
    world_name      str  (auto-generated if empty)
    protagonist_name str
"""

from __future__ import annotations
import sys, os

# Project root on path so all imports work from any working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from procgen.core import Designer
from procgen_domain import JRPGDomain
from engine.core.world import World
from engine.runtime import GameEngine   # noqa: F401 — re-exported


class GameDesigner:
    """
    Fluent builder for generating game worlds.

    All parameters are optional — unset ones use sensible defaults.

    Example::

        world = GameDesigner() \\
            .set("tone",  "dark") \\
            .set("protagonist", "mage") \\
            .generate(seed=7)
    """

    def __init__(self):
        self._domain   = JRPGDomain()
        self._designer = Designer(domain=self._domain)
        # Provide defaults for required fields so generate() always works
        self._designer.set("genre",       "jrpg")
        self._designer.set("tone",        "heroic")
        self._designer.set("protagonist", "warrior")
        self._designer.set("antagonist",  "demon_lord")

    def set(self, key: str, value) -> "GameDesigner":
        """Set a designer parameter. Returns self for chaining."""
        self._designer.set(key, value)
        return self

    def generate(self, seed: int | None = None) -> World:
        """Generate and return the game world."""
        if seed is not None:
            self._designer.set_seed(seed)
        return self._designer.generate()

    def run(self, seed: int | None = None,
            asset_paths: dict | None = None) -> None:
        """Generate the world and immediately run the game (requires pygame)."""
        world = self.generate(seed)
        GameEngine(world, asset_paths=asset_paths).run()

    def options(self) -> str:
        """Return a string listing all available parameters and their choices."""
        lines = ["GameDesigner parameters:\n"]
        for key, rule in self._domain.schema._rules.items():
            req  = "REQUIRED" if rule.required else f"default={rule.default!r}"
            line = f"  {key:<22} ({rule.type.__name__}, {req})"
            if rule.choices:
                line += f"\n    choices : {[str(c) for c in rule.choices]}"
            if rule.range:
                line += f"\n    range   : {rule.range}"
            if rule.description:
                line += f"\n    about   : {rule.description}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def presets() -> dict[str, dict]:
        """
        Named configuration presets.

        Usage::

            for k, v in GameDesigner.presets()["dark_fantasy"].items():
                designer.set(k, v)
        """
        return {
            "classic_adventure": {
                "genre": "jrpg", "tone": "heroic",
                "protagonist": "warrior", "antagonist": "demon_lord",
                "world_size": "medium", "difficulty": "normal", "chapters": 3,
            },
            "dark_fantasy": {
                "genre": "jrpg", "tone": "dark",
                "protagonist": "outcast", "antagonist": "ancient_evil",
                "world_size": "large", "difficulty": "hard", "chapters": 4,
            },
            "comedic_quest": {
                "genre": "jrpg", "tone": "comedic",
                "protagonist": "rogue", "antagonist": "empire",
                "world_size": "small", "difficulty": "easy", "chapters": 2,
            },
            "arcane_mystery": {
                "genre": "jrpg", "tone": "mysterious",
                "protagonist": "scholar", "antagonist": "corruption",
                "world_size": "medium", "difficulty": "normal", "chapters": 3,
            },
            "tragic_hero": {
                "genre": "jrpg", "tone": "tragic",
                "protagonist": "warrior", "antagonist": "rival",
                "world_size": "large", "difficulty": "hard", "chapters": 5,
            },
            "mage_adventure": {
                "genre": "jrpg", "tone": "heroic",
                "protagonist": "mage", "antagonist": "ancient_evil",
                "world_size": "medium", "difficulty": "normal", "chapters": 3,
            },
            "quick_dungeon": {
                "genre": "jrpg", "tone": "heroic",
                "protagonist": "warrior", "antagonist": "demon_lord",
                "world_size": "tiny", "difficulty": "easy", "chapters": 1,
            },
            "epic_saga": {
                "genre": "jrpg", "tone": "heroic",
                "protagonist": "warrior", "antagonist": "demon_lord",
                "world_size": "epic", "difficulty": "expert", "chapters": 6,
            },
        }


__all__ = ["GameDesigner", "GameEngine", "World"]
