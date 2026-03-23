"""
NarrativeDomain: Procedural story / quest generation.

Uses Markov chains for beat sequencing and grammars for prose.

Designer knobs:
    genre       — str   : "fantasy" | "horror" | "mystery" | "scifi" | "romance"
    tone        — str   : "dark" | "hopeful" | "comedic" | "tragic"
    length      — str   : "short" | "medium" | "long"
    protagonist — str   : "hero" | "antihero" | "reluctant" | "chosen"
    stakes      — float : 0.0 (personal) → 1.0 (world-ending)

Produces a NarrativeOutline with acts, beats, and scene descriptions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from procgen.domains.base import DomainBase
from procgen.core import Intent, GenerationPlan
from procgen.generators import GeneratorBase
from procgen.grammar import Grammar, ITEM_NAME_GRAMMAR
from procgen.random import MarkovChain, WeightedTable


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class Beat:
    name: str
    description: str
    tension: float      # 0.0–1.0


@dataclass
class Act:
    number: int
    name: str
    beats: list[Beat] = field(default_factory=list)


@dataclass
class NarrativeOutline:
    title: str
    genre: str
    tone: str
    protagonist_archetype: str
    acts: list[Act] = field(default_factory=list)
    MacGuffin: str = ""     # the object/goal driving the story
    antagonist: str = ""
    metadata: dict = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            f"📖 Narrative Outline",
            f"   Title      : {self.title}",
            f"   Genre      : {self.genre}  |  Tone: {self.tone}",
            f"   Protagonist: {self.protagonist_archetype}",
            f"   MacGuffin  : {self.MacGuffin}",
            f"   Antagonist : {self.antagonist}",
            "",
        ]
        for act in self.acts:
            lines.append(f"   ── Act {act.number}: {act.name} ──")
            for beat in act.beats:
                bar = "█" * int(beat.tension * 10)
                lines.append(f"      [{bar:<10}] {beat.name}")
                lines.append(f"               {beat.description}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grammars
# ---------------------------------------------------------------------------

_TITLE_GRAMMAR = Grammar({
    "START":     [("<DET> <ADJ> <NOUN>", 2), ("The <NOUN> of <PLACE>", 2),
                  ("<ADJ> <NOUN>", 1), ("<NOUN> and <NOUN>", 1)],
    "DET":       [("The", 3), ("A", 1)],
    "ADJ":       [("Last", 1), ("Broken", 1), ("Forgotten", 2), ("Rising", 1),
                  ("Lost", 1), ("Shattered", 1), ("Crimson", 1), ("Silent", 1)],
    "NOUN":      [("Kingdom", 1), ("Heir", 1), ("Prophecy", 1), ("Shadow", 2),
                  ("Star", 1), ("Covenant", 1), ("Reckoning", 1), ("Tide", 1)],
    "PLACE":     [("the Fallen", 1), ("Eternity", 1), ("the Deep", 1),
                  ("Ash", 1), ("the Abyss", 1)],
})

_BEAT_GRAMMAR: dict[str, Grammar] = {
    "ordinary_world":   Grammar({"S": [("The <PRO> lives a <ADJ> life, unaware of what is coming.", 1)],
                                  "PRO": [("protagonist",1)], "ADJ": [("quiet",1),("mundane",1),("desperate",1)]}),
    "call_to_adventure":Grammar({"S": [("A <CATALYST> shatters the ordinary world.", 1),
                                        ("The <PRO> receives a <CATALYST> that cannot be ignored.", 1)],
                                  "PRO": [("protagonist",1)], "CATALYST": [("message",1),("vision",1),("stranger",1),("loss",1)]}),
    "refusal":          Grammar({"S": [("The <PRO> hesitates, weighed down by <FEAR>.", 1)],
                                  "PRO": [("protagonist",1)], "FEAR": [("doubt",1),("loyalty",1),("fear of failure",1)]}),
    "crossing_threshold":Grammar({"S": [("The point of no return: the <PRO> commits.", 1)],
                                   "PRO": [("protagonist",1)]}),
    "tests_allies_enemies":Grammar({"S": [("New allies and enemies reveal themselves through a series of <TRIALS>.", 1)],
                                     "TRIALS": [("trials",1),("conflicts",1),("revelations",1)]}),
    "ordeal":           Grammar({"S": [("A devastating <CRISIS> forces the <PRO> to their lowest point.", 1)],
                                  "CRISIS": [("betrayal",1),("loss",1),("revelation",1)], "PRO": [("protagonist",1)]}),
    "reward":           Grammar({"S": [("The <PRO> seizes the <PRIZE>, transformed by the ordeal.", 1)],
                                  "PRO": [("protagonist",1)], "PRIZE": [("prize",1),("knowledge",1),("power",1)]}),
    "road_back":        Grammar({"S": [("The return home proves as dangerous as the journey out.", 1)]}),
    "resurrection":     Grammar({"S": [("A final confrontation with the <ANT> tests everything learned.", 1)],
                                  "ANT": [("antagonist",1),("shadow self",1),("greater evil",1)]}),
    "return_elixir":    Grammar({"S": [("The <PRO> returns changed, carrying wisdom (or scars) for the world.", 1)],
                                  "PRO": [("protagonist",1)]}),
}

_ANTAGONIST_GRAMMAR = Grammar({
    "START":   [("<PREFIX> <ROLE>", 1), ("The <ADJ> <ROLE>", 2)],
    "PREFIX":  [("Lord", 1), ("The Fallen", 1), ("High", 1), ("Ancient", 1)],
    "ADJ":     [("Undying", 1), ("Shattered", 1), ("Hollow", 1), ("Wrathful", 1)],
    "ROLE":    [("King", 1), ("Oracle", 1), ("Architect", 1), ("Warden", 1), ("Voice", 1)],
})

# Hero's journey beat sequence (Markov)
_JOURNEY_CHAIN = MarkovChain({
    "ordinary_world":     [("call_to_adventure", 1.0)],
    "call_to_adventure":  [("refusal", 0.6), ("crossing_threshold", 0.4)],
    "refusal":            [("crossing_threshold", 1.0)],
    "crossing_threshold": [("tests_allies_enemies", 1.0)],
    "tests_allies_enemies":[("ordeal", 1.0)],
    "ordeal":             [("reward", 1.0)],
    "reward":             [("road_back", 1.0)],
    "road_back":          [("resurrection", 1.0)],
    "resurrection":       [("return_elixir", 1.0)],
    "return_elixir":      [("return_elixir", 1.0)],  # terminal
})

# Tension arc per beat
_BEAT_TENSION = {
    "ordinary_world": 0.1,
    "call_to_adventure": 0.25,
    "refusal": 0.2,
    "crossing_threshold": 0.4,
    "tests_allies_enemies": 0.55,
    "ordeal": 0.85,
    "reward": 0.6,
    "road_back": 0.7,
    "resurrection": 0.95,
    "return_elixir": 0.2,
}

_LENGTH_BEATS = {"short": 5, "medium": 7, "long": 10}


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class TitleGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> str:
        return _TITLE_GRAMMAR.generate(self.rng(), "START").title()


class MacGuffinGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> str:
        return ITEM_NAME_GRAMMAR.generate(self.rng(), "START").title()


class AntagonistGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> str:
        return _ANTAGONIST_GRAMMAR.generate(self.rng(), "START")


class BeatSequenceGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> list[Act]:
        rng = self.rng()
        length = params["length"]
        tone = params["tone"]
        num_beats = _LENGTH_BEATS.get(length, 7)

        # Walk the hero's journey chain
        beat_names = _JOURNEY_CHAIN.walk(rng, start="ordinary_world", steps=num_beats)

        # Tone modifier on tension
        tone_offset = {"dark": 0.1, "hopeful": -0.05, "comedic": -0.15, "tragic": 0.15}.get(tone, 0)

        beats = []
        for name in beat_names:
            grammar = _BEAT_GRAMMAR.get(name)
            if grammar:
                desc = grammar.generate(rng, "S")
            else:
                desc = f"The story moves through {name.replace('_', ' ')}."
            tension = min(1.0, max(0.0, _BEAT_TENSION.get(name, 0.5) + tone_offset))
            beats.append(Beat(
                name=name.replace("_", " ").title(),
                description=desc,
                tension=tension,
            ))

        # Split into 3 acts
        n = len(beats)
        splits = [n // 3, 2 * n // 3]
        act_names = ["The Ordinary World", "The Road of Trials", "The Return"]
        acts = []
        prev = 0
        for i, split in enumerate(splits + [n]):
            acts.append(Act(
                number=i + 1,
                name=act_names[i],
                beats=beats[prev:split],
            ))
            prev = split
        return acts


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class NarrativeDomain(DomainBase):
    def __init__(self):
        from procgen.constraints import ConstraintSet, Rule
        self.schema = ConstraintSet([
            Rule("genre",       type=str,   required=True,
                 choices=["fantasy","horror","mystery","scifi","romance"]),
            Rule("tone",        type=str,   default="hopeful",
                 choices=["dark","hopeful","comedic","tragic"]),
            Rule("length",      type=str,   default="medium",
                 choices=["short","medium","long"]),
            Rule("protagonist", type=str,   default="hero",
                 choices=["hero","antihero","reluctant","chosen"]),
            Rule("stakes",      type=float, default=0.5, range=(0.0, 1.0)),
        ])

    def build_plan(self, intent: Intent) -> GenerationPlan:
        from procgen.core import GenerationPlan
        plan = GenerationPlan()
        shared = dict(intent.items())
        plan.add_step("title",         shared)
        plan.add_step("macguffin",     shared)
        plan.add_step("antagonist",    shared)
        plan.add_step("beat_sequence", shared)
        return plan

    def assemble(self, outputs: dict[str, Any], plan: GenerationPlan) -> NarrativeOutline:
        intent = plan.metadata["intent_summary"]
        return NarrativeOutline(
            title                 = outputs["title"],
            genre                 = intent["genre"],
            tone                  = intent["tone"],
            protagonist_archetype = intent["protagonist"],
            acts                  = outputs["beat_sequence"],
            MacGuffin             = outputs["macguffin"],
            antagonist            = outputs["antagonist"],
            metadata              = {"intent": intent},
        )

    @property
    def generators(self):
        return {
            "title":         TitleGenerator,
            "macguffin":     MacGuffinGenerator,
            "antagonist":    AntagonistGenerator,
            "beat_sequence": BeatSequenceGenerator,
        }
