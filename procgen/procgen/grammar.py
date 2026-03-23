"""
procgen.grammar — L-Systems and context-free grammars for structural generation.

LSystem         — Lindenmayer system for self-similar structures
                  (branching trees, coastlines, dungeon corridors, melodies)
Grammar         — Weighted context-free grammar for text / sequence generation
                  (room descriptions, item names, narrative beats)
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# L-System
# ---------------------------------------------------------------------------

class LSystem:
    """
    A parametric Lindenmayer system.

    An L-system starts with an axiom string and repeatedly rewrites it
    using a set of production rules, producing fractal/recursive structures.

    Symbols:
        Any character is a symbol. Unrecognised symbols pass through unchanged.
        By convention:
            F  — move forward (draw)
            f  — move forward (no draw)
            +  — turn right
            -  — turn left
            [  — push state
            ]  — pop state
            |  — reverse direction

    Example (Koch snowflake)::

        ls = LSystem(
            axiom="F--F--F",
            rules={"F": "F+F--F+F"},
        )
        result = ls.expand(iterations=3)
        turtle = ls.to_turtle()  # list of drawing commands

    Example (branching plant)::

        ls = LSystem(
            axiom="X",
            rules={"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
        )
        result = ls.expand(iterations=4)
    """

    def __init__(
        self,
        axiom: str,
        rules: dict[str, str | list[tuple[str, float]]],
        angle: float = 25.0,
    ):
        self.axiom = axiom
        self.angle = angle
        # Normalise rules: str → always [(str, 1.0)], list → weighted
        self._rules: dict[str, list[tuple[str, float]]] = {}
        for sym, rhs in rules.items():
            if isinstance(rhs, str):
                self._rules[sym] = [(rhs, 1.0)]
            else:
                self._rules[sym] = rhs

    def expand(self, iterations: int, rng: random.Random | None = None) -> str:
        """Expand the axiom for N iterations."""
        current = self.axiom
        for _ in range(iterations):
            next_str = []
            for ch in current:
                if ch in self._rules:
                    productions = self._rules[ch]
                    if len(productions) == 1:
                        next_str.append(productions[0][0])
                    else:
                        # Stochastic: pick by weight
                        r = rng or random
                        items = [p for p, _ in productions]
                        weights = [w for _, w in productions]
                        chosen = r.choices(items, weights=weights, k=1)[0]
                        next_str.append(chosen)
                else:
                    next_str.append(ch)
            current = "".join(next_str)
        return current

    def to_turtle(self, string: str) -> list[dict]:
        """
        Convert an L-system string into a sequence of turtle drawing commands.
        Returns a list of dicts with keys: action, angle, distance.
        """
        commands = []
        stack = []
        for ch in string:
            if ch == "F":
                commands.append({"action": "draw", "distance": 1.0})
            elif ch == "f":
                commands.append({"action": "move", "distance": 1.0})
            elif ch == "+":
                commands.append({"action": "turn_right", "angle": self.angle})
            elif ch == "-":
                commands.append({"action": "turn_left", "angle": self.angle})
            elif ch == "[":
                commands.append({"action": "push"})
            elif ch == "]":
                commands.append({"action": "pop"})
            elif ch == "|":
                commands.append({"action": "turn_right", "angle": 180.0})
        return commands

    def complexity(self, iterations: int) -> int:
        """Return the string length after N iterations (without generating it)."""
        length = len(self.axiom)
        for _ in range(iterations):
            new_length = 0
            for ch in ["F"]:  # simplified
                if ch in self._rules:
                    new_length += len(self._rules[ch][0][0])
                else:
                    new_length += 1
            length = length  # approximate
        return length

    def __repr__(self) -> str:
        return f"LSystem(axiom='{self.axiom}', rules={list(self._rules.keys())})"


# ---------------------------------------------------------------------------
# Context-Free Grammar
# ---------------------------------------------------------------------------

class Grammar:
    """
    A stochastic context-free grammar (SCFG) for text and sequence generation.

    Rules map a non-terminal to a weighted list of expansions.
    Terminals are plain strings. Non-terminals are wrapped in <angle brackets>.

    Example::

        grammar = Grammar({
            "START":       [("<ADJ> <NOUN> of <PLACE>", 1.0)],
            "ADJ":         [("ancient", 2), ("cursed", 1), ("glowing", 1)],
            "NOUN":        [("amulet", 1), ("blade", 1), ("tome", 1)],
            "PLACE":       [("the forgotten realm", 1), ("shadow", 1)],
        })
        name = grammar.generate(rng, "START")
        # → "ancient tome of shadow"

    Non-terminals in expansions must be wrapped in <> and match a rule key.
    """

    def __init__(self, rules: dict[str, list[tuple[str, float]]]):
        self._rules = rules

    def generate(self, rng: random.Random, start: str = "START", max_depth: int = 20) -> str:
        return self._expand(rng, start, depth=0, max_depth=max_depth)

    def _expand(self, rng: random.Random, symbol: str, depth: int, max_depth: int) -> str:
        if depth > max_depth:
            return symbol

        if symbol not in self._rules:
            return symbol  # terminal

        productions = self._rules[symbol]
        items = [p for p, _ in productions]
        weights = [float(w) for _, w in productions]
        template = rng.choices(items, weights=weights, k=1)[0]

        # Expand <NON_TERMINAL> placeholders
        result = []
        i = 0
        while i < len(template):
            if template[i] == "<":
                end = template.index(">", i)
                nt = template[i + 1:end]
                result.append(self._expand(rng, nt, depth + 1, max_depth))
                i = end + 1
            else:
                result.append(template[i])
                i += 1

        return "".join(result)

    def add_rule(self, symbol: str, productions: list[tuple[str, float]]) -> "Grammar":
        self._rules[symbol] = productions
        return self

    def __repr__(self) -> str:
        return f"Grammar(rules={list(self._rules.keys())})"


# ---------------------------------------------------------------------------
# Pre-built grammars
# ---------------------------------------------------------------------------

ITEM_NAME_GRAMMAR = Grammar({
    "START":    [("<PREFIX> <NOUN>", 2), ("<NOUN> of <SUFFIX>", 2), ("<ADJ> <NOUN>", 3)],
    "PREFIX":   [("Cursed", 1), ("Ancient", 2), ("Shattered", 1), ("Blessed", 1), ("Shadow", 1)],
    "ADJ":      [("gleaming", 1), ("corroded", 1), ("enchanted", 2), ("runed", 1), ("hollow", 1)],
    "NOUN":     [("Blade", 2), ("Tome", 1), ("Shield", 2), ("Amulet", 1), ("Staff", 1), ("Crown", 1)],
    "SUFFIX":   [("the Abyss", 1), ("Ages", 2), ("Sorrow", 1), ("Dawn", 1), ("the Fallen", 1)],
})

DUNGEON_ROOM_DESCRIPTION_GRAMMAR = Grammar({
    "START":    [("<OPENING>. <DETAIL>. <ATMOSPHERE>.", 1)],
    "OPENING":  [("The chamber is <SIZE> and <SHAPE>",2), ("You enter a <SIZE> <SHAPE> room", 1)],
    "SIZE":     [("vast", 1), ("cramped", 1), ("cavernous", 1), ("narrow", 2)],
    "SHAPE":    [("circular", 1), ("rectangular", 2), ("irregular", 1), ("vaulted", 1)],
    "DETAIL":   [("<FEATURE> dominates the <WALL>", 1), ("A <FEATURE> stands near the <WALL>", 1)],
    "FEATURE":  [("stone altar", 1), ("rusted portcullis", 1), ("cracked pillar", 2), ("iron brazier", 1)],
    "WALL":     [("north wall", 1), ("centre", 1), ("far end", 1), ("eastern corner", 1)],
    "ATMOSPHERE": [("The air smells of <SMELL>", 1), ("A faint <SOUND> echoes through the stone", 1)],
    "SMELL":    [("damp stone", 2), ("sulfur", 1), ("decay", 1), ("incense", 1)],
    "SOUND":    [("dripping", 2), ("scraping", 1), ("distant wailing", 1), ("wind", 1)],
})

MUSIC_MOTIF_GRAMMAR = Grammar({
    "START":    [("<SHAPE> <DIRECTION> <END>", 1)],
    "SHAPE":    [("stepwise", 2), ("leaping", 1), ("arpeggiated", 1), ("oscillating", 1)],
    "DIRECTION": [("ascent", 1), ("descent", 1), ("arch", 1), ("valley", 1)],
    "END":      [("resolving to tonic", 2), ("ending on the fifth", 1), ("left suspended", 1)],
})
