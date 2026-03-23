"""
ConstraintSet and Rule: declarative validation for Intent values.

Domains use these to catch invalid designer input early, before any
generation happens.

Example::

    schema = ConstraintSet([
        Rule("energy",     type=float, range=(0.0, 1.0), required=True),
        Rule("mood",       type=str,   choices=["dark","neutral","bright"]),
        Rule("num_rooms",  type=int,   range=(1, 50), default=10),
    ])
    schema.validate(intent)   # raises ValueError on bad input
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Rule:
    """
    Validation rule for a single intent key.

    Attributes:
        key         — the intent key this rule applies to
        type        — expected Python type (str, int, float, bool, list)
        required    — if True, the key must be present in the intent
        default     — used to fill in missing optional keys
        choices     — if set, value must be one of these (for str/int)
        range       — (min, max) inclusive, for numeric types
        description — human-readable description for UI / --help
    """
    key: str
    type: type = str
    required: bool = False
    default: Any = None
    choices: list | None = None
    range: tuple | None = None
    description: str = ""

    def validate(self, value: Any) -> Any:
        """Validate and coerce a single value. Returns cleaned value."""
        # type check
        if not isinstance(value, self.type):
            try:
                value = self.type(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Intent key '{self.key}': expected {self.type.__name__}, "
                    f"got {type(value).__name__} ({value!r})"
                )

        # range check
        if self.range is not None and isinstance(value, (int, float)):
            lo, hi = self.range
            if not (lo <= value <= hi):
                raise ValueError(
                    f"Intent key '{self.key}': value {value} is outside "
                    f"allowed range [{lo}, {hi}]"
                )

        # choices check
        if self.choices is not None:
            if value not in self.choices:
                raise ValueError(
                    f"Intent key '{self.key}': value {value!r} is not one of "
                    f"{self.choices}"
                )

        return value


class ConstraintSet:
    """
    A collection of Rules that validates a complete Intent.

    Usage::

        schema = ConstraintSet([Rule("energy", type=float, required=True)])
        schema.validate(intent)   # fills defaults, raises ValueError on violations
    """

    def __init__(self, rules: list[Rule]):
        self._rules: dict[str, Rule] = {r.key: r for r in rules}

    def validate(self, intent: "Intent") -> None:
        """
        Validate the intent in place.
        - Fills missing optional keys with their defaults.
        - Raises ValueError for any violation.
        """
        for key, rule in self._rules.items():
            value = intent.get(key)
            if value is None:
                if rule.required:
                    raise ValueError(
                        f"Intent is missing required key '{key}'. "
                        f"{rule.description}"
                    )
                if rule.default is not None:
                    intent.set(key, rule.default, description=rule.description)
            else:
                cleaned = rule.validate(value)
                intent.set(key, cleaned)

    def describe(self) -> str:
        """Return a human-readable description of all rules."""
        lines = []
        for key, rule in self._rules.items():
            req = "required" if rule.required else f"default={rule.default!r}"
            extras = []
            if rule.choices:
                extras.append(f"choices={rule.choices}")
            if rule.range:
                extras.append(f"range={rule.range}")
            extra_str = ", ".join(extras)
            line = f"  {key} ({rule.type.__name__}, {req})"
            if extra_str:
                line += f"  [{extra_str}]"
            if rule.description:
                line += f"  — {rule.description}"
            lines.append(line)
        return "ConstraintSet:\n" + "\n".join(lines)
