"""
engine.core.state — Runtime game state.

Separate from World (generated data), State is what changes during play:
player position, inventory, flags, quest progress, party stats, etc.

State is serializable for save/load.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .world import Stats, StatType


@dataclass
class CharacterState:
    """Runtime state of a character (player or party member)."""
    npc_id:     str
    name:       str
    class_id:   str
    level:      int   = 1
    exp:        int   = 0
    exp_to_next: int  = 100
    current_hp: int   = 10
    current_mp: int   = 5
    stats:      Stats = field(default_factory=Stats)
    skills:     list[str] = field(default_factory=list)
    equipment:  dict[str, str | None] = field(default_factory=lambda: {
        "weapon": None, "armor": None, "accessory": None
    })
    status_effects: list[str] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.current_hp > 0

    def heal(self, amount: int) -> int:
        healed = min(amount, self.stats.max_hp - self.current_hp)
        self.current_hp += healed
        return healed

    def take_damage(self, amount: int) -> int:
        damage = max(1, amount - self.stats.defense)
        self.current_hp = max(0, self.current_hp - damage)
        return damage

    def restore_mp(self, amount: int) -> int:
        restored = min(amount, self.stats.max_mp - self.current_mp)
        self.current_mp += restored
        return restored

    def use_mp(self, amount: int) -> bool:
        if self.current_mp >= amount:
            self.current_mp -= amount
            return True
        return False

    def gain_exp(self, amount: int) -> list[int]:
        """Gain exp, return list of levels gained."""
        self.exp += amount
        levels_gained = []
        while self.exp >= self.exp_to_next:
            self.exp -= self.exp_to_next
            self.level += 1
            levels_gained.append(self.level)
            self.exp_to_next = int(self.exp_to_next * 1.5)
        return levels_gained


@dataclass
class Inventory:
    """Player's item storage."""
    items:    dict[str, int] = field(default_factory=dict)   # item_id → count
    gold:     int            = 0
    capacity: int            = 99

    def add(self, item_id: str, count: int = 1) -> bool:
        self.items[item_id] = self.items.get(item_id, 0) + count
        return True

    def remove(self, item_id: str, count: int = 1) -> bool:
        if self.items.get(item_id, 0) >= count:
            self.items[item_id] -= count
            if self.items[item_id] <= 0:
                del self.items[item_id]
            return True
        return False

    def has(self, item_id: str, count: int = 1) -> bool:
        return self.items.get(item_id, 0) >= count

    def count(self, item_id: str) -> int:
        return self.items.get(item_id, 0)


@dataclass
class QuestState:
    """Tracks quest progress."""
    active:    set[str] = field(default_factory=set)    # active quest ids
    completed: set[str] = field(default_factory=set)    # completed quest ids
    failed:    set[str] = field(default_factory=set)
    progress:  dict[str, dict] = field(default_factory=dict)  # quest_id → {obj_id: count}

    def start(self, quest_id: str) -> None:
        self.active.add(quest_id)
        self.progress[quest_id] = {}

    def complete(self, quest_id: str) -> None:
        self.active.discard(quest_id)
        self.completed.add(quest_id)

    def advance(self, quest_id: str, objective_id: str, count: int = 1) -> None:
        if quest_id not in self.progress:
            self.progress[quest_id] = {}
        self.progress[quest_id][objective_id] = (
            self.progress[quest_id].get(objective_id, 0) + count
        )

    def objective_progress(self, quest_id: str, objective_id: str) -> int:
        return self.progress.get(quest_id, {}).get(objective_id, 0)


@dataclass
class PlayerPosition:
    area_id: str
    x:       int
    y:       int
    facing:  str = "down"  # "up", "down", "left", "right"


@dataclass
class GameFlags:
    """Boolean and integer flags for tracking story/world state."""
    flags:   dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any = True) -> None:
        self.flags[key] = value

    def get(self, key: str, default: Any = False) -> Any:
        return self.flags.get(key, default)

    def is_set(self, key: str) -> bool:
        return bool(self.flags.get(key, False))


@dataclass
class GameState:
    """
    Complete runtime state of a game session.
    This is what gets saved to disk.
    """
    # Player
    position:    PlayerPosition
    party:       list[CharacterState] = field(default_factory=list)
    inventory:   Inventory            = field(default_factory=Inventory)
    quests:      QuestState           = field(default_factory=QuestState)
    flags:       GameFlags            = field(default_factory=GameFlags)

    # Progress
    chapter:     int   = 1
    play_time:   float = 0.0    # seconds
    step_count:  int   = 0

    # Opened chests, triggered events (set of area:x:y strings)
    opened_chests: set[str] = field(default_factory=set)
    triggered:     set[str] = field(default_factory=set)

    # Battle
    encounter_steps: int = 0    # steps since last encounter

    @property
    def player(self) -> CharacterState | None:
        return self.party[0] if self.party else None

    @property
    def alive_party(self) -> list[CharacterState]:
        return [c for c in self.party if c.is_alive]

    @property
    def is_game_over(self) -> bool:
        return len(self.alive_party) == 0

    def chest_key(self, area_id: str, x: int, y: int) -> str:
        return f"{area_id}:{x}:{y}"

    def open_chest(self, area_id: str, x: int, y: int) -> bool:
        key = self.chest_key(area_id, x, y)
        if key not in self.opened_chests:
            self.opened_chests.add(key)
            return True
        return False

    def to_dict(self) -> dict:
        """Serialize for save file."""
        return {
            "position": {
                "area_id": self.position.area_id,
                "x": self.position.x,
                "y": self.position.y,
                "facing": self.position.facing,
            },
            "party": [{
                "npc_id":     c.npc_id,
                "name":       c.name,
                "class_id":   c.class_id,
                "level":      c.level,
                "exp":        c.exp,
                "exp_to_next": c.exp_to_next,
                "current_hp": c.current_hp,
                "current_mp": c.current_mp,
                "stats": c.stats.to_dict(),
                "skills":     c.skills,
                "equipment":  c.equipment,
                "status_effects": c.status_effects,
            } for c in self.party],
            "inventory":  {"items": self.inventory.items, "gold": self.inventory.gold},
            "quests":     {
                "active":    list(self.quests.active),
                "completed": list(self.quests.completed),
                "failed":    list(self.quests.failed),
                "progress":  self.quests.progress,
            },
            "flags":      self.flags.flags,
            "chapter":    self.chapter,
            "play_time":  self.play_time,
            "step_count": self.step_count,
            "opened_chests": list(self.opened_chests),
            "triggered":  list(self.triggered),
        }
