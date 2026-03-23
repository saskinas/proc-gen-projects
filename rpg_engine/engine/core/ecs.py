"""
engine.core.ecs — Entity Component System.

A lightweight ECS for managing game objects at runtime.
Entities are just integer IDs. Components are data. Systems process them.

This keeps the engine extensible — new game styles add new components
and systems without modifying existing code.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar, Iterator
import itertools

T = TypeVar("T")
_entity_counter = itertools.count(1)


# ─────────────────────────────────────────────────────────────────────────────
# Core ECS types
# ─────────────────────────────────────────────────────────────────────────────

def new_entity() -> int:
    return next(_entity_counter)


class World_ECS:
    """
    The ECS world: stores all entities and their components.
    """
    def __init__(self):
        # component_type → {entity_id → component}
        self._components: dict[type, dict[int, Any]] = {}
        self._entities:   set[int]                   = set()
        self._tags:       dict[str, set[int]]         = {}    # tag → entity_ids

    # ── Entity lifecycle ─────────────────────────────────────────────────────

    def create(self, *components, tags: list[str] = None) -> int:
        eid = new_entity()
        self._entities.add(eid)
        for comp in components:
            self.add_component(eid, comp)
        for tag in (tags or []):
            self._tags.setdefault(tag, set()).add(eid)
        return eid

    def destroy(self, entity_id: int) -> None:
        self._entities.discard(entity_id)
        for store in self._components.values():
            store.pop(entity_id, None)
        for tag_set in self._tags.values():
            tag_set.discard(entity_id)

    def alive(self, entity_id: int) -> bool:
        return entity_id in self._entities

    # ── Component operations ─────────────────────────────────────────────────

    def add_component(self, entity_id: int, component: Any) -> None:
        ct = type(component)
        if ct not in self._components:
            self._components[ct] = {}
        self._components[ct][entity_id] = component

    def remove_component(self, entity_id: int, component_type: type) -> None:
        self._components.get(component_type, {}).pop(entity_id, None)

    def get_component(self, entity_id: int, component_type: Type[T]) -> T | None:
        return self._components.get(component_type, {}).get(entity_id)

    def has_component(self, entity_id: int, component_type: type) -> bool:
        return entity_id in self._components.get(component_type, {})

    # ── Queries ──────────────────────────────────────────────────────────────

    def query(self, *component_types: type) -> Iterator[tuple[int, ...]]:
        """Yield (entity_id, comp1, comp2, ...) for all matching entities."""
        if not component_types:
            return
        stores = [self._components.get(ct, {}) for ct in component_types]
        # Intersect on the smallest store
        smallest = min(stores, key=len)
        for eid in list(smallest.keys()):
            comps = [s.get(eid) for s in stores]
            if all(c is not None for c in comps):
                yield (eid, *comps)

    def query_tag(self, tag: str) -> set[int]:
        return self._tags.get(tag, set())

    def first_tagged(self, tag: str) -> int | None:
        entities = self._tags.get(tag, set())
        return next(iter(entities), None) if entities else None

    def tag(self, entity_id: int, *tags: str) -> None:
        for t in tags:
            self._tags.setdefault(t, set()).add(entity_id)

    def untag(self, entity_id: int, *tags: str) -> None:
        for t in tags:
            self._tags.get(t, set()).discard(entity_id)

    def all_entities(self) -> set[int]:
        return set(self._entities)


# ─────────────────────────────────────────────────────────────────────────────
# Built-in Components
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    x:    float
    y:    float
    area: str = ""

@dataclass
class Velocity:
    dx: float = 0.0
    dy: float = 0.0

@dataclass
class Facing:
    direction: str = "down"   # "up", "down", "left", "right"

@dataclass
class Sprite:
    key:          str         # key in AssetManifest
    frame:        int  = 0
    anim:         str  = "idle"
    flip_x:       bool = False
    visible:      bool = True
    z_order:      int  = 0    # higher = drawn on top
    offset_x:     int  = 0
    offset_y:     int  = 0

@dataclass
class Collider:
    width:   int
    height:  int
    offset_x: int = 0
    offset_y: int = 0
    solid:   bool = True

@dataclass
class Character:
    """Links an entity to the world data model."""
    npc_id:     str
    name:       str
    is_player:  bool = False
    is_party:   bool = False

@dataclass
class DialogueTrigger:
    dialogue_id: str
    npc_id:      str
    active:      bool = True

@dataclass
class Interactable:
    """Entity can be interacted with (NPCs, chests, signs, doors)."""
    type:        str     # "npc", "chest", "sign", "door", "warp"
    payload:     dict = field(default_factory=dict)

@dataclass
class Walker:
    """NPC that patrols a path."""
    path:       list[tuple[int, int]] = field(default_factory=list)
    path_idx:   int   = 0
    wait_timer: float = 0.0
    speed:      float = 2.0
    loop:       bool  = True

@dataclass
class Camera:
    target_entity: int | None = None
    offset_x:      float = 0.0
    offset_y:      float = 0.0
    zoom:          float = 1.0
    smooth:        float = 8.0   # lerp speed
