"""
DungeonDomain: Procedural dungeon/level generation for games.

Designer knobs:
    theme       — str   : "cave" | "ruin" | "dungeon" | "forest" | "ice"
    danger      — float : 0.0 (peaceful) → 1.0 (brutal)
    density     — str   : "sparse" | "normal" | "dense"
    num_rooms   — int   : 3–30
    has_boss    — bool  : whether to include a boss chamber
    purpose     — str   : "explore" | "survive" | "puzzle" | "stealth"

Produces a DungeonMap with rooms, corridors, enemies, and loot.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from .base import DomainBase
from ..core import Intent, GenerationPlan
from ..generators import GeneratorBase


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class Room:
    id: int
    name: str
    width: int
    height: int
    room_type: str      # "start", "boss", "treasure", "enemy", "puzzle", "corridor"
    enemies: list[str] = field(default_factory=list)
    loot: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    connections: list[int] = field(default_factory=list)  # room ids


@dataclass
class DungeonMap:
    theme: str
    rooms: list[Room]
    start_room: int
    boss_room: int | None
    exit_room: int
    difficulty: float
    metadata: dict = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            f"🗺  Dungeon Map ({self.theme.upper()})",
            f"   Rooms    : {len(self.rooms)}",
            f"   Difficulty: {self.difficulty:.2f}",
            f"   Start    : Room {self.start_room}",
            f"   Exit     : Room {self.exit_room}",
        ]
        if self.boss_room is not None:
            lines.append(f"   Boss Room: Room {self.boss_room}")
        lines.append("")
        for room in self.rooms:
            marker = ""
            if room.id == self.start_room: marker = " [START]"
            elif room.id == self.boss_room: marker = " [BOSS]"
            elif room.id == self.exit_room: marker = " [EXIT]"
            lines.append(f"   Room {room.id:02d}: {room.name} ({room.room_type}){marker}  {room.width}×{room.height}")
            if room.enemies:
                lines.append(f"            Enemies : {', '.join(room.enemies)}")
            if room.loot:
                lines.append(f"            Loot    : {', '.join(room.loot)}")
            if room.features:
                lines.append(f"            Features: {', '.join(room.features)}")
            lines.append(f"            Connects: {room.connections}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

_THEME_DATA = {
    "cave": {
        "room_names": ["Grotto", "Cavern", "Stalactite Hall", "Underground Pool",
                       "Crystal Chamber", "Fungal Den", "Bat Roost", "Chasm Bridge"],
        "enemies":    ["goblin", "cave troll", "bat swarm", "giant spider", "cave bear"],
        "loot":       ["gold nuggets", "raw gems", "ancient bones", "mushroom extract"],
        "features":   ["stalactites", "underground river", "glowing fungi", "collapsed ceiling"],
    },
    "dungeon": {
        "room_names": ["Torture Chamber", "Guard Post", "Armory", "Prison Cell",
                       "Throne Room", "Library", "Barracks", "Chapel"],
        "enemies":    ["skeleton", "zombie", "dark knight", "cultist", "wraith"],
        "loot":       ["gold coins", "magic scroll", "iron key", "potion", "cursed amulet"],
        "features":   ["iron portcullis", "arrow slit", "trapdoor", "altar", "rusty chains"],
    },
    "ruin": {
        "room_names": ["Crumbling Hall", "Overgrown Court", "Shattered Sanctum",
                       "Vine-covered Tower", "Mossy Antechamber", "Collapsed Nave"],
        "enemies":    ["animated statue", "shadow", "stone golem", "giant scorpion"],
        "loot":       ["ancient tome", "cracked pottery", "faded map", "enchanted shard"],
        "features":   ["crumbling pillars", "ancient mural", "overgrown altar", "sunken floor"],
    },
    "forest": {
        "room_names": ["Moonlit Clearing", "Gnarled Hollow", "Druid Circle",
                       "Poisonous Glade", "Bark Bridge", "Canopy Overlook"],
        "enemies":    ["wolf", "corrupted dryad", "giant wasp", "thorn elemental"],
        "loot":       ["rare herb", "feather token", "acorn of holding", "bark shield"],
        "features":   ["twisted roots", "natural spring", "fairy ring", "ancient tree"],
    },
    "ice": {
        "room_names": ["Frost Hall", "Frozen Lake", "Blizzard Pass", "Ice Spike Chamber",
                       "Glacier Vault", "Rime Corridor"],
        "enemies":    ["ice wraith", "frost troll", "yeti", "frozen revenant"],
        "loot":       ["frost crystal", "preserved relic", "ice gem", "cold-iron ingot"],
        "features":   ["frozen fountain", "ice mirror", "cracked glacier wall", "snowdrift"],
    },
}


class RoomLayoutGenerator(GeneratorBase):
    """Generates room sizes and connectivity graph."""

    def generate(self, params: dict, context: dict) -> list[dict]:
        rng = self.rng()
        num_rooms = params["num_rooms"]
        density = params["density"]

        size_ranges = {
            "sparse":  (8, 16),
            "normal":  (6, 20),
            "dense":   (4, 14),
        }
        lo, hi = size_ranges.get(density, (6, 18))

        rooms = []
        for i in range(num_rooms):
            rooms.append({
                "id": i,
                "width": rng.randint(lo, hi),
                "height": rng.randint(lo, hi),
                "connections": [],
            })

        # Build a spanning tree so all rooms are reachable
        shuffled = list(range(1, num_rooms))
        rng.shuffle(shuffled)
        for i, room_id in enumerate(shuffled):
            connect_to = rng.randint(0, room_id - 1)
            rooms[room_id]["connections"].append(connect_to)
            rooms[connect_to]["connections"].append(room_id)

        # Add extra connections based on density
        extra = {"sparse": 0, "normal": num_rooms // 4, "dense": num_rooms // 2}
        for _ in range(extra.get(density, 0)):
            a, b = rng.sample(range(num_rooms), 2)
            if b not in rooms[a]["connections"]:
                rooms[a]["connections"].append(b)
                rooms[b]["connections"].append(a)

        return rooms


class RoomPopulationGenerator(GeneratorBase):
    """Assigns names, types, enemies, loot, and features to each room."""

    def generate(self, params: dict, context: dict) -> list[Room]:
        rng = self.rng()
        layout = context.get("room_layout", [])
        theme = params["theme"]
        danger = params["danger"]
        has_boss = params["has_boss"]
        purpose = params["purpose"]
        num_rooms = len(layout)

        theme_data = _THEME_DATA.get(theme, _THEME_DATA["dungeon"])

        rooms: list[Room] = []
        num_rooms = len(layout)

        # Assign special room indices
        start_idx = 0
        boss_idx = num_rooms - 1 if has_boss else None
        exit_idx = num_rooms - 2 if has_boss else num_rooms - 1

        for raw in layout:
            i = raw["id"]

            # Room type
            if i == start_idx:
                rtype = "start"
            elif i == boss_idx:
                rtype = "boss"
            elif i == exit_idx:
                rtype = "exit"
            elif purpose == "puzzle" and rng.random() < 0.25:
                rtype = "puzzle"
            elif rng.random() < 0.15 * danger + 0.05:
                rtype = "treasure"
            else:
                rtype = "enemy"

            # Enemies
            enemies = []
            if rtype in ("enemy", "boss") and danger > 0.1:
                pool = theme_data["enemies"]
                count = 1 if rtype == "enemy" else 3
                count = max(1, int(count * (0.5 + danger)))
                if rtype == "boss":
                    enemies = [f"Elite {rng.choice(pool)}", f"minion {rng.choice(pool)}"]
                else:
                    enemies = [rng.choice(pool) for _ in range(min(count, 3))]

            # Loot
            loot = []
            if rtype in ("treasure", "boss") or (rng.random() < 0.3):
                loot_count = rng.randint(1, 3) if rtype == "treasure" else rng.randint(0, 2)
                loot = rng.sample(theme_data["loot"], min(loot_count, len(theme_data["loot"])))

            # Features
            feature_count = rng.randint(0, 2)
            features = rng.sample(theme_data["features"], min(feature_count, len(theme_data["features"])))

            # Name
            base_names = theme_data["room_names"]
            name = rng.choice(base_names)
            if rtype == "boss":
                name = f"Boss: {name}"
            elif rtype == "start":
                name = "Entrance"
            elif rtype == "exit":
                name = "Exit Chamber"

            rooms.append(Room(
                id=i,
                name=name,
                width=raw["width"],
                height=raw["height"],
                room_type=rtype,
                enemies=enemies,
                loot=loot,
                features=features,
                connections=raw["connections"],
            ))

        return rooms


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class DungeonDomain(DomainBase):
    """
    Procedural dungeon generation domain.

    Decision logic:
    - theme    → enemy/loot/feature pools, visual palette
    - danger   → enemy count/difficulty, trap density, loot quality
    - density  → room count, corridor branching
    - purpose  → room type distribution (puzzle, explore, survive, stealth)
    - has_boss → whether to include a climactic boss chamber
    """

    def __init__(self):
        from ..constraints import ConstraintSet, Rule
        self.schema = ConstraintSet([
            Rule("theme",     type=str,   required=True,
                 choices=["cave", "dungeon", "ruin", "forest", "ice"],
                 description="Visual and narrative theme"),
            Rule("danger",    type=float, required=True, range=(0.0, 1.0),
                 description="0=safe exploration, 1=brutal survival"),
            Rule("density",   type=str,   default="normal",
                 choices=["sparse", "normal", "dense"],
                 description="Room count and connectivity"),
            Rule("num_rooms", type=int,   default=10, range=(3, 30),
                 description="Number of rooms"),
            Rule("has_boss",  type=bool,  default=True,
                 description="Include a boss chamber at the end"),
            Rule("purpose",   type=str,   default="explore",
                 choices=["explore", "survive", "puzzle", "stealth"],
                 description="What the level is designed around"),
        ])

    def build_plan(self, intent: Intent) -> GenerationPlan:
        from ..core import GenerationPlan
        plan = GenerationPlan()

        shared = {
            "theme":     intent.get("theme"),
            "danger":    intent.get("danger"),
            "density":   intent.get("density"),
            "num_rooms": intent.get("num_rooms"),
            "has_boss":  intent.get("has_boss"),
            "purpose":   intent.get("purpose"),
        }

        plan.add_step("room_layout",     shared)
        plan.add_step("room_population", shared)

        return plan

    def assemble(self, outputs: dict[str, Any], plan: GenerationPlan) -> DungeonMap:
        rooms: list[Room] = outputs["room_population"]
        num_rooms = len(rooms)
        has_boss = plan.metadata["intent_summary"].get("has_boss", True)

        boss_room_id = rooms[-1].id if has_boss else None
        exit_room_id = rooms[-2].id if has_boss else rooms[-1].id

        return DungeonMap(
            theme      = plan.metadata["intent_summary"]["theme"],
            rooms      = rooms,
            start_room = rooms[0].id,
            boss_room  = boss_room_id,
            exit_room  = exit_room_id,
            difficulty = plan.metadata["intent_summary"]["danger"],
            metadata   = {"intent": plan.metadata.get("intent_summary", {})},
        )

    @property
    def generators(self):
        return {
            "room_layout":     RoomLayoutGenerator,
            "room_population": RoomPopulationGenerator,
        }
