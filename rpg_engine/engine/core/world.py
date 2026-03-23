"""
gameengine.world — The generated game world data model.

This module defines every data structure that the procgen framework produces
and the game engine consumes. It is the contract between the two systems.

The engine never calls procgen directly — it only reads World objects.
Procgen never calls engine code — it only produces World objects.

Hierarchy:
    World
    ├── WorldMap          (tile grid, regions, connections between areas)
    ├── NPCRoster         (all NPCs: name, stats, location, dialogue, schedule)
    ├── QuestGraph        (quest objectives, dependencies, rewards)
    ├── ItemRegistry      (all items: weapons, armor, consumables)
    ├── EncounterTable    (enemy groups per zone, with probabilities)
    ├── ShopInventory     (which shops carry what)
    └── GameConfig        (title, genre, difficulty, starting conditions)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class TileType(Enum):
    GRASS      = "grass"
    WATER      = "water"
    MOUNTAIN   = "mountain"
    FOREST     = "forest"
    SAND       = "sand"
    SNOW       = "snow"
    LAVA       = "lava"
    DUNGEON    = "dungeon"
    ROAD       = "road"
    TOWN_FLOOR = "town_floor"
    WALL       = "wall"
    DOOR       = "door"
    CHEST      = "chest"
    SIGN       = "sign"
    WARP       = "warp"

    @property
    def walkable(self) -> bool:
        return self not in (TileType.WATER, TileType.MOUNTAIN, TileType.WALL)

    @property
    def is_encounter_zone(self) -> bool:
        return self in (TileType.GRASS, TileType.FOREST, TileType.DUNGEON,
                        TileType.SNOW, TileType.SAND)


class AreaType(Enum):
    WORLD_MAP  = "world_map"
    TOWN       = "town"
    DUNGEON    = "dungeon"
    OVERWORLD  = "overworld"
    SHOP       = "shop"
    HOUSE      = "house"
    CASTLE     = "castle"
    CAVE       = "cave"


class ItemType(Enum):
    WEAPON     = "weapon"
    ARMOR      = "armor"
    ACCESSORY  = "accessory"
    CONSUMABLE = "consumable"
    KEY_ITEM   = "key_item"
    MATERIAL   = "material"


class StatType(Enum):
    HP         = "hp"
    MP         = "mp"
    ATTACK     = "attack"
    DEFENSE    = "defense"
    MAGIC      = "magic"
    SPEED      = "speed"
    LUCK       = "luck"


class QuestStatus(Enum):
    LOCKED     = "locked"
    AVAILABLE  = "available"
    ACTIVE     = "active"
    COMPLETED  = "completed"
    FAILED     = "failed"


class DialogueNodeType(Enum):
    TEXT       = "text"       # display text, then continue
    CHOICE     = "choice"     # player picks from options
    CONDITION  = "condition"  # branch on game state
    GIVE_ITEM  = "give_item"  # NPC gives item
    TAKE_ITEM  = "take_item"  # NPC takes item
    SET_FLAG   = "set_flag"   # set a game flag
    START_BATTLE = "start_battle"
    WARP       = "warp"


# ─────────────────────────────────────────────────────────────────────────────
# Tile and Map
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tile:
    type:        TileType
    sprite_key:  str = ""          # e.g. "grass_01" — key into AssetManifest
    passable:    bool = True
    encounter:   bool = False
    warp_target: str | None = None # area_id:x:y if this is a warp tile
    object_key:  str | None = None # "chest_01", "sign_01", etc.
    metadata:    dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.sprite_key:
            self.sprite_key = self.type.value
        self.passable  = self.type.walkable
        self.encounter = self.type.is_encounter_zone


@dataclass
class Area:
    """
    A single map area (world map, town, dungeon floor, etc.).
    The tile grid is row-major: grid[y][x].
    """
    id:          str
    name:        str
    area_type:   AreaType
    width:       int
    height:      int
    grid:        list[list[Tile]]
    music_key:   str = ""          # key into AssetManifest
    ambience:    str = ""
    connections: list[str] = field(default_factory=list)  # area ids reachable from here
    npc_spawns:  list[dict] = field(default_factory=list)  # [{npc_id, x, y}]
    chest_spawns: list[dict] = field(default_factory=list) # [{item_id, x, y, opened}]
    metadata:    dict = field(default_factory=dict)

    def tile_at(self, x: int, y: int) -> Tile | None:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def find_warp(self, target: str) -> tuple[int, int] | None:
        """Find the tile that warps to target area_id."""
        for y in range(self.height):
            for x in range(self.width):
                t = self.grid[y][x]
                if t.warp_target and t.warp_target.startswith(target):
                    return (x, y)
        return None

    def walkable_tiles(self) -> list[tuple[int, int]]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.grid[y][x].passable]

    def center(self) -> tuple[int, int]:
        return (self.width // 2, self.height // 2)


@dataclass
class WorldMap:
    areas:       dict[str, Area]   # area_id → Area
    start_area:  str               # id of starting area
    start_pos:   tuple[int, int]   # (x, y) in start_area
    overworld_id: str = ""         # main overworld area id

    def area(self, area_id: str) -> Area | None:
        return self.areas.get(area_id)

    def all_areas(self) -> list[Area]:
        return list(self.areas.values())

    def areas_of_type(self, t: AreaType) -> list[Area]:
        return [a for a in self.areas.values() if a.area_type == t]


# ─────────────────────────────────────────────────────────────────────────────
# Characters and NPCs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Stats:
    hp:      int = 10
    mp:      int = 5
    attack:  int = 5
    defense: int = 5
    magic:   int = 5
    speed:   int = 5
    luck:    int = 5
    max_hp:  int = 10
    max_mp:  int = 5

    def to_dict(self) -> dict:
        return {s.value: getattr(self, s.value) for s in StatType}


@dataclass
class GrowthCurve:
    """How a character's stats grow per level."""
    hp_per_level:      float = 8.0
    mp_per_level:      float = 3.0
    attack_per_level:  float = 1.5
    defense_per_level: float = 1.2
    magic_per_level:   float = 1.0
    speed_per_level:   float = 0.8

    def stats_at_level(self, base: Stats, level: int) -> Stats:
        l = max(0, level - 1)
        return Stats(
            hp      = int(base.hp      + self.hp_per_level      * l),
            mp      = int(base.mp      + self.mp_per_level      * l),
            attack  = int(base.attack  + self.attack_per_level  * l),
            defense = int(base.defense + self.defense_per_level * l),
            magic   = int(base.magic   + self.magic_per_level   * l),
            speed   = int(base.speed   + self.speed_per_level   * l),
            luck    = base.luck,
            max_hp  = int(base.hp      + self.hp_per_level      * l),
            max_mp  = int(base.mp      + self.mp_per_level      * l),
        )


@dataclass
class Skill:
    id:          str
    name:        str
    description: str
    mp_cost:     int  = 0
    power:       int  = 10
    is_magic:    bool = False
    target:      str  = "enemy"   # "enemy", "ally", "all_enemies", "all_allies", "self"
    element:     str  = "none"    # "fire", "ice", "lightning", "holy", "dark", "none"
    learn_level: int  = 1
    sprite_key:  str  = ""
    effect:      str  = "damage"  # "damage", "heal", "status", "buff", "debuff"
    status_effect: str | None = None  # "poison", "sleep", "stun", etc.


@dataclass
class PlayerClass:
    id:          str
    name:        str
    description: str
    base_stats:  Stats
    growth:      GrowthCurve
    skills:      list[Skill] = field(default_factory=list)
    sprite_key:  str = ""
    portrait_key: str = ""


@dataclass
class NPC:
    id:          str
    name:        str
    sprite_key:  str
    portrait_key: str = ""
    area_id:     str  = ""
    spawn_x:     int  = 0
    spawn_y:     int  = 0
    dialogue_id: str  = ""       # root dialogue node id
    is_shopkeeper: bool = False
    shop_id:     str  = ""
    is_enemy:    bool = False
    stats:       Stats | None  = None
    skills:      list[str]     = field(default_factory=list)  # skill ids
    loot_table:  list[dict]    = field(default_factory=list)  # [{item_id, chance}]
    schedule:    list[dict]    = field(default_factory=list)  # [{time, area_id, x, y}]
    role:        str  = "villager" # "villager", "guard", "merchant", "boss", "party_member"
    metadata:    dict = field(default_factory=dict)


@dataclass
class NPCRoster:
    npcs:          dict[str, NPC]           # npc_id → NPC
    player_classes: dict[str, PlayerClass]  # class_id → PlayerClass
    party_members: list[str]               # npc_ids that can join the party

    def npc(self, npc_id: str) -> NPC | None:
        return self.npcs.get(npc_id)

    def enemies(self) -> list[NPC]:
        return [n for n in self.npcs.values() if n.is_enemy]

    def friendly_npcs(self) -> list[NPC]:
        return [n for n in self.npcs.values() if not n.is_enemy]


# ─────────────────────────────────────────────────────────────────────────────
# Dialogue
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DialogueChoice:
    text:    str
    next_id: str
    condition: str | None = None  # flag/item check expression


@dataclass
class DialogueNode:
    id:       str
    type:     DialogueNodeType
    speaker:  str = ""            # NPC name or "PLAYER"
    text:     str = ""
    next_id:  str | None = None   # auto-advance to this node
    choices:  list[DialogueChoice] = field(default_factory=list)
    # For condition nodes
    condition:       str | None = None
    true_branch_id:  str | None = None
    false_branch_id: str | None = None
    # For action nodes
    item_id:   str | None = None
    flag_key:  str | None = None
    flag_value: Any       = None
    warp_area: str | None = None
    warp_x:    int        = 0
    warp_y:    int        = 0


@dataclass
class DialogueTree:
    id:    str
    nodes: dict[str, DialogueNode]  # node_id → DialogueNode
    root:  str                       # starting node id

    def node(self, node_id: str) -> DialogueNode | None:
        return self.nodes.get(node_id)

    def root_node(self) -> DialogueNode | None:
        return self.nodes.get(self.root)


# ─────────────────────────────────────────────────────────────────────────────
# Items
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Item:
    id:          str
    name:        str
    description: str
    item_type:   ItemType
    price:       int  = 0
    sprite_key:  str  = ""
    # Weapon/armor stats
    stat_bonus:  dict[str, int] = field(default_factory=dict)  # {"attack": 5}
    # Consumable effect
    effect:      str  = ""       # "heal_hp", "heal_mp", "revive", "cure_status"
    effect_power: int = 0
    # Equip restrictions
    equip_classes: list[str] = field(default_factory=list)  # [] = all classes
    stackable:   bool = True
    key_item:    bool = False
    metadata:    dict = field(default_factory=dict)


@dataclass
class ItemRegistry:
    items: dict[str, Item]   # item_id → Item

    def item(self, item_id: str) -> Item | None:
        return self.items.get(item_id)

    def weapons(self) -> list[Item]:
        return [i for i in self.items.values() if i.item_type == ItemType.WEAPON]

    def consumables(self) -> list[Item]:
        return [i for i in self.items.values() if i.item_type == ItemType.CONSUMABLE]

    def key_items(self) -> list[Item]:
        return [i for i in self.items.values() if i.key_item]


# ─────────────────────────────────────────────────────────────────────────────
# Quests
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestObjective:
    id:          str
    description: str
    type:        str   # "defeat_enemy", "collect_item", "reach_area",
                       # "talk_to_npc", "deliver_item", "escort_npc"
    target_id:   str   = ""   # enemy_id, item_id, area_id, npc_id
    target_count: int  = 1
    optional:    bool  = False


@dataclass
class Quest:
    id:          str
    name:        str
    description: str
    giver_npc:   str              # npc_id of quest giver
    objectives:  list[QuestObjective] = field(default_factory=list)
    prerequisites: list[str]     = field(default_factory=list)  # quest ids
    reward_gold:  int             = 0
    reward_exp:   int             = 0
    reward_items: list[str]       = field(default_factory=list)  # item ids
    reward_unlock: list[str]      = field(default_factory=list)  # quest ids to unlock
    is_main_quest: bool           = False
    chapter:      int             = 1
    metadata:     dict            = field(default_factory=dict)


@dataclass
class QuestGraph:
    quests:     dict[str, Quest]   # quest_id → Quest
    main_chain: list[str]          # ordered list of main quest ids

    def quest(self, quest_id: str) -> Quest | None:
        return self.quests.get(quest_id)

    def main_quests(self) -> list[Quest]:
        return [self.quests[qid] for qid in self.main_chain if qid in self.quests]

    def side_quests(self) -> list[Quest]:
        return [q for q in self.quests.values() if not q.is_main_quest]

    def available_quests(self, completed: set[str]) -> list[Quest]:
        result = []
        for q in self.quests.values():
            if all(p in completed for p in q.prerequisites):
                result.append(q)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Encounters and Shops
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnemyGroup:
    enemies:    list[str]   # list of npc_ids (can repeat for multiples)
    weight:     float = 1.0
    min_level:  int   = 1   # only spawns if player level >= min_level
    max_level:  int   = 99
    boss:       bool  = False


@dataclass
class EncounterTable:
    """Maps area_id (or area_type) to possible enemy groups."""
    tables: dict[str, list[EnemyGroup]]  # area_id → [EnemyGroup]

    def groups_for(self, area_id: str) -> list[EnemyGroup]:
        return self.tables.get(area_id, [])

    def random_encounter(self, area_id: str, rng) -> EnemyGroup | None:
        groups = self.groups_for(area_id)
        if not groups:
            return None
        weights = [g.weight for g in groups]
        total = sum(weights)
        r = rng.random() * total
        cumulative = 0.0
        for group, w in zip(groups, weights):
            cumulative += w
            if r <= cumulative:
                return group
        return groups[-1]


@dataclass
class ShopInventory:
    shops: dict[str, list[dict]]  # shop_id → [{item_id, stock, price_mult}]

    def shop(self, shop_id: str) -> list[dict]:
        return self.shops.get(shop_id, [])


# ─────────────────────────────────────────────────────────────────────────────
# Asset Manifest — what assets the engine needs to load
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssetManifest:
    """
    Lists every asset the engine needs, keyed by sprite_key/music_key.
    The user provides the actual files; the manifest tells the engine what to load.
    If a key has no path, the engine uses a procedurally-drawn placeholder.
    """
    sprites:   dict[str, str] = field(default_factory=dict)  # key → filepath
    music:     dict[str, str] = field(default_factory=dict)  # key → filepath
    sounds:    dict[str, str] = field(default_factory=dict)  # key → filepath
    fonts:     dict[str, str] = field(default_factory=dict)  # key → filepath
    portraits: dict[str, str] = field(default_factory=dict)  # key → filepath

    def sprite(self, key: str) -> str | None:
        return self.sprites.get(key)

    def music_path(self, key: str) -> str | None:
        return self.music.get(key)

    def all_required_sprites(self) -> list[str]:
        return list(self.sprites.keys())

    def missing_sprites(self) -> list[str]:
        return [k for k, v in self.sprites.items() if not v]


# ─────────────────────────────────────────────────────────────────────────────
# Game Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GameConfig:
    title:          str  = "Unnamed RPG"
    subtitle:       str  = ""
    genre:          str  = "jrpg"     # jrpg, action_rpg, dungeon_crawler, visual_novel
    tone:           str  = "heroic"   # heroic, dark, comedic, mysterious
    world_name:     str  = "Unknown Land"
    protagonist_name: str = "Hero"
    protagonist_class: str = "warrior"
    party_size:     int  = 4
    starting_gold:  int  = 100
    starting_level: int  = 1
    difficulty:     str  = "normal"   # easy, normal, hard, expert
    encounter_rate: float = 0.1       # probability per step in encounter zone
    tile_size:      int  = 32         # pixels
    screen_width:   int  = 640
    screen_height:  int  = 480
    target_fps:     int  = 60
    # Battle config
    battle_system:  str  = "turn_based"  # turn_based, active_time, action
    exp_curve:      str  = "standard"    # standard, fast, slow
    # Progression
    chapters:       int  = 3
    metadata:       dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# The complete generated World
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class World:
    """
    Everything the procgen framework produces.
    The engine reads this and runs the game.
    """
    config:         GameConfig
    world_map:      WorldMap
    npc_roster:     NPCRoster
    quest_graph:    QuestGraph
    item_registry:  ItemRegistry
    encounter_table: EncounterTable
    shop_inventory: ShopInventory
    dialogues:      dict[str, DialogueTree]   # dialogue_id → DialogueTree
    asset_manifest: AssetManifest
    seed:           int = 0
    metadata:       dict = field(default_factory=dict)

    def describe(self) -> str:
        wm    = self.world_map
        roster = self.npc_roster
        quests = self.quest_graph
        items  = self.item_registry
        lines  = [
            f"🌍  {self.config.title}",
            f"    Genre     : {self.config.genre}  |  Tone: {self.config.tone}",
            f"    World     : {self.config.world_name}",
            f"    Seed      : {self.seed}",
            f"",
            f"    Areas     : {len(wm.areas)}  "
                  f"(start: {wm.start_area} @ {wm.start_pos})",
        ]
        for area in wm.all_areas():
            npc_count = len(area.npc_spawns)
            lines.append(f"      [{area.area_type.value:<12}] {area.name:<22} "
                         f"{area.width:2d}×{area.height:2d}  npcs={npc_count}")
        lines += [
            f"",
            f"    NPCs      : {len(roster.npcs)} total  "
                  f"({len(roster.enemies())} enemies, "
                  f"{len(roster.friendly_npcs())} friendlies)",
            f"    Classes   : {list(roster.player_classes.keys())}",
            f"    Party     : {roster.party_members}",
            f"",
            f"    Quests    : {len(quests.quests)} total  "
                  f"({len(quests.main_quests())} main, "
                  f"{len(quests.side_quests())} side)",
            f"    Items     : {len(items.items)} total",
            f"",
            f"    Assets needed:",
            f"      sprites : {len(self.asset_manifest.sprites)}",
            f"      music   : {len(self.asset_manifest.music)}",
        ]
        missing = self.asset_manifest.missing_sprites()
        if missing:
            lines.append(f"      ⚠ missing : {len(missing)} sprites "
                         f"(placeholders will be used)")
        return "\n".join(lines)
