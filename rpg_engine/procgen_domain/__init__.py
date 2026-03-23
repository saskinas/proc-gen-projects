"""
gameengine.procgen_domain — The procgen framework domain for game world generation.

Designer intent → GenerationPlan → Generators → World

High-level knobs the designer sets:
    genre          "jrpg" | "action_rpg" | "dungeon_crawler" | "visual_novel"
    tone           "heroic" | "dark" | "comedic" | "mysterious" | "tragic"
    world_size     "tiny" | "small" | "medium" | "large" | "epic"
    difficulty     "easy" | "normal" | "hard" | "expert"
    protagonist    "warrior" | "mage" | "rogue" | "scholar" | "outcast"
    antagonist     "empire" | "demon_lord" | "corruption" | "ancient_evil" | "rival"
    world_name     str (optional)
    title          str (optional)
    chapters       int (1-6)
    party_size     int (1-4)

The procgen framework orchestrates these generators in order:
    1. config_gen     → GameConfig
    2. classes_gen    → player classes + enemy types
    3. items_gen      → item registry
    4. skills_gen     → skill definitions
    5. world_map_gen  → area layout + tile grids
    6. npc_gen        → NPC roster + enemy stats
    7. dialogue_gen   → dialogue trees for all NPCs
    8. quest_gen      → quest graph
    9. encounter_gen  → encounter tables
    10. shop_gen      → shop inventories
    11. asset_gen     → asset manifest (lists what sprites/music are needed)
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Any

from procgen.core import Intent, GenerationPlan
from procgen.domains.base import DomainBase
from procgen.generators import GeneratorBase
from procgen.constraints import ConstraintSet, Rule
from procgen.seeds import Seed

from engine.core.world import (
    World, WorldMap, Area, Tile, TileType, AreaType,
    NPC, NPCRoster, PlayerClass, Stats, GrowthCurve, Skill,
    Quest, QuestObjective, QuestGraph,
    Item, ItemType, ItemRegistry,
    EnemyGroup, EncounterTable,
    ShopInventory,
    DialogueTree, DialogueNode, DialogueNodeType, DialogueChoice,
    AssetManifest, GameConfig,
)

from procgen.grammar import Grammar


# ─────────────────────────────────────────────────────────────────────────────
# Content tables (genre/tone → content biases)
# ─────────────────────────────────────────────────────────────────────────────

_PROTAGONIST_CLASSES = {
    "warrior": {
        "name": "Warrior", "description": "A seasoned fighter with high HP and attack.",
        "base": Stats(hp=35, mp=8, attack=12, defense=10, magic=4, speed=7, luck=5,
                      max_hp=35, max_mp=8),
        "growth": GrowthCurve(hp_per_level=12, mp_per_level=2, attack_per_level=2.5,
                               defense_per_level=2.0, magic_per_level=0.5, speed_per_level=1.0),
    },
    "mage": {
        "name": "Mage", "description": "A powerful spellcaster with high magic but low defense.",
        "base": Stats(hp=20, mp=30, attack=4, defense=4, magic=14, speed=8, luck=7,
                      max_hp=20, max_mp=30),
        "growth": GrowthCurve(hp_per_level=6, mp_per_level=8, attack_per_level=0.8,
                               defense_per_level=0.6, magic_per_level=3.0, speed_per_level=1.2),
    },
    "rogue": {
        "name": "Rogue", "description": "Quick and cunning, excels at critical hits.",
        "base": Stats(hp=25, mp=12, attack=10, defense=6, magic=6, speed=14, luck=12,
                      max_hp=25, max_mp=12),
        "growth": GrowthCurve(hp_per_level=8, mp_per_level=3, attack_per_level=2.0,
                               defense_per_level=1.0, magic_per_level=1.0, speed_per_level=2.5),
    },
    "scholar": {
        "name": "Scholar", "description": "Balanced stats with unique support abilities.",
        "base": Stats(hp=22, mp=22, attack=7, defense=7, magic=10, speed=9, luck=9,
                      max_hp=22, max_mp=22),
        "growth": GrowthCurve(hp_per_level=8, mp_per_level=6, attack_per_level=1.5,
                               defense_per_level=1.5, magic_per_level=2.0, speed_per_level=1.5),
    },
    "outcast": {
        "name": "Outcast", "description": "Mysterious past. High luck, unpredictable power.",
        "base": Stats(hp=28, mp=18, attack=9, defense=7, magic=9, speed=10, luck=15,
                      max_hp=28, max_mp=18),
        "growth": GrowthCurve(hp_per_level=9, mp_per_level=5, attack_per_level=1.8,
                               defense_per_level=1.4, magic_per_level=1.8, speed_per_level=1.8),
    },
}

_TONE_AREA_NAMES = {
    "heroic":     ["Brightwater Village", "Sunridge Keep", "Golden Plains",
                   "Crystal Cavern", "Dragon's Peak", "The Ancient Shrine"],
    "dark":       ["Ashenveil", "The Forsaken Mire", "Shadowkeep",
                   "Blighted Hollow", "The Crimson Wastes", "Tomb of the Fallen"],
    "comedic":    ["Bumbletown", "Cheese Village", "The Slightly Dangerous Dungeon",
                   "Goblin Giggles Inn", "Mount Whoops", "The Confused Forest"],
    "mysterious": ["The Nameless Ruin", "Mistwhisper", "The Sunken Library",
                   "Veilgate", "The Dreaming Stones", "Edge of the Known"],
    "tragic":     ["Greymoor", "The Lost Capital", "Widow's Crossing",
                   "The Last Bastion", "Ashen Spire", "The Final Gate"],
}

_TONE_NPC_NAMES = {
    "heroic":     ["Aldric", "Sera", "Tomas", "Lyria", "Brand", "Nessa", "Gareth"],
    "dark":       ["Morrhen", "Vesper", "Carrow", "Silke", "Drath", "Neva", "Crull"],
    "comedic":    ["Bumble", "Wiggins", "Toot", "Flopsie", "Gerald", "Marge", "Dorf"],
    "mysterious": ["Auren", "Thessaly", "Ziven", "Mira", "Solan", "Elyse", "Vorath"],
    "tragic":     ["Edmund", "Lysa", "Rowan", "Isolde", "Cael", "Briar", "Theron"],
}

_TONE_ENEMY_NAMES = {
    "heroic":     ["Goblin", "Orc Warrior", "Giant Spider", "Stone Golem", "Dark Knight"],
    "dark":       ["Revenant", "Plague Rat", "Bone Archer", "Shadow Fiend", "Wight"],
    "comedic":    ["Confused Slime", "Grumpy Mushroom", "Tiny Dragon", "Angry Toad"],
    "mysterious": ["Void Stalker", "Echo Beast", "Dream Weaver", "Null Entity"],
    "tragic":     ["Lost Soul", "Corrupted Guard", "Hollow Knight", "Grief Specter"],
}

_GENRE_SKILLS = {
    "jrpg": [
        Skill("attack",  "Attack",   "A basic physical attack.", 0,  10, False, "enemy",   "none", 1, effect="damage"),
        Skill("fire",    "Fire",     "Burn one enemy.",          5,  25, True,  "enemy",   "fire", 3, effect="damage"),
        Skill("heal",    "Heal",     "Restore HP to one ally.",  6,  30, True,  "ally",    "none", 2, effect="heal"),
        Skill("thunder", "Thunder",  "Lightning hits all foes.", 10, 20, True,  "all_enemies","lightning",6, effect="damage"),
        Skill("blizzard","Blizzard", "Ice storm.",               8,  28, True,  "enemy",   "ice",  5, effect="damage"),
        Skill("cure",    "Cure",     "Remove one status effect.",4,   0, True,  "ally",    "none", 4, effect="status"),
        Skill("slash",   "Slash",    "Powerful sword strike.",   0,  20, False, "enemy",   "none", 4, effect="damage"),
        Skill("dark",    "Dark",     "Dark energy bolt.",        7,  22, True,  "enemy",   "dark", 5, effect="damage"),
    ],
}

_ITEM_TEMPLATES = [
    # Consumables
    {"id":"potion",     "name":"Potion",      "type":ItemType.CONSUMABLE, "price":50,
     "desc":"Restores 50 HP.", "effect":"heal_hp", "power":50},
    {"id":"hi_potion",  "name":"Hi-Potion",   "type":ItemType.CONSUMABLE, "price":150,
     "desc":"Restores 150 HP.", "effect":"heal_hp", "power":150},
    {"id":"ether",      "name":"Ether",       "type":ItemType.CONSUMABLE, "price":80,
     "desc":"Restores 30 MP.", "effect":"heal_mp", "power":30},
    {"id":"elixir",     "name":"Elixir",      "type":ItemType.CONSUMABLE, "price":500,
     "desc":"Fully restores HP and MP.", "effect":"full_restore", "power":999},
    {"id":"antidote",   "name":"Antidote",    "type":ItemType.CONSUMABLE, "price":30,
     "desc":"Cures poison.", "effect":"cure_status", "power":0},
    {"id":"revive",     "name":"Phoenix Down", "type":ItemType.CONSUMABLE,"price":200,
     "desc":"Revives a fallen ally with 25% HP.", "effect":"revive", "power":25},
    # Weapons
    {"id":"sword_01",   "name":"Iron Sword",  "type":ItemType.WEAPON, "price":200,
     "desc":"A reliable iron blade.", "stat_bonus":{"attack":8}},
    {"id":"staff_01",   "name":"Oak Staff",   "type":ItemType.WEAPON, "price":180,
     "desc":"Channels magical energy.", "stat_bonus":{"magic":8, "attack":3}},
    {"id":"dagger_01",  "name":"Short Dagger","type":ItemType.WEAPON, "price":160,
     "desc":"Fast and precise.", "stat_bonus":{"attack":5, "speed":4}},
    {"id":"sword_02",   "name":"Steel Sword", "type":ItemType.WEAPON, "price":500,
     "desc":"A fine steel blade.", "stat_bonus":{"attack":15}},
    # Armor
    {"id":"armor_01",   "name":"Leather Armor","type":ItemType.ARMOR, "price":150,
     "desc":"Basic leather protection.", "stat_bonus":{"defense":6}},
    {"id":"armor_02",   "name":"Chain Mail",  "type":ItemType.ARMOR, "price":400,
     "desc":"Interlocking rings of steel.", "stat_bonus":{"defense":12}},
    {"id":"robe_01",    "name":"Mage Robe",   "type":ItemType.ARMOR, "price":200,
     "desc":"Enhances magical power.", "stat_bonus":{"magic":5, "defense":3}},
    # Accessories
    {"id":"ring_01",    "name":"Speed Ring",  "type":ItemType.ACCESSORY, "price":300,
     "desc":"Increases agility.", "stat_bonus":{"speed":8}},
    {"id":"amulet_01",  "name":"Luck Charm",  "type":ItemType.ACCESSORY, "price":250,
     "desc":"Brings good fortune.", "stat_bonus":{"luck":10}},
]

_ANTAGONIST_NAMES = {
    "empire":       ("The Emperor Vexos", "empire_boss"),
    "demon_lord":   ("Lord Malachar",     "demon_boss"),
    "corruption":   ("The Blight",        "corruption_boss"),
    "ancient_evil": ("The Sealed Horror", "ancient_boss"),
    "rival":        ("Kiran, the Fallen", "rival_boss"),
}

_WORLD_NAME_GRAMMAR = Grammar({
    "START": [("<ADJ> <NOUN>", 2), ("The Land of <NOUN>", 1), ("<NOUN>", 1)],
    "ADJ":   [("Verdant",1),("Shadow",1),("Golden",1),("Ancient",1),("Forsaken",1),("Crimson",1)],
    "NOUN":  [("Aleria",1),("Valdris",1),("Thornmoor",1),("Arcanthos",1),("Serenfall",1),("Grimwood",1)],
})


# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────

class ConfigGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> GameConfig:
        rng   = self.rng()
        tone  = params["tone"]
        genre = params["genre"]
        world_name = params.get("world_name") or _WORLD_NAME_GRAMMAR.generate(rng)
        title = params.get("title") or f"Tales of {world_name}"
        return GameConfig(
            title           = title,
            genre           = genre,
            tone            = tone,
            world_name      = world_name,
            protagonist_name = params.get("protagonist_name", "Hero"),
            protagonist_class = params["protagonist"],
            party_size      = params["party_size"],
            starting_gold   = {"easy":200,"normal":100,"hard":50,"expert":30}[params["difficulty"]],
            difficulty      = params["difficulty"],
            chapters        = params["chapters"],
            tile_size       = 32,
            screen_width    = 640,
            screen_height   = 480,
            encounter_rate  = {"easy":0.05,"normal":0.10,"hard":0.15,"expert":0.20}[params["difficulty"]],
        )


class ClassesGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> dict[str, PlayerClass]:
        rng   = self.rng()
        tone  = params["tone"]
        genre = params["genre"]
        protagonist = params["protagonist"]
        skills_pool = _GENRE_SKILLS.get(genre, _GENRE_SKILLS["jrpg"])

        classes = {}
        for class_id, data in _PROTAGONIST_CLASSES.items():
            # Assign skills based on class
            class_skills = []
            for skill in skills_pool:
                if class_id == "warrior" and not skill.is_magic:
                    class_skills.append(skill)
                elif class_id == "mage" and skill.is_magic:
                    class_skills.append(skill)
                elif class_id in ("rogue", "outcast", "scholar"):
                    class_skills.append(skill)

            classes[class_id] = PlayerClass(
                id          = class_id,
                name        = data["name"],
                description = data["description"],
                base_stats  = data["base"],
                growth      = data["growth"],
                skills      = class_skills,
                sprite_key  = f"player_{class_id}",
                portrait_key = f"portrait_{class_id}",
            )
        return classes


class ItemsGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> ItemRegistry:
        rng = self.rng()
        items = {}
        for tmpl in _ITEM_TEMPLATES:
            item = Item(
                id          = tmpl["id"],
                name        = tmpl["name"],
                description = tmpl.get("desc", ""),
                item_type   = tmpl["type"],
                price       = tmpl.get("price", 50),
                sprite_key  = f"item_{tmpl['id']}",
                stat_bonus  = tmpl.get("stat_bonus", {}),
                effect      = tmpl.get("effect", ""),
                effect_power = tmpl.get("power", 0),
                key_item    = tmpl.get("key_item", False),
            )
            items[item.id] = item
        # Add key items based on quest chain
        for i in range(1, params.get("chapters", 3) + 1):
            kid = f"key_orb_{i}"
            items[kid] = Item(
                id         = kid,
                name       = f"Sacred Orb {i}",
                description = f"A mysterious orb needed to unseal the path to chapter {i}.",
                item_type  = ItemType.KEY_ITEM,
                price      = 0,
                sprite_key = f"item_{kid}",
                key_item   = True,
            )
        return ItemRegistry(items=items)


class WorldMapGenerator(GeneratorBase):
    """
    Generates all Area objects for the game world:
      - One overworld (large grass/terrain map with warp tiles to towns/dungeons)
      - N towns (interior maps with buildings, NPCs, shops)
      - N dungeons (maze-like maps with encounters and chests)

    All map generation is strictly bounded — no while-loops that depend on
    random convergence. All iteration is for-loop bounded.
    """

    def generate(self, params: dict, context: dict) -> WorldMap:
        rng        = self.rng()
        tone       = params["tone"]
        world_size = params["world_size"]
        chapters   = params["chapters"]
        config: GameConfig = context["config"]

        size_map = {
            "tiny":   (2, 1),
            "small":  (3, 2),
            "medium": (4, 3),
            "large":  (6, 4),
            "epic":   (8, 6),
        }
        n_towns, n_dungeons = size_map.get(world_size, (4, 3))

        # Shuffle a copy — never mutate the module-level list
        area_names = list(_TONE_AREA_NAMES.get(tone, _TONE_AREA_NAMES["heroic"]))
        rng.shuffle(area_names)

        areas: dict[str, Area] = {}

        # Build towns first so we know their IDs for warp targets
        town_ids = []
        for i in range(n_towns):
            name = area_names[i % len(area_names)]
            town = self._make_town(rng, f"town_{i}", name, i)
            areas[town.id] = town
            town_ids.append(town.id)

        # Build dungeons
        dungeon_ids = []
        n_dungeon = min(n_dungeons, chapters)
        for i in range(n_dungeon):
            di = n_towns + i
            name = area_names[di % len(area_names)] if di < len(area_names) else f"Dungeon {i+1}"
            dungeon = self._make_dungeon(rng, f"dungeon_{i}", name, i + 1)
            areas[dungeon.id] = dungeon
            dungeon_ids.append(dungeon.id)

        # Build overworld last — places warp tiles pointing at the areas above
        overworld = self._make_overworld(rng, town_ids, dungeon_ids, config)
        areas[overworld.id] = overworld

        # Link connections
        for tid in town_ids:
            overworld.connections.append(tid)
            areas[tid].connections.append(overworld.id)
        for did in dungeon_ids:
            overworld.connections.append(did)
            areas[did].connections.append(overworld.id)

        start_town  = town_ids[0] if town_ids else overworld.id
        start_area  = areas[start_town]
        # Start just above the south exit of the first town
        start_x     = start_area.width  // 2
        start_y     = start_area.height - 3

        return WorldMap(
            areas        = areas,
            start_area   = start_town,
            start_pos    = (start_x, start_y),
            overworld_id = overworld.id,
        )

    # ── Overworld ─────────────────────────────────────────────────────────────

    def _make_overworld(self, rng, town_ids: list, dungeon_ids: list, config) -> Area:
        w, h = 32, 24
        grid = [[None] * w for _ in range(h)]

        # Fill base terrain
        for y in range(h):
            for x in range(w):
                if x == 0 or x == w-1 or y == 0 or y == h-1:
                    grid[y][x] = Tile(TileType.MOUNTAIN, sprite_key="mountain", passable=False)
                elif rng.random() < 0.06:
                    grid[y][x] = Tile(TileType.WATER,   sprite_key="water",   passable=False)
                elif rng.random() < 0.12:
                    grid[y][x] = Tile(TileType.FOREST,  sprite_key="forest")
                elif rng.random() < 0.04:
                    grid[y][x] = Tile(TileType.MOUNTAIN, sprite_key="mountain", passable=False)
                else:
                    grid[y][x] = Tile(TileType.GRASS,   sprite_key="grass")

        # Place warp tiles for towns and dungeons — spread evenly across the map
        all_area_ids = town_ids + dungeon_ids
        placed: list[tuple[int, int]] = []
        cols = max(1, int(len(all_area_ids) ** 0.5) + 1)
        cell_w = (w - 4) // max(1, cols)
        cell_h = (h - 4) // max(1, (len(all_area_ids) + cols - 1) // cols)

        for idx, area_id in enumerate(all_area_ids):
            col = idx % cols
            row = idx // cols
            # Pick a random spot inside this cell
            base_x = 2 + col * cell_w
            base_y = 2 + row * cell_h
            # Try up to 20 positions in the cell
            wx, wy = base_x + cell_w // 2, base_y + cell_h // 2
            for _ in range(20):
                cx2 = base_x + rng.randint(0, max(0, cell_w - 1))
                cy2 = base_y + rng.randint(0, max(0, cell_h - 1))
                if (1 < cx2 < w-2 and 1 < cy2 < h-2 and
                        grid[cy2][cx2].passable and
                        all(abs(cx2-px) + abs(cy2-py) > 2 for px, py in placed)):
                    wx, wy = cx2, cy2
                    break

            # Default entry point inside each area
            target_area = area_id
            grid[wy][wx] = Tile(TileType.WARP, sprite_key="warp",
                                 warp_target=f"{target_area}:10:12")
            placed.append((wx, wy))

            # Make sure the tile is reachable — clear adjacent tiles
            for dx2, dy2 in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx2, ny2 = wx+dx2, wy+dy2
                if 0 < nx2 < w-1 and 0 < ny2 < h-1 and not grid[ny2][nx2].passable:
                    grid[ny2][nx2] = Tile(TileType.GRASS, sprite_key="grass")

        return Area(
            id="overworld", name=config.world_name, area_type=AreaType.WORLD_MAP,
            width=w, height=h, grid=grid, music_key="overworld_theme",
        )

    # ── Town ──────────────────────────────────────────────────────────────────

    def _make_town(self, rng, area_id: str, name: str, town_index: int) -> Area:
        w, h = 20, 16
        grid = [[None] * w for _ in range(h)]

        # Floor everywhere inside
        for y in range(h):
            for x in range(w):
                grid[y][x] = Tile(TileType.TOWN_FLOOR, sprite_key="town_floor")

        # Outer wall
        for x in range(w):
            grid[0][x]   = Tile(TileType.WALL, sprite_key="wall", passable=False)
            grid[h-1][x] = Tile(TileType.WALL, sprite_key="wall", passable=False)
        for y in range(h):
            grid[y][0]   = Tile(TileType.WALL, sprite_key="wall", passable=False)
            grid[y][w-1] = Tile(TileType.WALL, sprite_key="wall", passable=False)

        # Buildings: 3–4 rectangular blocks of walls with a door
        n_buildings = rng.randint(3, 4)
        for _ in range(n_buildings):
            bw2 = rng.randint(3, 5)
            bh2 = rng.randint(2, 3)
            bx  = rng.randint(1, w - bw2 - 2)
            by2 = rng.randint(1, h - bh2 - 2)
            # Walls
            for iy in range(by2, by2 + bh2):
                for ix in range(bx, bx + bw2):
                    if 0 < ix < w-1 and 0 < iy < h-1:
                        grid[iy][ix] = Tile(TileType.WALL, sprite_key="wall", passable=False)
            # Door on south face
            door_x = bx + bw2 // 2
            door_y = by2 + bh2 - 1
            if 0 < door_x < w-1 and 0 < door_y < h-1:
                grid[door_y][door_x] = Tile(TileType.DOOR, sprite_key="door")

        # South exit warp to overworld
        cx = w // 2
        grid[h-2][cx] = Tile(TileType.WARP, sprite_key="warp",
                              warp_target="overworld:16:12")
        # Make sure the tiles above the exit are walkable
        if 0 < cx < w-1:
            grid[h-3][cx] = Tile(TileType.TOWN_FLOOR, sprite_key="town_floor")

        return Area(
            id=area_id, name=name, area_type=AreaType.TOWN,
            width=w, height=h, grid=grid, music_key="town_theme",
        )

    # ── Dungeon ───────────────────────────────────────────────────────────────

    def _make_dungeon(self, rng, area_id: str, name: str, chapter: int) -> Area:
        w, h = 24, 20
        # Start with all walls
        grid = [[Tile(TileType.WALL, sprite_key="wall", passable=False)
                 for _ in range(w)] for _ in range(h)]

        # Carve rooms using a simple BSP-style approach
        rooms: list[tuple[int,int,int,int]] = []  # (x,y,w,h)
        n_rooms = rng.randint(5, 8)
        for _ in range(n_rooms * 5):  # try multiple times to place rooms
            if len(rooms) >= n_rooms:
                break
            rw = rng.randint(3, 6)
            rh = rng.randint(3, 5)
            rx = rng.randint(1, w - rw - 1)
            ry = rng.randint(1, h - rh - 1)
            # Check for overlap
            overlap = any(
                rx < ox+ow+1 and rx+rw > ox-1 and ry < oy+oh+1 and ry+rh > oy-1
                for ox, oy, ow, oh in rooms
            )
            if not overlap:
                rooms.append((rx, ry, rw, rh))
                # Carve the room
                for cy2 in range(ry, ry+rh):
                    for cx2 in range(rx, rx+rw):
                        grid[cy2][cx2] = Tile(TileType.DUNGEON, sprite_key="dungeon")

        # Connect rooms with L-shaped corridors
        for i in range(1, len(rooms)):
            x1 = rooms[i-1][0] + rooms[i-1][2]//2
            y1 = rooms[i-1][1] + rooms[i-1][3]//2
            x2 = rooms[i][0]   + rooms[i][2]//2
            y2 = rooms[i][1]   + rooms[i][3]//2
            # Horizontal leg
            x_lo, x_hi = (x1, x2+1) if x1 <= x2 else (x2, x1+1)
            for cx2 in range(max(1,x_lo), min(w-1, x_hi)):
                grid[y1][cx2] = Tile(TileType.DUNGEON, sprite_key="dungeon")
            # Vertical leg
            y_lo, y_hi = (y1, y2+1) if y1 <= y2 else (y2, y1+1)
            for cy2 in range(max(1,y_lo), min(h-1, y_hi)):
                grid[cy2][x2] = Tile(TileType.DUNGEON, sprite_key="dungeon")

        # Entry warp in first room's center
        if rooms:
            ex = rooms[0][0] + rooms[0][2]//2
            ey = rooms[0][1] + rooms[0][3]//2
            grid[ey][ex] = Tile(TileType.WARP, sprite_key="warp",
                                 warp_target="overworld:16:12")
        else:
            grid[h-2][w//2] = Tile(TileType.WARP, sprite_key="warp",
                                    warp_target="overworld:16:12")

        # Boss tile in last room's center
        if len(rooms) >= 2:
            bx = rooms[-1][0] + rooms[-1][2]//2
            by2 = rooms[-1][1] + rooms[-1][3]//2
            grid[by2][bx] = Tile(TileType.WARP, sprite_key="warp",
                                  warp_target=f"town_0:{w//2}:{h//2}")

        # Chests in middle rooms
        chests = []
        chest_rooms = rooms[1:-1] if len(rooms) > 2 else rooms
        item_pool   = ["potion", "hi_potion", "ether", f"key_orb_{chapter}"]
        for i, (rx, ry, rw2, rh2) in enumerate(chest_rooms[:3+chapter]):
            # Place chest in a corner of the room (not the center)
            cx3 = rx + 1
            cy3 = ry + 1
            if grid[cy3][cx3].type == TileType.DUNGEON:
                grid[cy3][cx3] = Tile(TileType.CHEST, sprite_key="chest")
                chests.append({
                    "item_id": item_pool[i % len(item_pool)],
                    "x": cx3, "y": cy3, "opened": False,
                })

        # Gate entry on the key orb for this chapter
        # (engine checks area.metadata["requires_item"] before warping in)
        gate_item = f"key_orb_{chapter - 1}" if chapter > 1 else None

        return Area(
            id=area_id, name=name, area_type=AreaType.DUNGEON,
            width=w, height=h, grid=grid, music_key="dungeon_theme",
            chest_spawns=chests,
            metadata={"requires_item": gate_item} if gate_item else {},
        )


class NPCGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> dict:
        rng         = self.rng()
        tone        = params["tone"]
        antagonist  = params["antagonist"]
        difficulty  = params["difficulty"]
        world_map: WorldMap = context["world_map"]
        classes: dict       = context["classes"]

        npc_names   = _TONE_NPC_NAMES.get(tone, _TONE_NPC_NAMES["heroic"])
        enemy_names = _TONE_ENEMY_NAMES.get(tone, _TONE_ENEMY_NAMES["heroic"])

        npcs     = {}
        party_members = []

        # Shopkeeper in each town
        towns = world_map.areas_of_type(AreaType.TOWN)
        for i, town in enumerate(towns):
            npc_id = f"shopkeeper_{i}"
            name   = rng.choice(npc_names)
            npc_names_copy = [n for n in npc_names if n != name] or npc_names
            shop_id = f"shop_{i}"
            sx, sy = self._safe_spawn(town, town.width//2, town.height//2, rng)
            npc = NPC(
                id=npc_id, name=name, sprite_key="npc_shopkeeper",
                portrait_key=f"portrait_npc_{i}",
                area_id=town.id,
                spawn_x=sx,
                spawn_y=sy,
                dialogue_id=f"dlg_{npc_id}",
                is_shopkeeper=True, shop_id=shop_id,
                role="merchant",
            )
            npcs[npc_id] = npc
            town.npc_spawns.append({"npc_id": npc_id, "x": npc.spawn_x, "y": npc.spawn_y})

            # Add 2 villagers per town
            for j in range(2):
                vid = f"villager_{i}_{j}"
                vname = rng.choice(npc_names_copy)
                vsx, vsy = self._safe_spawn(town, rng.randint(2,town.width-3), rng.randint(2,town.height-3), rng)
                villager = NPC(
                    id=vid, name=vname, sprite_key="npc_villager",
                    area_id=town.id,
                    spawn_x=vsx,
                    spawn_y=vsy,
                    dialogue_id=f"dlg_{vid}",
                    role="villager",
                )
                npcs[vid] = villager
                town.npc_spawns.append({"npc_id": vid, "x": villager.spawn_x, "y": villager.spawn_y})

        # Quest giver in first town
        qgiver_id = "quest_giver_0"
        qname = rng.choice(npc_names)
        qg_area = towns[0] if towns else None
        qgsx, qgsy = self._safe_spawn(qg_area, 6, 4, rng) if qg_area else (6, 4)
        qgiver = NPC(
            id=qgiver_id, name=qname, sprite_key="npc_elder",
            area_id=towns[0].id if towns else "town_0",
            spawn_x=qgsx, spawn_y=qgsy,
            dialogue_id=f"dlg_{qgiver_id}",
            role="villager",
        )
        npcs[qgiver_id] = qgiver
        if towns:
            towns[0].npc_spawns.append({"npc_id": qgiver_id, "x": 6, "y": 4})

        # Enemies for each dungeon
        dungeons = world_map.areas_of_type(AreaType.DUNGEON)
        diff_mult = {"easy": 0.7, "normal": 1.0, "hard": 1.4, "expert": 1.8}[difficulty]
        for i, dungeon in enumerate(dungeons):
            level = (i + 1) * 3
            for j, ename in enumerate(enemy_names[:3]):
                eid = f"enemy_{i}_{j}"
                hp  = int((20 + level * 8) * diff_mult)
                atk = int((5  + level * 2) * diff_mult)
                dfs = int((3  + level * 1) * diff_mult)
                npc = NPC(
                    id=eid, name=ename, sprite_key=f"enemy_{j}",
                    is_enemy=True,
                    stats=Stats(hp=hp, mp=0, attack=atk, defense=dfs,
                                magic=2, speed=5, luck=3, max_hp=hp, max_mp=0),
                    loot_table=[
                        {"item_id":"potion", "chance":0.4},
                        {"item_id":"antidote", "chance":0.2},
                    ],
                    role="enemy",
                )
                npcs[eid] = npc

            # Boss for this dungeon
            boss_id   = f"boss_{i}"
            boss_name, boss_sprite = _ANTAGONIST_NAMES.get(
                antagonist, ("Dark Lord", "boss")
            ) if i == len(dungeons)-1 else (f"Guardian {i+1}", "boss")
            boss_hp   = int((80 + level * 15) * diff_mult)
            boss_atk  = int((12 + level * 3)  * diff_mult)
            boss = NPC(
                id=boss_id, name=boss_name, sprite_key=boss_sprite,
                is_enemy=True,
                stats=Stats(hp=boss_hp, mp=20, attack=boss_atk, defense=int(boss_atk*0.7),
                            magic=10, speed=6, luck=5, max_hp=boss_hp, max_mp=20),
                loot_table=[{"item_id": f"key_orb_{i+1}", "chance": 1.0}],
                role="boss",
            )
            npcs[boss_id] = boss

        # Add 1-2 recruitable party members (not the protagonist class)
        all_classes = list(_PROTAGONIST_CLASSES.keys())
        protagonist_class = params.get("protagonist", "warrior")
        other_classes = [c for c in all_classes if c != protagonist_class]
        rng.shuffle(other_classes)
        recruit_classes = other_classes[:min(2, params.get("party_size", 4) - 1)]

        for i, recruit_class in enumerate(recruit_classes):
            rid    = f"recruit_{i}"
            rname  = rng.choice([n for n in npc_names if n not in [
                npc.name for npc in npcs.values()]][:3] or npc_names)
            recruit_town = towns[min(i, len(towns)-1)] if towns else None
            rsx, rsy = self._safe_spawn(recruit_town, 4 + i*3, 5, rng) if recruit_town else (4, 5)
            recruit_npc = NPC(
                id=rid, name=rname, sprite_key=f"npc_{recruit_class}",
                portrait_key=f"portrait_{recruit_class}",
                area_id=recruit_town.id if recruit_town else "town_0",
                spawn_x=rsx, spawn_y=rsy,
                dialogue_id=f"dlg_{rid}",
                role="party_member",
                metadata={"class_id": recruit_class},
            )
            npcs[rid] = recruit_npc
            party_members.append(rid)
            if recruit_town:
                recruit_town.npc_spawns.append({"npc_id": rid, "x": rsx, "y": rsy})

        return {
            "npcs": npcs,
            "party_members": party_members,
            "classes": classes,
        }

    def _safe_spawn(self, area, preferred_x: int, preferred_y: int, rng) -> tuple:
        """Find nearest walkable tile to the preferred position."""
        if area is None:
            return preferred_x, preferred_y
        # Spiral outward from preferred position
        for radius in range(0, 8):
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue  # only check ring
                    x = preferred_x + dx
                    y = preferred_y + dy
                    tile = area.tile_at(x, y)
                    if tile and tile.passable and tile.warp_target is None:
                        return x, y
        # Fallback: first walkable tile
        for wy in range(1, area.height-1):
            for wx in range(1, area.width-1):
                t = area.tile_at(wx, wy)
                if t and t.passable and t.warp_target is None:
                    return wx, wy
        return preferred_x, preferred_y


class DialogueGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> dict[str, DialogueTree]:
        rng   = self.rng()
        tone  = params["tone"]
        npcs: dict[str, NPC] = context["npc_data"]["npcs"]
        world_map: WorldMap  = context["world_map"]

        dialogues = {}
        greetings = {
            "heroic":     ["Welcome, brave traveler!", "Adventure awaits you!",
                           "The kingdom needs heroes like you."],
            "dark":       ["You shouldn't be here...", "Leave while you still can.",
                           "There is no hope left in this land."],
            "comedic":    ["Oh! A customer! Or are you lost?", "Nice hat! Wait, that's your head.",
                           "I sell things! Sometimes they're useful!"],
            "mysterious": ["You seek something...", "The stars spoke of your coming.",
                           "Not many find their way here."],
            "tragic":     ["Another wanderer in these cursed lands.", "We have little to offer.",
                           "Hope? We buried that long ago."],
        }[tone]

        shop_lines = ["I have wares, if you have coin.", "Buy something or move along.",
                      "Quality goods at reasonable prices!"]

        for npc_id, npc in npcs.items():
            if npc.is_enemy:
                continue
            tree_id = f"dlg_{npc_id}"

            if npc.is_shopkeeper:
                root_node = DialogueNode(
                    id=f"{tree_id}_root",
                    type=DialogueNodeType.CHOICE,
                    speaker=npc.name,
                    text=rng.choice(shop_lines),
                    choices=[
                        DialogueChoice("Browse wares", f"{tree_id}_shop"),
                        DialogueChoice("Never mind", f"{tree_id}_bye"),
                    ]
                )
                # SET_FLAG node: sets open_shop flag so engine opens shop
                shop_node = DialogueNode(
                    id=f"{tree_id}_shop",
                    type=DialogueNodeType.SET_FLAG,
                    speaker=npc.name,
                    text="Take a look.",
                    flag_key=f"open_shop",
                    flag_value=npc.shop_id,
                )
                bye_node = DialogueNode(
                    id=f"{tree_id}_bye",
                    type=DialogueNodeType.TEXT,
                    speaker=npc.name,
                    text="Come back anytime.",
                )
                nodes = {
                    root_node.id: root_node,
                    shop_node.id: shop_node,
                    bye_node.id:  bye_node,
                }
            else:
                greeting = rng.choice(greetings)
                root_node = DialogueNode(
                    id=f"{tree_id}_root",
                    type=DialogueNodeType.TEXT,
                    speaker=npc.name,
                    text=greeting,
                )
                nodes = {root_node.id: root_node}

            dialogues[tree_id] = DialogueTree(
                id=tree_id, nodes=nodes, root=f"{tree_id}_root"
            )

        # Dialogue for recruitable party members
        for npc_id, npc in npcs.items():
            if npc.role != "party_member":
                continue
            tree_id  = f"dlg_{npc_id}"
            join_lines = {
                "heroic":     "I've been waiting for someone like you. Let me join your quest!",
                "dark":       "I have nowhere else to go. Perhaps our fates are entwined.",
                "comedic":    "Oh! A hero! Can I come? I'm very good at carrying things!",
                "mysterious": "The stars showed me this moment. I was meant to travel with you.",
                "tragic":     "I've lost everything. Fighting alongside you... it gives me purpose.",
            }
            join_text = join_lines.get(tone, "I'd like to join your journey, if you'll have me.")
            root_node = DialogueNode(
                id=f"{tree_id}_root",
                type=DialogueNodeType.CHOICE,
                speaker=npc.name,
                text=join_text,
                choices=[
                    DialogueChoice("Yes, join us!", f"{tree_id}_yes"),
                    DialogueChoice("Not now.",      f"{tree_id}_no"),
                ]
            )
            yes_node = DialogueNode(
                id=f"{tree_id}_yes",
                type=DialogueNodeType.SET_FLAG,
                speaker=npc.name,
                text="",
                flag_key=f"recruit_{npc_id}",
                flag_value=True,
            )
            no_node = DialogueNode(
                id=f"{tree_id}_no",
                type=DialogueNodeType.TEXT,
                speaker=npc.name,
                text="I'll be here if you change your mind.",
            )
            dialogues[tree_id] = DialogueTree(
                id=tree_id,
                nodes={root_node.id: root_node, yes_node.id: yes_node, no_node.id: no_node},
                root=root_node.id,
            )

        return dialogues


class QuestGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> QuestGraph:
        rng       = self.rng()
        chapters  = params["chapters"]
        antagonist = params["antagonist"]
        npcs: dict = context["npc_data"]["npcs"]
        world_map: WorldMap = context["world_map"]

        quests    = {}
        main_chain = []

        ant_name, _ = _ANTAGONIST_NAMES.get(antagonist, ("Dark Lord","boss"))
        dungeons = world_map.areas_of_type(AreaType.DUNGEON)

        # Main quest chain
        for i in range(chapters):
            qid = f"main_q{i+1}"
            dname = dungeons[i].name if i < len(dungeons) else f"The Citadel"
            quest = Quest(
                id=qid,
                name=f"Chapter {i+1}: The Path Ahead" if i < chapters-1 else "Final Chapter: The Last Stand",
                description=(f"Reach {dname} and retrieve the Sacred Orb."
                             if i < chapters-1
                             else f"Confront {ant_name} and end the darkness."),
                giver_npc="quest_giver_0",
                objectives=[
                    QuestObjective(f"{qid}_reach", f"Reach {dname}",
                                   "reach_area", dungeons[i].id if i < len(dungeons) else "dungeon_0"),
                    QuestObjective(f"{qid}_item",  f"Obtain the Sacred Orb {i+1}",
                                   "collect_item", f"key_orb_{i+1}"),
                ],
                prerequisites=[f"main_q{i}"] if i > 0 else [],
                reward_gold   = (i+1) * 200,
                reward_exp    = (i+1) * 300,
                reward_items  = ["hi_potion"],
                reward_unlock = [f"main_q{i+2}"] if i < chapters-1 else [],
                is_main_quest = True,
                chapter       = i+1,
            )
            quests[qid] = quest
            main_chain.append(qid)

        # Side quests (one per town)
        towns = world_map.areas_of_type(AreaType.TOWN)
        side_quest_names = [
            ("Missing Ingredient", "Collect an herb from the forest."),
            ("Lost Traveler",      "Find a missing villager."),
            ("Stolen Goods",       "Retrieve stolen items from monsters."),
            ("Old Grudge",         "Deliver a letter to a distant town."),
        ]
        for i, town in enumerate(towns[:len(side_quest_names)]):
            sqid = f"side_q{i+1}"
            sname, sdesc = side_quest_names[i % len(side_quest_names)]
            # Find an NPC in this town to give the quest
            givers = [nid for nid, n in npcs.items()
                      if n.area_id == town.id and not n.is_enemy and not n.is_shopkeeper]
            giver  = givers[0] if givers else "quest_giver_0"
            sq = Quest(
                id=sqid, name=sname, description=sdesc,
                giver_npc=giver,
                objectives=[
                    QuestObjective(f"{sqid}_obj", sdesc.rstrip("."),
                                   "collect_item", rng.choice(["potion","antidote","ether"])),
                ],
                reward_gold  = 50 + i * 25,
                reward_exp   = 80 + i * 40,
                reward_items = ["potion"],
                is_main_quest = False,
            )
            quests[sqid] = sq

        return QuestGraph(quests=quests, main_chain=main_chain)


class EncounterGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> EncounterTable:
        rng         = self.rng()
        world_map: WorldMap = context["world_map"]
        npc_data    = context["npc_data"]
        npcs        = npc_data["npcs"]
        tables      = {}

        dungeons = world_map.areas_of_type(AreaType.DUNGEON)
        for i, dungeon in enumerate(dungeons):
            enemy_ids = [eid for eid, n in npcs.items()
                         if n.is_enemy and not n.role == "boss"
                         and eid.startswith(f"enemy_{i}_")]
            if not enemy_ids:
                continue
            groups = []
            # Single enemy
            for eid in enemy_ids:
                groups.append(EnemyGroup([eid], weight=3.0, min_level=1))
            # Pair
            if len(enemy_ids) >= 2:
                groups.append(EnemyGroup([enemy_ids[0], enemy_ids[1]], weight=2.0))
            # Triple
            if len(enemy_ids) >= 1:
                groups.append(EnemyGroup([enemy_ids[0]]*3, weight=1.0))
            tables[dungeon.id] = groups

        # Overworld encounters (weaker)
        overworld_enemies = [eid for eid, n in npcs.items()
                             if n.is_enemy and eid.startswith("enemy_0_")]
        if overworld_enemies:
            tables["overworld"] = [
                EnemyGroup([overworld_enemies[0]], weight=3.0),
                EnemyGroup([overworld_enemies[0]] * 2, weight=1.0),
            ]

        return EncounterTable(tables=tables)


class ShopGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> ShopInventory:
        rng         = self.rng()
        world_map: WorldMap = context["world_map"]
        npc_data    = context["npc_data"]
        items: ItemRegistry = context["items"]

        shops = {}
        towns = world_map.areas_of_type(AreaType.TOWN)
        for i, town in enumerate(towns):
            shop_id = f"shop_{i}"
            # Earlier towns have cheaper items, later towns have better stock
            tier = i / max(1, len(towns) - 1)
            available = []
            for item in items.items.values():
                if item.key_item:
                    continue
                if item.price <= 100 + tier * 400:
                    available.append({"item_id": item.id, "stock": 99,
                                      "price_mult": 1.0 + tier * 0.1})
            rng.shuffle(available)
            shops[shop_id] = available[:8]  # max 8 items per shop

        return ShopInventory(shops=shops)


class AssetManifestGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> AssetManifest:
        world_map: WorldMap = context["world_map"]
        npc_data            = context["npc_data"]
        items: ItemRegistry = context["items"]
        classes: dict       = context["classes"]

        sprites = {}
        music   = {}

        # Tile sprites
        for tile_type in TileType:
            sprites[tile_type.value] = ""  # empty = use placeholder

        # Character sprites
        for class_id in classes:
            sprites[f"player_{class_id}"] = ""
            sprites[f"portrait_{class_id}"] = ""
        sprites["player"] = ""  # generic fallback

        # NPC sprites
        for npc in npc_data["npcs"].values():
            if npc.sprite_key:
                sprites[npc.sprite_key] = ""
            if npc.portrait_key:
                sprites[npc.portrait_key] = ""

        # Item sprites
        for item in items.items.values():
            sprites[f"item_{item.id}"] = ""

        # Chest variants
        sprites["chest"]      = ""
        sprites["chest_open"] = ""

        # UI
        sprites["cursor"] = ""

        # Music
        for key in ["overworld_theme", "town_theme", "dungeon_theme",
                    "battle_theme", "boss_theme", "fanfare", "game_over"]:
            music[key] = ""

        return AssetManifest(sprites=sprites, music=music)


# ─────────────────────────────────────────────────────────────────────────────
# Domain
# ─────────────────────────────────────────────────────────────────────────────

class JRPGDomain(DomainBase):
    """
    The procgen domain for JRPG game generation.

    Designer intent → GenerationPlan → Generators → World
    """

    def __init__(self):
        self.schema = ConstraintSet([
            Rule("genre",          type=str, required=True,
                 choices=["jrpg","action_rpg","dungeon_crawler","visual_novel"],
                 description="Game genre"),
            Rule("tone",           type=str, required=True,
                 choices=["heroic","dark","comedic","mysterious","tragic"],
                 description="Narrative tone"),
            Rule("world_size",     type=str, default="medium",
                 choices=["tiny","small","medium","large","epic"],
                 description="Number of areas and content density"),
            Rule("difficulty",     type=str, default="normal",
                 choices=["easy","normal","hard","expert"],
                 description="Combat and progression difficulty"),
            Rule("protagonist",    type=str, default="warrior",
                 choices=list(_PROTAGONIST_CLASSES.keys()),
                 description="Protagonist's class archetype"),
            Rule("antagonist",     type=str, default="demon_lord",
                 choices=list(_ANTAGONIST_NAMES.keys()),
                 description="The main villain type"),
            Rule("chapters",       type=int, default=3, range=(1, 6),
                 description="Number of story chapters"),
            Rule("party_size",     type=int, default=4, range=(1, 4),
                 description="Maximum party members"),
            Rule("title",          type=str, default="",
                 description="Game title (auto-generated if empty)"),
            Rule("world_name",     type=str, default="",
                 description="World name (auto-generated if empty)"),
            Rule("protagonist_name", type=str, default="Hero",
                 description="The player character's name"),
        ])

    def validate_intent(self, intent: Intent) -> None:
        self.schema.validate(intent)

    def build_plan(self, intent: Intent) -> GenerationPlan:
        plan = GenerationPlan()
        shared = {k: intent.get(k) for k in intent.keys()}
        plan.add_step("config",    shared)
        plan.add_step("classes",   shared)
        plan.add_step("items",     shared)
        plan.add_step("world_map", shared)
        plan.add_step("npc_data",  shared)
        plan.add_step("dialogues", shared)
        plan.add_step("quests",    shared)
        plan.add_step("encounters",shared)
        plan.add_step("shops",     shared)
        plan.add_step("assets",    shared)
        return plan

    def assemble(self, outputs: dict, plan: GenerationPlan) -> World:
        npc_d  = outputs["npc_data"]
        params = plan.steps[0][1] if plan.steps else {}
        return World(
            config          = outputs["config"],
            world_map       = outputs["world_map"],
            npc_roster      = NPCRoster(
                npcs            = npc_d["npcs"],
                player_classes  = npc_d["classes"],
                party_members   = npc_d["party_members"],
            ),
            quest_graph     = outputs["quests"],
            item_registry   = outputs["items"],
            encounter_table = outputs["encounters"],
            shop_inventory  = outputs["shops"],
            dialogues       = outputs["dialogues"],
            asset_manifest  = outputs["assets"],
            seed            = plan.seed.value if plan.seed else 0,
            # Store generation params so save/load can reconstruct the world
            metadata        = {
                "antagonist":  params.get("antagonist",  "demon_lord"),
                "world_size":  params.get("world_size",  "medium"),
                "protagonist_name": params.get("protagonist_name", "Hero"),
            },
        )

    @property
    def generators(self):
        return {
            "config":     ConfigGenerator,
            "classes":    ClassesGenerator,
            "items":      ItemsGenerator,
            "world_map":  WorldMapGenerator,
            "npc_data":   NPCGenerator,
            "dialogues":  DialogueGenerator,
            "quests":     QuestGenerator,
            "encounters": EncounterGenerator,
            "shops":      ShopGenerator,
            "assets":     AssetManifestGenerator,
        }
