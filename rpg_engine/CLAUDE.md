# RPG Engine — Claude Code Instructions

## What this project is

A procedurally generated JRPG engine built in pure Python. Given high-level
designer intent (tone, protagonist class, antagonist type, world size), it
generates a complete playable game world and runs it with pygame.

Two systems, cleanly separated:

- **procgen/** — Generic procedural generation framework (seeds, generators,
  constraints, WFC, grammars). Domain-agnostic — knows nothing about games.
- **engine/** — Game runtime (ECS entities, tile maps, battle system,
  dialogue, shops, quests, save/load). Knows nothing about generation.
- **procgen_domain/** — The bridge. Translates designer intent into World
  data that the engine reads.

## Project layout

```
rpg_engine/
├── gameengine.py          # Public API — start here
├── demo.py                # Demo / showcase script
├── tests.py               # Test suite (33 tests)
│
├── engine/
│   ├── runtime.py         # pygame game loop, all Scene classes
│   └── core/
│       ├── world.py       # World data model (generated content)
│       ├── state.py       # Runtime state (player, inventory, quests)
│       └── ecs.py         # Entity Component System
│
├── procgen/               # Generic procgen framework (don't modify)
│   ├── core.py            # Intent, GenerationPlan, Designer
│   ├── engine.py          # DecisionEngine
│   ├── generators.py      # GeneratorBase
│   ├── seeds.py           # Deterministic child seeds
│   ├── constraints.py     # Schema validation
│   ├── grammar.py         # Context-free grammars
│   ├── wfc.py             # Wave Function Collapse
│   └── domains/           # Example domains (music, dungeon, terrain)
│
├── procgen_domain/
│   └── __init__.py        # JRPGDomain + all 10 generators
│
├── assets/                # Put your sprites/music here
│   ├── sprites/           # PNG files (32×32 recommended)
│   ├── tiles/             # Tile sprites
│   ├── music/             # OGG/MP3 files
│   ├── sounds/            # Sound effects
│   └── fonts/             # TTF fonts (optional)
│
└── saves/                 # Save files go here
```

## How to run

```bash
pip install pygame          # one-time setup

python demo.py              # generate + show world summaries
python demo.py --run        # generate + run the game
python demo.py --run dark_fantasy seed=42
```

Or from Python:

```python
from gameengine import GameDesigner, GameEngine

# Generate and run
GameDesigner().set("tone","dark").set("protagonist","mage").run(seed=42)

# Generate only (no pygame needed)
world = GameDesigner().set("tone","heroic").generate(seed=42)
print(world.describe())

# Load a saved game
engine = GameEngine.load_game("saves/save_01.json")
engine.run()
```

## Key controls (in game)

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move |
| Enter / Z | Interact / Confirm |
| Escape | Menu |
| ESC in menu | Close menu |

## Adding your own assets

The engine generates colored placeholder sprites for everything that's
missing. To use real assets, pass `asset_paths` to `GameEngine`:

```python
world = GameDesigner().generate(seed=42)
GameEngine(world, asset_paths={
    # Sprite keys from world.asset_manifest.sprites
    "player":      "assets/sprites/hero.png",
    "grass":       "assets/tiles/grass.png",
    "wall":        "assets/tiles/wall.png",
    "town_floor":  "assets/tiles/floor.png",
    "dungeon":     "assets/tiles/dungeon.png",
    "npc_villager":"assets/sprites/villager.png",
    "chest":       "assets/sprites/chest_closed.png",
    "chest_open":  "assets/sprites/chest_open.png",
    # Music keys from world.asset_manifest.music
    "overworld_theme": "assets/music/overworld.ogg",
    "town_theme":      "assets/music/town.ogg",
    "dungeon_theme":   "assets/music/dungeon.ogg",
    "battle_theme":    "assets/music/battle.ogg",
}).run()
```

To see all sprite/music keys a world needs:
```python
world = GameDesigner().generate(seed=42)
print("Sprites:", list(world.asset_manifest.sprites.keys()))
print("Music:",   list(world.asset_manifest.music.keys()))
```

## Running tests

```bash
python tests.py
```

All 33 tests should pass. Tests cover: generation correctness, world
integrity (no null tiles, all warps valid, NPCs on walkable tiles),
variety across seeds, runtime (MapScene, BattleScene, inventory, quests,
save/load), and content quality.

## Architecture — how generation works

```
GameDesigner.generate(seed)
    └── Designer.generate()
            └── DecisionEngine.resolve(intent, seed)
                    └── GenerationPlan (10 ordered steps)
            └── _execute(plan)
                    ├── ConfigGenerator       → GameConfig
                    ├── ClassesGenerator      → player classes
                    ├── ItemsGenerator        → ItemRegistry
                    ├── WorldMapGenerator     → WorldMap (areas + tile grids)
                    ├── NPCGenerator          → NPCs, enemies, recruits
                    ├── DialogueGenerator     → DialogueTrees
                    ├── QuestGenerator        → QuestGraph
                    ├── EncounterGenerator    → EncounterTable
                    ├── ShopGenerator         → ShopInventory
                    └── AssetManifestGenerator→ AssetManifest
            └── JRPGDomain.assemble() → World
```

Each generator receives `params` (intent values) and `context` (outputs
from all previous steps). The seed is deterministic — same seed always
produces the same world.

## Architecture — how the engine works

```
GameEngine(world).run()
    └── pygame main loop
            ├── TitleScene          (title screen, start music)
            ├── MapScene            (tile rendering, movement, interaction)
            │   ├── BattleScene     (turn-based combat, pushed onto stack)
            │   ├── DialogueScene   (NPC dialogue trees)
            │   ├── ShopScene       (buy/sell items)
            │   ├── SkillMenuScene  (in-battle skill selection)
            │   ├── ItemMenuScene   (in-battle item use)
            │   ├── MenuScene       (pause menu)
            │   ├── StatusScene     (party stats)
            │   └── InventoryScene  (item list)
            └── GameOverScene       (defeat screen)
```

Scenes are stacked — MapScene stays rendered under DialogueScene, etc.
Only the top scene receives input and update calls.

## Common tasks for Claude Code

**Add a new item type:**
Edit `procgen_domain/__init__.py` → `_ITEM_TEMPLATES` list.
Add an entry with `id`, `name`, `type` (ItemType.*), `price`, `effect`.

**Add a new NPC role:**
Edit `NPCGenerator.generate()` in `procgen_domain/__init__.py`.
Create NPC objects and append to `npcs` dict.

**Add a new quest type:**
Edit `QuestGenerator.generate()`. Add Quest with objectives of the new
`type` string. Add handling in `MapScene._check_quest_completion_map()`.

**Add a new Scene:**
Subclass `Scene` in `engine/runtime.py`. Implement `handle_event()`,
`update()`, `render()`. Push with `engine.push_scene(MyScene(engine))`.

**Extend the world data model:**
Edit `engine/core/world.py`. Data classes only — no game logic.
Update `JRPGDomain.assemble()` if you added a top-level field to `World`.

**Extend game state:**
Edit `engine/core/state.py`. Add fields to `GameState`.
Update `GameState.to_dict()` and `GameEngine.load_game()` for persistence.

**Add a new genre (not just JRPG):**
Create a new domain class in `procgen_domain/` following the same
`GeneratorBase` pattern. Register it in a new `Designer(domain=MyDomain())`.
The procgen framework is fully generic.
