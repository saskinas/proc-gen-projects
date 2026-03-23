# RPG Engine

A procedurally generated JRPG engine. Describe the game you want in plain
parameters — the engine generates the world, populates it with NPCs, quests,
items, and enemies, and runs it as a playable pygame game.

## Quick start

```bash
pip install pygame
python demo.py --run
```

## Design your own game

```python
from gameengine import GameDesigner

GameDesigner() \
    .set("tone",             "dark") \
    .set("protagonist",      "mage") \
    .set("antagonist",       "ancient_evil") \
    .set("world_size",       "medium") \
    .set("difficulty",       "hard") \
    .set("title",            "Echoes of the Void") \
    .set("protagonist_name", "Kira") \
    .run(seed=42)
```

## Parameters

| Parameter | Choices | Default |
|-----------|---------|---------|
| `tone` | `heroic` `dark` `comedic` `mysterious` `tragic` | `heroic` |
| `protagonist` | `warrior` `mage` `rogue` `scholar` `outcast` | `warrior` |
| `antagonist` | `demon_lord` `empire` `corruption` `ancient_evil` `rival` | `demon_lord` |
| `world_size` | `tiny` `small` `medium` `large` `epic` | `medium` |
| `difficulty` | `easy` `normal` `hard` `expert` | `normal` |
| `chapters` | 1–6 | `3` |
| `party_size` | 1–4 | `4` |
| `title` | any string | auto-generated |
| `world_name` | any string | auto-generated |
| `protagonist_name` | any string | `"Hero"` |

## Presets

```python
from gameengine import GameDesigner
for k, v in GameDesigner.presets()["dark_fantasy"].items():
    ...
# Presets: classic_adventure, dark_fantasy, comedic_quest, arcane_mystery,
#          tragic_hero, mage_adventure, quick_dungeon, epic_saga
```

## Controls

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move |
| Enter / Z | Interact |
| Escape | Menu |

## Bring your own assets

All sprites and music are optional — the engine generates colored
placeholder tiles for anything missing. See `CLAUDE.md` for the full
asset key list and how to provide real files.

## Save / Load

The game auto-saves to `saves/` via the in-game menu. Load with:

```python
from gameengine import GameEngine
GameEngine.load_game("saves/save_01.json").run()
```

## Running tests

```bash
python tests.py   # 33 tests, all should pass
```
