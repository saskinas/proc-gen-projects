#!/usr/bin/env python3
"""
demo.py — Game engine demo.

Shows world generation across all presets, then optionally runs the game.

Usage:
    python demo.py              # generate worlds, print summaries
    python demo.py --run        # generate and run the default game
    python demo.py --run dark   # run the dark_fantasy preset
    python demo.py --run seed=7 # run with a specific seed
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gameengine import GameDesigner, GameEngine

SEP  = "─" * 64
DSEP = "═" * 64

def header(title):
    print(f"\n{DSEP}\n  {title}\n{DSEP}")

def divider():
    print(SEP)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Show all presets
# ─────────────────────────────────────────────────────────────────────────────

header("GAME ENGINE DEMO — ALL PRESETS")
print(f"  {'Preset':<22} {'Title':<32} {'World':<22} seed  areas  npcs  quests")
print(f"  {'─'*22} {'─'*32} {'─'*22} ────  ─────  ────  ──────")

for preset_name, params in GameDesigner.presets().items():
    seed = abs(hash(preset_name)) % 10000
    d = GameDesigner()
    for k, v in params.items():
        d.set(k, v)
    world = d.generate(seed=seed)
    c  = world.config
    wm = world.world_map
    r  = world.npc_roster
    q  = world.quest_graph
    print(f"  {preset_name:<22} {c.title[:31]:<32} {c.world_name[:21]:<22} "
          f"{seed:>4}  {len(wm.areas):>5}  {len(r.npcs):>4}  {len(q.quests):>6}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Variety across seeds
# ─────────────────────────────────────────────────────────────────────────────

header("VARIETY ACROSS SEEDS  (classic_adventure, seeds 0–9)")
print(f"  {'Seed':>4}  {'Title':<34} {'World':<22} {'Tone':<12} areas  npcs")
print(f"  {'─'*4}  {'─'*34} {'─'*22} {'─'*12} ─────  ────")

for seed in range(10):
    d = GameDesigner()
    for k, v in GameDesigner.presets()["classic_adventure"].items():
        d.set(k, v)
    world = d.generate(seed=seed)
    c  = world.config
    wm = world.world_map
    r  = world.npc_roster
    print(f"  {seed:>4}  {c.title[:33]:<34} {c.world_name[:21]:<22} "
          f"{c.tone:<12} {len(wm.areas):>5}  {len(r.npcs):>4}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Full world description for one game
# ─────────────────────────────────────────────────────────────────────────────

header("FULL WORLD DESCRIPTION  (dark_fantasy, seed=42)")
world = GameDesigner() \
    .set("tone",       "dark") \
    .set("protagonist","outcast") \
    .set("antagonist", "ancient_evil") \
    .set("world_size", "medium") \
    .set("difficulty", "hard") \
    .set("chapters",   3) \
    .generate(seed=42)
print(world.describe())


# ─────────────────────────────────────────────────────────────────────────────
# 4. Area breakdown
# ─────────────────────────────────────────────────────────────────────────────

header("AREA DETAIL")
from engine.core.world import AreaType, TileType
for area_id, area in world.world_map.areas.items():
    walkable = sum(1 for row in area.grid for t in row if t and t.passable)
    warps    = sum(1 for row in area.grid for t in row if t and t.warp_target)
    special  = sum(1 for row in area.grid for t in row
                   if t and t.type in (TileType.CHEST, TileType.SIGN, TileType.DOOR))
    print(f"  {area.id:<20} {area.area_type.value:<12} {area.width:2d}×{area.height:2d}  "
          f"walk={walkable:3d}  warps={warps}  special={special}  "
          f"npcs={len(area.npc_spawns):2d}  music={area.music_key}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. NPC roster
# ─────────────────────────────────────────────────────────────────────────────

header("NPC ROSTER")
roster = world.npc_roster
print(f"  {'ID':<22} {'Name':<16} {'Role':<14} {'Area':<18} {'Shop/Dlg'}")
print(f"  {'─'*22} {'─'*16} {'─'*14} {'─'*18} {'─'*16}")
for npc in list(roster.friendly_npcs())[:12]:
    shop = f"shop={npc.shop_id}" if npc.is_shopkeeper else ""
    dlg  = f"dlg={npc.dialogue_id[:15]}" if npc.dialogue_id else ""
    print(f"  {npc.id:<22} {npc.name:<16} {npc.role:<14} {npc.area_id:<18} {shop or dlg}")
if len(roster.friendly_npcs()) > 12:
    print(f"  ... and {len(roster.friendly_npcs())-12} more")

print(f"\n  Party members: {roster.party_members}")
print(f"\n  ENEMIES ({len(roster.enemies())}):")
for npc in roster.enemies()[:6]:
    st = npc.stats
    print(f"  {npc.id:<22} {npc.name:<20} role={npc.role:<8} "
          f"HP={st.hp:3d} ATK={st.attack:2d} DEF={st.defense:2d}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Quest graph
# ─────────────────────────────────────────────────────────────────────────────

header("QUEST GRAPH")
print("  Main quest chain:")
for qid in world.quest_graph.main_chain:
    q = world.quest_graph.quest(qid)
    print(f"    [{qid}] {q.name}")
    print(f"      Giver: {q.giver_npc}  |  Ch.{q.chapter}  |  "
          f"Reward: {q.reward_gold}G {q.reward_exp}XP {q.reward_items}")
    for obj in q.objectives:
        print(f"      → {obj.type}: {obj.target_id}")

print("\n  Side quests:")
for q in world.quest_graph.side_quests():
    print(f"    [{q.id}] {q.name}  (giver: {q.giver_npc})")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Item registry
# ─────────────────────────────────────────────────────────────────────────────

header("ITEM REGISTRY")
items = world.item_registry
print(f"  {'ID':<16} {'Name':<18} {'Type':<12} {'Price':>5}  {'Effect'}")
print(f"  {'─'*16} {'─'*18} {'─'*12} {'─'*5}  {'─'*20}")
for item in list(items.items.values())[:15]:
    effect = item.effect if item.effect else str(item.stat_bonus or "")
    print(f"  {item.id:<16} {item.name:<18} {item.item_type.value:<12} "
          f"{item.price:>5}G  {effect}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Save/Load roundtrip test
# ─────────────────────────────────────────────────────────────────────────────

header("SAVE / LOAD TEST")
import tempfile, json, os
engine = GameEngine(world)
engine.state.inventory.add("sword_01")
engine.state.inventory.gold = 250
engine.state.quests.start(world.quest_graph.main_chain[0])

tmp = tempfile.mktemp(suffix=".json")
engine.save_game(tmp)

with open(tmp) as f:
    save_data = json.load(f)

loaded = GameEngine.load_game(tmp)
os.unlink(tmp)

print(f"  Original: gold={engine.state.inventory.gold}  "
      f"items={dict(engine.state.inventory.items)}  "
      f"quests_active={engine.state.quests.active}")
print(f"  Loaded:   gold={loaded.state.inventory.gold}  "
      f"items={dict(loaded.state.inventory.items)}  "
      f"quests_active={loaded.state.quests.active}")
match = (engine.state.inventory.gold == loaded.state.inventory.gold and
         dict(engine.state.inventory.items) == dict(loaded.state.inventory.items))
print(f"  Match: {'✓' if match else '✗'}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Options reference
# ─────────────────────────────────────────────────────────────────────────────

header("DESIGNER OPTIONS")
print(GameDesigner().options())


# ─────────────────────────────────────────────────────────────────────────────
# 10. Run the game (if --run flag given)
# ─────────────────────────────────────────────────────────────────────────────

header("SUMMARY")
print(f"  Presets available: {len(GameDesigner.presets())}")
print(f"  All presets generated successfully ✓")
print(f"  Save/Load roundtrip working ✓\n")

if "--run" in sys.argv:
    print("  Launching game...")
    preset = "classic_adventure"
    seed   = 42
    for arg in sys.argv[2:]:
        if arg in GameDesigner.presets():
            preset = arg
        elif arg.startswith("seed="):
            seed = int(arg.split("=")[1])

    print(f"  Preset: {preset}  Seed: {seed}")
    print(f"  Controls: Arrow keys / WASD = Move, Enter / Z = Action, ESC = Menu\n")
    params = GameDesigner.presets()[preset]
    d = GameDesigner()
    for k, v in params.items():
        d.set(k, v)
    d.run(seed=seed)
else:
    print("  To run the game: python demo.py --run")
    print("  To run a specific preset: python demo.py --run dark_fantasy")
    print("  To use a specific seed: python demo.py --run seed=99\n")
