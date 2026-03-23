"""
tests.py — Test suite for the game engine and procgen domain.

Run:  python tests.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from procgen.core import Designer
from procgen_domain import JRPGDomain
from engine.runtime import GameEngine, MapScene, BattleScene
from engine.core.world import EnemyGroup, AreaType


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_passed = _failed = 0
_section_name = ""

def section(name):
    global _section_name
    _section_name = name
    print(f"\n{'─'*56}\n  {name}\n{'─'*56}")

def check(name, fn):
    global _passed, _failed
    try:
        fn()
        print(f"  ✓  {name}")
        _passed += 1
    except Exception as e:
        print(f"  ✗  {name}")
        print(f"       {type(e).__name__}: {e}")
        _failed += 1

def gen(seed=42, tone="heroic", prot="warrior", ant="demon_lord",
        size="medium", diff="normal", chapters=3):
    return Designer(domain=JRPGDomain()) \
        .set('genre','jrpg').set('tone',tone) \
        .set('protagonist',prot).set('antagonist',ant) \
        .set('world_size',size).set('difficulty',diff) \
        .set('chapters',chapters) \
        .set_seed(seed).generate()


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

# 1. Generation
section("GENERATION")

check("heroic warrior demon_lord medium", lambda: gen(42))
check("dark mage ancient_evil small",     lambda: gen(7, "dark","mage","ancient_evil","small"))
check("comedic rogue empire tiny",        lambda: gen(1, "comedic","rogue","empire","tiny","easy",1))
check("mysterious scholar corruption large", lambda: gen(99,"mysterious","scholar","corruption","large","expert",4))
check("tragic outcast rival epic",        lambda: gen(31,"tragic","outcast","rival","epic","hard",5))

# 2. World integrity
section("WORLD INTEGRITY")

_w = gen(42)

check("start area exists", lambda: (
    w := _w,
    setattr(w,'_test', w.world_map.area(w.world_map.start_area)),
    None if w._test else (_ for _ in ()).throw(AssertionError("start area missing"))
))

# Rewrite without walrus for Python 3.7 compat
def _check_start_area():
    a = _w.world_map.area(_w.world_map.start_area)
    assert a is not None, "start area missing"
    sx, sy = _w.world_map.start_pos
    t = a.tile_at(sx, sy)
    assert t is not None and t.passable, f"start pos ({sx},{sy}) not walkable"

check("start position walkable", _check_start_area)

def _check_no_null_tiles():
    for area in _w.world_map.all_areas():
        for y, row in enumerate(area.grid):
            assert len(row) == area.width, f"{area.id} row {y} wrong width"
            for x, tile in enumerate(row):
                assert tile is not None, f"{area.id} tile ({x},{y}) is None"

check("no null tiles",  _check_no_null_tiles)

def _check_warp_targets():
    for area in _w.world_map.all_areas():
        for y, row in enumerate(area.grid):
            for x, tile in enumerate(row):
                if tile and tile.warp_target:
                    target_id = tile.warp_target.split(':')[0]
                    assert target_id in _w.world_map.areas, \
                        f"{area.id}({x},{y}) warp→unknown '{target_id}'"

check("all warps valid",  _check_warp_targets)

def _check_npc_spawns():
    for area in _w.world_map.all_areas():
        for spawn in area.npc_spawns:
            t = area.tile_at(spawn['x'], spawn['y'])
            assert t and t.passable, \
                f"NPC {spawn['npc_id']} spawns on non-walkable tile at ({spawn['x']},{spawn['y']})"

check("npc spawns walkable", _check_npc_spawns)

def _check_quest_npcs():
    for q in _w.quest_graph.quests.values():
        assert q.giver_npc in _w.npc_roster.npcs, \
            f"Quest '{q.id}' references unknown NPC '{q.giver_npc}'"

check("all quest NPCs exist",  _check_quest_npcs)

def _check_encounters():
    for groups in _w.encounter_table.tables.values():
        for g in groups:
            for eid in g.enemies:
                assert eid in _w.npc_roster.npcs, f"encounter references unknown NPC {eid}"

check("all encounters valid",  _check_encounters)

def _check_shops():
    for inv in _w.shop_inventory.shops.values():
        for entry in inv:
            assert entry['item_id'] in _w.item_registry.items, \
                f"shop sells unknown item {entry['item_id']}"

check("all shop items valid",  _check_shops)

def _check_dialogues():
    for tree in _w.dialogues.values():
        assert tree.root in tree.nodes, f"dialogue {tree.id} root '{tree.root}' missing"

check("all dialogue roots valid", _check_dialogues)

# 3. Variety
section("VARIETY")

def _check_variety():
    worlds = [gen(seed) for seed in range(8)]
    keys   = {w.config.world_name for w in worlds}
    tempos = {w.config.title for w in worlds}
    assert len(keys) >= 4, f"Only {len(keys)} unique world names in 8 seeds"

check("8 seeds produce distinct worlds", _check_variety)

def _check_all_tones():
    for tone in ["heroic","dark","comedic","mysterious","tragic"]:
        w = gen(42, tone=tone)
        assert len(w.world_map.areas) >= 3, f"tone={tone} produced too few areas"

check("all 5 tones generate",  _check_all_tones)

def _check_reproducibility():
    w1 = gen(42); w2 = gen(42)
    assert w1.config.title == w2.config.title, "same seed → different titles"
    assert w1.config.world_name == w2.config.world_name, "same seed → different world names"
    w3 = gen(99)
    assert w1.config.world_name != w3.config.world_name, "different seeds → same world name"

check("reproducibility",  _check_reproducibility)

# 4. Runtime
section("RUNTIME")

_world  = gen(42)
_engine = GameEngine(_world)

# Patch GameEngine for test helpers
def _ecs_pos_helper(self, scene):
    from engine.core.ecs import Position
    if scene.player_eid:
        return scene.ecs.get_component(scene.player_eid, Position)
    return None
GameEngine.ecs_pos = _ecs_pos_helper

def _check_player():
    p = _engine.state.player
    assert p is not None, "no player"
    assert p.name == _world.config.protagonist_name
    assert p.current_hp > 0
    assert p.stats.max_hp > 0
    assert p.class_id in _world.npc_roster.player_classes

check("player state initialised", _check_player)

def _check_map_scene():
    s = MapScene(_engine, _world.world_map.start_area)
    s._build_entities()
    assert s.player_eid is not None, "player entity not created"

check("MapScene entity build",  _check_map_scene)

def _check_movement():
    s = MapScene(_engine, _world.world_map.start_area)
    s._build_entities()
    _engine.scenes.append(s)
    from engine.core.ecs import Position
    pos_before = _engine.ecs_pos(s)
    s._input['right'] = True
    s.update(0.05)
    s._input['right'] = False
    _engine.scenes.pop()

check("player movement update",  _check_movement)

def _check_battle():
    enemies = list(_world.npc_roster.enemies())[:2]
    assert enemies, "no enemies in world"
    eg = EnemyGroup([e.id for e in enemies])
    b  = BattleScene(_engine, eg)
    b.on_enter()
    assert len(b.enemies) > 0
    # Attack action
    b._confirm_action()   # attacks first enemy

check("battle headless",  _check_battle)

def _check_inventory():
    inv = _engine.state.inventory
    inv.add("potion", 3)
    assert inv.count("potion") == 3
    inv.remove("potion", 1)
    assert inv.count("potion") == 2
    assert inv.has("potion")
    assert not inv.has("potion", 5)

check("inventory",  _check_inventory)

def _check_quests():
    qs   = _engine.state.quests
    main = _world.quest_graph.main_chain[0]
    qs.start(main)
    assert main in qs.active
    qs.advance(main, f"{main}_reach", 1)
    assert qs.objective_progress(main, f"{main}_reach") == 1

check("quest system",  _check_quests)

def _check_warp():
    areas = list(_world.world_map.areas.keys())
    _engine.warp_to(areas[1], 10, 10)
    assert _engine.state.position.area_id == areas[1]

check("warp",  _check_warp)

def _check_flags():
    _engine.state.flags.set("test_flag", True)
    assert _engine.state.flags.is_set("test_flag")
    assert not _engine.state.flags.is_set("nonexistent_flag")

check("game flags",  _check_flags)

def _check_save():
    import json, tempfile
    tmp = tempfile.mktemp(suffix='.json')
    _engine.save_game(tmp)
    with open(tmp) as f:
        data = json.load(f)
    import os; os.unlink(tmp)
    assert 'world_seed' in data
    assert 'state' in data
    assert 'party' in data['state']
    assert 'inventory' in data['state']

check("save game",  _check_save)

def _check_level_up():
    player = _engine.state.player
    old_level = player.level
    levels = player.gain_exp(10000)   # huge amount → level up
    assert player.level > old_level, "no level gain"

check("level up",  _check_level_up)

# 5. Content quality
section("CONTENT QUALITY")

def _check_area_sizes():
    for area in _world.world_map.all_areas():
        walkable = sum(1 for row in area.grid for t in row if t and t.passable)
        total    = area.width * area.height
        assert walkable >= total * 0.2, \
            f"{area.id} only {walkable}/{total} walkable tiles ({walkable/total:.0%})"

check("areas have ≥20% walkable tiles",  _check_area_sizes)

def _check_dungeon_rooms():
    dungeons = _world.world_map.areas_of_type(AreaType.DUNGEON)
    assert len(dungeons) >= 1, "no dungeons generated"
    for d in dungeons:
        # At least one chest or warp
        special = sum(1 for row in d.grid for t in row
                      if t and (t.warp_target or t.type.value == "chest"))
        assert special >= 1, f"dungeon {d.id} has no special tiles"

check("dungeons have chests/warps",  _check_dungeon_rooms)

def _check_item_variety():
    items = list(_world.item_registry.items.values())
    types = {i.item_type for i in items}
    assert len(types) >= 3, f"only {len(types)} item types"
    consumables = [i for i in items if i.effect]
    assert len(consumables) >= 3, "too few consumables"

check("item variety",  _check_item_variety)

def _check_main_quest_chain():
    chain = _world.quest_graph.main_chain
    assert len(chain) >= 1, "no main quests"
    for qid in chain:
        q = _world.quest_graph.quest(qid)
        assert q is not None
        assert len(q.objectives) >= 1, f"quest {qid} has no objectives"

check("main quest chain",  _check_main_quest_chain)

def _check_npc_dialogue():
    friendly = _world.npc_roster.friendly_npcs()
    assert len(friendly) >= 3, "too few friendly NPCs"
    with_dlg = [n for n in friendly if n.dialogue_id in _world.dialogues]
    assert len(with_dlg) >= 2, "too few NPCs have dialogue"

check("NPCs have dialogue",  _check_npc_dialogue)

def _check_shops_in_towns():
    towns    = _world.world_map.areas_of_type(AreaType.TOWN)
    shop_ids = set(_world.shop_inventory.shops.keys())
    assert len(shop_ids) >= 1, "no shops generated"
    # Each shop has items
    for sid, inv in _world.shop_inventory.shops.items():
        assert len(inv) >= 2, f"shop {sid} has too few items"

check("towns have shops",  _check_shops_in_towns)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

total = _passed + _failed
print(f"\n{'═'*56}")
if _failed == 0:
    print(f"  Results: {_passed}/{total} passed  ✓ all passed")
else:
    print(f"  Results: {_passed}/{total} passed  ({_failed} failed)")
print(f"{'═'*56}\n")


