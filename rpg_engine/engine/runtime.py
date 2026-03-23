"""
engine.runtime — Main game engine runtime.

Owns:
  - pygame window and main loop
  - System dispatch (input → movement → interaction → render)
  - Scene management (map, battle, dialogue, menu)
  - Asset loading (with placeholder generation for missing assets)

Does NOT own:
  - World generation (that's procgen)
  - Game data structures (that's world.py / state.py)
"""

from __future__ import annotations
import sys
import math
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from .core.world import World, Area, Tile, TileType, NPC, Item, ItemType
from .core.state import GameState, CharacterState, PlayerPosition, Inventory
from .core.ecs   import (
    World_ECS, Position, Velocity, Facing, Sprite, Collider,
    Character, DialogueTrigger, Interactable, Walker, Camera,
)


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (used for placeholders and UI)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "black":    (0,   0,   0),
    "white":    (255, 255, 255),
    "grey":     (128, 128, 128),
    "dark_grey":(40,  40,  40),
    "red":      (200, 60,  60),
    "green":    (60,  180, 60),
    "blue":     (60,  100, 200),
    "yellow":   (220, 200, 60),
    "purple":   (140, 60,  200),
    "cyan":     (60,  200, 200),
    "orange":   (220, 140, 40),
    "brown":    (140, 90,  50),
    "grass":    (80,  160, 60),
    "water":    (60,  100, 200),
    "mountain": (120, 110, 100),
    "forest":   (40,  120, 40),
    "sand":     (210, 190, 130),
    "snow":     (220, 230, 240),
    "lava":     (220, 80,  20),
    "dungeon":  (60,  60,  80),
    "road":     (180, 160, 120),
    "wall":     (100, 90,  80),
}

TILE_COLORS = {
    TileType.GRASS:      PALETTE["grass"],
    TileType.WATER:      PALETTE["water"],
    TileType.MOUNTAIN:   PALETTE["mountain"],
    TileType.FOREST:     PALETTE["forest"],
    TileType.SAND:       PALETTE["sand"],
    TileType.SNOW:       PALETTE["snow"],
    TileType.LAVA:       PALETTE["lava"],
    TileType.DUNGEON:    PALETTE["dungeon"],
    TileType.ROAD:       PALETTE["road"],
    TileType.TOWN_FLOOR: PALETTE["road"],
    TileType.WALL:       PALETTE["wall"],
    TileType.DOOR:       PALETTE["brown"],
    TileType.CHEST:      PALETTE["yellow"],
    TileType.SIGN:       PALETTE["brown"],
    TileType.WARP:       PALETTE["purple"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Asset Manager
# ─────────────────────────────────────────────────────────────────────────────

class AssetManager:
    """
    Loads sprites, music, fonts from the AssetManifest.
    For missing assets, generates colored placeholder surfaces.
    """

    def __init__(self, manifest, tile_size: int = 32):
        self.manifest  = manifest
        self.tile_size = tile_size
        self._sprites:  dict[str, Any] = {}
        self._fonts:    dict[str, Any] = {}
        self._loaded    = False

    def load_all(self, screen) -> None:
        if not PYGAME_AVAILABLE or self._loaded:
            return
        self._screen = screen
        # Load fonts first
        pygame.font.init()
        self._fonts["default"] = pygame.font.SysFont("monospace", 14)
        self._fonts["large"]   = pygame.font.SysFont("monospace", 20)
        self._fonts["title"]   = pygame.font.SysFont("monospace", 28, bold=True)

        # Load or generate sprites
        for key, path in self.manifest.sprites.items():
            self._sprites[key] = self._load_sprite(key, path)

        self._loaded = True

    def _load_sprite(self, key: str, path: str):
        """Load a sprite file, or generate a placeholder if missing."""
        if path:
            try:
                img = pygame.image.load(path).convert_alpha()
                return img
            except Exception:
                pass
        return self._make_placeholder(key)

    def _make_placeholder(self, key: str):
        """Generate a colored tile/character placeholder surface."""
        ts = self.tile_size
        surf = pygame.Surface((ts, ts), pygame.SRCALPHA)

        # Choose color by key prefix
        color = PALETTE["grey"]
        for tile_type in TileType:
            if key.startswith(tile_type.value):
                color = TILE_COLORS.get(tile_type, PALETTE["grey"])
                break
        if "player" in key or "hero" in key:
            color = PALETTE["blue"]
        elif "npc" in key or "villager" in key:
            color = PALETTE["green"]
        elif "enemy" in key or "monster" in key:
            color = PALETTE["red"]
        elif "boss" in key:
            color = PALETTE["purple"]
        elif "chest" in key:
            color = PALETTE["yellow"]

        surf.fill(color)
        # Draw a border so individual tiles are visible
        pygame.draw.rect(surf, tuple(min(255, c+40) for c in color), (0,0,ts,ts), 1)
        # Draw first letter of key
        if PYGAME_AVAILABLE:
            font = pygame.font.SysFont("monospace", max(10, ts//3))
            char = key[0].upper() if key else "?"
            txt  = font.render(char, True, PALETTE["white"])
            surf.blit(txt, (ts//2 - txt.get_width()//2, ts//2 - txt.get_height()//2))
        return surf

    def get_sprite(self, key: str):
        if key in self._sprites:
            return self._sprites[key]
        # Generate on demand if not pre-loaded
        surf = self._make_placeholder(key)
        self._sprites[key] = surf
        return surf

    def get_font(self, size: str = "default"):
        return self._fonts.get(size, self._fonts.get("default"))


# ─────────────────────────────────────────────────────────────────────────────
# Scenes
# ─────────────────────────────────────────────────────────────────────────────

class Scene:
    """Base class for all game scenes."""
    def __init__(self, engine: "GameEngine"):
        self.engine = engine

    def on_enter(self): pass
    def on_exit(self):  pass
    def update(self, dt: float): pass
    def render(self, screen): pass
    def handle_event(self, event): pass


class MapScene(Scene):
    """
    The overworld/town/dungeon map scene.
    Handles tile rendering, player movement, NPC movement, interactions.
    """

    MOVE_SPEED = 3.0   # tiles per second

    def __init__(self, engine: "GameEngine", area_id: str):
        super().__init__(engine)
        self.area_id  = area_id
        self.ecs      = World_ECS()
        self.player_eid = None
        self.camera_x   = 0.0
        self.camera_y   = 0.0
        self._input     = {"up": False, "down": False, "left": False, "right": False}
        self._interact_cooldown = 0.0
        self._step_timer = 0.0

    def on_enter(self):
        self._build_entities()

    def _build_entities(self):
        """Populate ECS from world data for the current area."""
        world  = self.engine.world
        state  = self.engine.state
        area   = world.world_map.area(self.area_id)
        ts     = world.config.tile_size
        if not area:
            return

        self.ecs = World_ECS()

        # Player entity
        px, py = state.position.x, state.position.y
        self.player_eid = self.ecs.create(
            Position(px * ts, py * ts, self.area_id),
            Velocity(),
            Facing(state.position.facing),
            Sprite("player", anim="idle", z_order=5),
            Collider(ts - 4, ts - 4, 2, 2),
            Character("player", state.player.name if state.player else "Hero",
                      is_player=True),
            tags=["player"],
        )

        # Camera
        self.ecs.create(Camera(target_entity=self.player_eid), tags=["camera"])

        # NPC entities
        for spawn in area.npc_spawns:
            npc = world.npc_roster.npc(spawn["npc_id"])
            if not npc:
                continue
            components = [
                Position(spawn["x"] * ts, spawn["y"] * ts, self.area_id),
                Velocity(),
                Facing("down"),
                Sprite(npc.sprite_key or "npc", z_order=4),
                Collider(ts - 4, ts - 4, 2, 2, solid=not npc.is_enemy),
                Character(npc.id, npc.name),
                Interactable("npc", {"npc_id": npc.id}),
            ]
            if npc.dialogue_id:
                components.append(DialogueTrigger(npc.dialogue_id, npc.id))
            self.ecs.create(*components, tags=["npc", npc.id])

        # Chest entities
        for chest in area.chest_spawns:
            key = state.chest_key(self.area_id, chest["x"], chest["y"])
            opened = key in state.opened_chests
            sprite_key = "chest_open" if opened else "chest"
            self.ecs.create(
                Position(chest["x"] * ts, chest["y"] * ts, self.area_id),
                Sprite(sprite_key, z_order=3),
                Interactable("chest", {"item_id": chest["item_id"],
                                        "area_id": self.area_id,
                                        "x": chest["x"], "y": chest["y"]}),
                tags=["chest"],
            )

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type == pygame.KEYDOWN:
            key_map = {
                pygame.K_UP: "up", pygame.K_w: "up",
                pygame.K_DOWN: "down", pygame.K_s: "down",
                pygame.K_LEFT: "left", pygame.K_a: "left",
                pygame.K_RIGHT: "right", pygame.K_d: "right",
            }
            if event.key in key_map:
                self._input[key_map[event.key]] = True
            elif event.key in (pygame.K_RETURN, pygame.K_z, pygame.K_SPACE):
                self._try_interact()
            elif event.key == pygame.K_ESCAPE:
                self.engine.push_scene(MenuScene(self.engine))

        elif event.type == pygame.KEYUP:
            key_map = {
                pygame.K_UP: "up", pygame.K_w: "up",
                pygame.K_DOWN: "down", pygame.K_s: "down",
                pygame.K_LEFT: "left", pygame.K_a: "left",
                pygame.K_RIGHT: "right", pygame.K_d: "right",
            }
            if event.key in key_map:
                self._input[key_map[event.key]] = False

    def update(self, dt: float):
        if not self.player_eid:
            return
        self._interact_cooldown = max(0, self._interact_cooldown - dt)
        self._update_player(dt)
        self._update_walkers(dt)
        self._update_camera(dt)
        self._check_tile_events()
        self._check_encounters(dt)

    def _update_player(self, dt: float):
        pos  = self.ecs.get_component(self.player_eid, Position)
        face = self.ecs.get_component(self.player_eid, Facing)
        spr  = self.ecs.get_component(self.player_eid, Sprite)
        if not pos or not face:
            return

        world = self.engine.world
        area  = world.world_map.area(self.area_id)
        ts    = world.config.tile_size
        speed = self.MOVE_SPEED * ts   # pixels/sec

        dx = dy = 0.0
        if self._input["up"]:    dy = -speed * dt; face.direction = "up"
        if self._input["down"]:  dy =  speed * dt; face.direction = "down"
        if self._input["left"]:  dx = -speed * dt; face.direction = "left"
        if self._input["right"]: dx =  speed * dt; face.direction = "right"

        moving = dx != 0 or dy != 0

        # Collision check
        if area:
            new_x = pos.x + dx
            new_y = pos.y + dy
            col   = self.ecs.get_component(self.player_eid, Collider)
            cx = int((new_x + (col.offset_x if col else 0)) / ts)
            cy = int((new_y + (col.offset_y if col else 0)) / ts)
            tile = area.tile_at(cx, cy)
            if tile and tile.passable:
                pos.x = new_x
                pos.y = new_y
            elif dx != 0 and dy == 0:
                pass   # blocked
            elif dy != 0 and dx == 0:
                pass
            else:
                # Try X only
                new_x2 = pos.x + dx
                cx2 = int(new_x2 / ts)
                tile2 = area.tile_at(cx2, int(pos.y / ts))
                if tile2 and tile2.passable:
                    pos.x = new_x2

        # Animate
        if spr:
            spr.anim = "walk" if moving else "idle"
            if moving:
                self._step_timer += dt

        # Sync state position
        self.engine.state.position.x = int(pos.x / ts)
        self.engine.state.position.y = int(pos.y / ts)
        self.engine.state.position.facing = face.direction

    def _update_walkers(self, dt: float):
        """NPC idle wander and patrol-path movement."""
        area = self.engine.world.world_map.area(self.area_id)
        ts   = self.engine.world.config.tile_size
        if not area:
            return
        import math as _m
        for eid, pos, walker in self.ecs.query(Position, Walker):
            if not walker.path:
                # Idle wander: wait, then step to a random adjacent tile
                walker.wait_timer += dt
                if walker.wait_timer < 2.0:
                    continue
                walker.wait_timer = 0.0
                import random as _r
                for _ in range(8):
                    dx = _r.choice([-1, 0, 0, 1])
                    dy = _r.choice([-1, 0, 0, 1])
                    nx = int(pos.x / ts) + dx
                    ny = int(pos.y / ts) + dy
                    t  = area.tile_at(nx, ny)
                    if t and t.passable and t.warp_target is None:
                        pos.x = nx * ts
                        pos.y = ny * ts
                        # Update facing direction
                        face = self.ecs.get_component(eid, Facing)
                        if face:
                            if dx > 0: face.direction = "right"
                            elif dx < 0: face.direction = "left"
                            elif dy > 0: face.direction = "down"
                            elif dy < 0: face.direction = "up"
                        break
                continue

            # Follow patrol path
            tx = walker.path[walker.path_idx][0] * ts
            ty = walker.path[walker.path_idx][1] * ts
            ddx = tx - pos.x
            ddy = ty - pos.y
            dist = _m.hypot(ddx, ddy)
            if dist < 2.0:
                pos.x = tx; pos.y = ty
                walker.wait_timer += dt
                if walker.wait_timer >= 0.6:
                    walker.wait_timer = 0.0
                    walker.path_idx = (walker.path_idx + 1) % len(walker.path)
            else:
                spd = walker.speed * ts * dt
                pos.x += (ddx / dist) * spd
                pos.y += (ddy / dist) * spd

    def _update_camera(self, dt: float):
        for eid, cam in self.ecs.query(Camera):
            if cam.target_entity is None:
                continue
            target_pos = self.ecs.get_component(cam.target_entity, Position)
            if not target_pos:
                continue
            config = self.engine.world.config
            target_x = target_pos.x - config.screen_width  / 2
            target_y = target_pos.y - config.screen_height / 2
            self.camera_x += (target_x - self.camera_x) * min(1.0, cam.smooth * dt)
            self.camera_y += (target_y - self.camera_y) * min(1.0, cam.smooth * dt)
            # Clamp to map boundaries
            area = self.engine.world.world_map.area(self.area_id)
            if area:
                ts = self.engine.world.config.tile_size
                max_cx = area.width  * ts - config.screen_width
                max_cy = area.height * ts - config.screen_height
                self.camera_x = max(0, min(self.camera_x, max_cx))
                self.camera_y = max(0, min(self.camera_y, max_cy))

    def _check_tile_events(self):
        """Check if player is standing on a special tile (warp, etc.)."""
        if not self.player_eid:
            return
        pos  = self.ecs.get_component(self.player_eid, Position)
        area = self.engine.world.world_map.area(self.area_id)
        if not pos or not area:
            return
        ts   = self.engine.world.config.tile_size
        tile = area.tile_at(int(pos.x / ts), int(pos.y / ts))
        if tile and tile.warp_target and tile.type == TileType.WARP:
            parts = tile.warp_target.split(":")
            if len(parts) >= 3:
                self.engine.warp_to(parts[0], int(parts[1]), int(parts[2]))

    def _check_encounters(self, dt: float):
        """Random encounter check when moving in encounter zones."""
        if not self.player_eid:
            return
        pos  = self.ecs.get_component(self.player_eid, Position)
        if not pos:
            return
        area = self.engine.world.world_map.area(self.area_id)
        ts   = self.engine.world.config.tile_size
        if not area:
            return
        tile = area.tile_at(int(pos.x / ts), int(pos.y / ts))
        if not tile or not tile.encounter:
            return
        moving = any(self._input.values())
        if not moving:
            return

        self.engine.state.encounter_steps += 1
        config = self.engine.world.config
        rate = config.encounter_rate

        import random
        if random.random() < rate * dt * 2:
            import procgen.seeds as _s
            rng = _s.Seed(self.engine.state.step_count).rng()
            group = self.engine.world.encounter_table.random_encounter(self.area_id, rng)
            if group:
                battle = BattleScene(self.engine, group)
                self.engine.push_scene(battle)

    def _try_interact(self):
        """Check for interactable entity in facing direction."""
        if not self.player_eid or self._interact_cooldown > 0:
            return
        pos  = self.ecs.get_component(self.player_eid, Position)
        face = self.ecs.get_component(self.player_eid, Facing)
        if not pos or not face:
            return

        ts = self.engine.world.config.tile_size
        dx, dy = {"up": (0,-1), "down": (0,1), "left": (-1,0), "right": (1,0)}[face.direction]
        target_px = pos.x + dx * ts
        target_py = pos.y + dy * ts

        for eid, inter_pos, inter in self.ecs.query(Position, Interactable):
            if eid == self.player_eid:
                continue
            if abs(inter_pos.x - target_px) < ts and abs(inter_pos.y - target_py) < ts:
                self._handle_interaction(eid, inter)
                self._interact_cooldown = 0.5
                return

    def _handle_interaction(self, entity_id: int, inter: Interactable):
        if inter.type == "npc":
            npc_id = inter.payload.get("npc_id")
            npc = self.engine.world.npc_roster.npc(npc_id)
            if npc and npc.dialogue_id:
                dlg_tree = self.engine.world.dialogues.get(npc.dialogue_id)
                if dlg_tree:
                    self.engine.push_scene(
                        DialogueScene(self.engine, dlg_tree, npc)
                    )
            if npc and npc.is_shopkeeper:
                self.engine.push_scene(ShopScene(self.engine, npc.shop_id))

        elif inter.type == "chest":
            area_id = inter.payload.get("area_id", self.area_id)
            cx, cy  = inter.payload.get("x", 0), inter.payload.get("y", 0)
            if self.engine.state.open_chest(area_id, cx, cy):
                item_id = inter.payload.get("item_id")
                if item_id:
                    self.engine.state.inventory.add(item_id)
                    item = self.engine.world.item_registry.item(item_id)
                    msg = f"Found {item.name}!" if item else "Found an item!"
                    self.engine.push_scene(MessageScene(self.engine, msg))
                # Update sprite
                spr = self.ecs.get_component(entity_id, Sprite)
                if spr:
                    spr.key = "chest_open"
                    # Check quest item collection
                    self._check_collect_objectives(inter.payload.get("item_id",""))

    def _check_collect_objectives(self, item_id: str) -> None:
        if not item_id:
            return
        state = self.engine.state
        wq    = self.engine.world.quest_graph
        # Auto-start first main quest if not yet started
        main = wq.main_chain[0] if wq.main_chain else None
        if main and main not in state.quests.active and main not in state.quests.completed:
            state.quests.start(main)
        for q_id in list(state.quests.active):
            quest = wq.quest(q_id)
            if not quest:
                continue
            for obj in quest.objectives:
                if obj.type == "collect_item" and obj.target_id == item_id:
                    needed  = obj.target_count
                    current = state.quests.objective_progress(q_id, obj.id)
                    if state.inventory.count(item_id) >= needed and current < needed:
                        state.quests.advance(q_id, obj.id, needed - current)
            self._check_quest_completion_map(q_id)

    def _check_quest_completion_map(self, quest_id: str) -> None:
        quest = self.engine.world.quest_graph.quest(quest_id)
        state = self.engine.state
        if not quest or quest_id in state.quests.completed:
            return
        for obj in quest.objectives:
            if obj.optional:
                continue
            if state.quests.objective_progress(quest_id, obj.id) < obj.target_count:
                return
        # All objectives met
        state.quests.complete(quest_id)
        state.inventory.gold += quest.reward_gold
        for char in state.party:
            char.gain_exp(quest.reward_exp)
        for item_id in quest.reward_items:
            state.inventory.add(item_id)
        for unlock_id in quest.reward_unlock:
            unlocked = self.engine.world.quest_graph.quest(unlock_id)
            if unlocked and unlock_id not in state.quests.active:
                state.quests.start(unlock_id)
        self.engine.push_scene(
            MessageScene(self.engine, f"Quest complete: {quest.name}!", duration=2.5)
        )

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        area   = self.engine.world.world_map.area(self.area_id)
        ts     = config.tile_size
        assets = self.engine.assets

        screen.fill(PALETTE["black"])

        if not area:
            return

        # Render tiles
        cx, cy = int(self.camera_x), int(self.camera_y)
        start_tx = max(0, cx // ts)
        start_ty = max(0, cy // ts)
        end_tx   = min(area.width,  start_tx + config.screen_width  // ts + 2)
        end_ty   = min(area.height, start_ty + config.screen_height // ts + 2)

        for ty in range(start_ty, end_ty):
            for tx in range(start_tx, end_tx):
                tile = area.tile_at(tx, ty)
                if not tile:
                    continue
                sx = tx * ts - cx
                sy = ty * ts - cy
                sprite = assets.get_sprite(tile.sprite_key)
                if sprite:
                    screen.blit(sprite, (sx, sy))
                else:
                    color = TILE_COLORS.get(tile.type, PALETTE["grey"])
                    pygame.draw.rect(screen, color, (sx, sy, ts, ts))
                    pygame.draw.rect(screen, PALETTE["dark_grey"], (sx, sy, ts, ts), 1)

        # Render entities (sorted by y for depth)
        entities_to_draw = []
        for eid, pos, spr in self.ecs.query(Position, Sprite):
            if not spr.visible:
                continue
            sx = pos.x - cx + spr.offset_x
            sy = pos.y - cy + spr.offset_y
            if -ts <= sx <= config.screen_width + ts and -ts <= sy <= config.screen_height + ts:
                entities_to_draw.append((spr.z_order, pos.y, eid, sx, sy, spr))

        entities_to_draw.sort(key=lambda e: (e[0], e[1]))
        for _, _, eid, sx, sy, spr in entities_to_draw:
            sprite_surf = assets.get_sprite(spr.key)
            if sprite_surf:
                if spr.flip_x:
                    sprite_surf = pygame.transform.flip(sprite_surf, True, False)
                screen.blit(sprite_surf, (sx, sy))

        # HUD
        self._render_hud(screen)

    def _render_hud(self, screen):
        if not self.engine.state.player:
            return
        player = self.engine.state.player
        font   = self.engine.assets.get_font("default")
        config = self.engine.world.config

        # Minibar at bottom
        bar_h  = 48
        bar_y  = config.screen_height - bar_h
        pygame.draw.rect(screen, PALETTE["dark_grey"],
                         (0, bar_y, config.screen_width, bar_h))
        pygame.draw.line(screen, PALETTE["grey"],
                         (0, bar_y), (config.screen_width, bar_y))

        # HP bar
        hp_pct = player.current_hp / max(1, player.stats.max_hp)
        hp_w   = 120
        pygame.draw.rect(screen, PALETTE["dark_grey"], (8, bar_y + 8, hp_w, 12))
        pygame.draw.rect(screen, PALETTE["red"], (8, bar_y + 8, int(hp_w * hp_pct), 12))
        hp_txt = font.render(f"HP {player.current_hp}/{player.stats.max_hp}", True, PALETTE["white"])
        screen.blit(hp_txt, (8, bar_y + 24))

        # MP bar
        mp_pct = player.current_mp / max(1, player.stats.max_mp)
        mp_w   = 80
        pygame.draw.rect(screen, PALETTE["dark_grey"], (140, bar_y + 8, mp_w, 12))
        pygame.draw.rect(screen, PALETTE["blue"], (140, bar_y + 8, int(mp_w * mp_pct), 12))
        mp_txt = font.render(f"MP {player.current_mp}/{player.stats.max_mp}", True, PALETTE["white"])
        screen.blit(mp_txt, (140, bar_y + 24))

        # Level / Gold
        info = font.render(
            f"Lv{player.level}  G{self.engine.state.inventory.gold}",
            True, PALETTE["yellow"]
        )
        screen.blit(info, (config.screen_width - info.get_width() - 8, bar_y + 12))

        # Area name
        area = self.engine.world.world_map.area(self.area_id)
        if area:
            area_txt = font.render(area.name, True, PALETTE["white"])
            screen.blit(area_txt, (config.screen_width//2 - area_txt.get_width()//2, bar_y + 12))


class BattleScene(Scene):
    """
    Turn-based battle scene.
    Supports: Attack, Skills, Items, Run.
    """

    def __init__(self, engine: "GameEngine", enemy_group):
        super().__init__(engine)
        self.enemy_group = enemy_group
        self.enemies:     list[CharacterState] = []
        self.turn_order:  list[str] = []    # "player_0", "enemy_0", etc.
        self.current_turn: int = 0
        self.phase:       str  = "player_choice"  # player_choice, enemy_turn, animation, result
        self.selected_action: int = 0
        self.selected_target: int = 0
        self.actions      = ["Attack", "Skills", "Items", "Run"]
        self.log:         list[str] = []
        self.result:      str | None = None  # "win", "lose", "escape"
        self._phase_timer: float = 0.0

    def on_enter(self):
        world  = self.engine.world
        roster = world.npc_roster
        for npc_id in self.enemy_group.enemies:
            npc = roster.npc(npc_id)
            if npc and npc.stats:
                from .core.state import CharacterState, Stats
                import copy
                enemy = CharacterState(
                    npc_id    = npc.id,
                    name      = npc.name,
                    class_id  = "enemy",
                    level     = max(1, self.engine.state.player.level if self.engine.state.player else 1),
                    current_hp = npc.stats.hp,
                    current_mp = npc.stats.mp,
                    stats     = copy.copy(npc.stats),
                    skills    = list(npc.skills),
                )
                self.enemies.append(enemy)
        self.log.append(f"Encountered {', '.join(e.name for e in self.enemies)}!")

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if self.phase != "player_choice":
            return
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.selected_action = (self.selected_action - 1) % len(self.actions)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.selected_action = (self.selected_action + 1) % len(self.actions)
            elif event.key in (pygame.K_RETURN, pygame.K_z, pygame.K_SPACE):
                self._confirm_action()

    def _confirm_action(self):
        action = self.actions[self.selected_action]
        player = self.engine.state.player
        if not player:
            return

        if action == "Attack":
            if self.enemies:
                target = self.enemies[0]
                dmg = max(1, player.stats.attack - target.stats.defense + 5)
                target.current_hp -= dmg
                self.log.append(f"{player.name} attacks {target.name} for {dmg} damage!")
                if target.current_hp <= 0:
                    self.log.append(f"{target.name} is defeated!")
                    self.enemies.remove(target)
            if not self.enemies:
                self._win()
            else:
                self.phase = "enemy_turn"

        elif action == "Run":
            import random
            if random.random() < 0.5:
                self.log.append("Escaped!")
                self.result = "escape"
                self._phase_timer = 1.5
                self.phase = "result"
            else:
                self.log.append("Couldn't escape!")
                self.phase = "enemy_turn"

        elif action == "Skills":
            self.engine.push_scene(SkillMenuScene(self.engine, self))
            return

        elif action == "Items":
            self.engine.push_scene(ItemMenuScene(self.engine, self))
            return

    def _win(self):
        player = self.engine.state.player
        base_exp  = sum(50 * max(1, e.level) for e in self.enemies)
        base_gold = sum(20 * max(1, e.level) for e in self.enemies)
        self.engine.state.inventory.gold += base_gold
        levels = []
        if player:
            levels = player.gain_exp(base_exp)
        self.log.append(f"Victory! Gained {base_exp} EXP, {base_gold} Gold.")
        if levels:
            self.log.append(f"{player.name} reached level {max(levels)}!")
        self.result = "win"
        self._phase_timer = 2.0
        self.phase = "result"

    def update(self, dt: float):
        if self.phase == "enemy_turn":
            self._enemy_turn()
            self.phase = "player_choice"
        elif self.phase == "result":
            self._phase_timer -= dt
            if self._phase_timer <= 0:
                self.engine.pop_scene()

    def _enemy_turn(self):
        player = self.engine.state.player
        if not player:
            return
        for enemy in self.enemies:
            if not enemy.is_alive:
                continue
            dmg = max(1, enemy.stats.attack - player.stats.defense)
            player.current_hp = max(0, player.current_hp - dmg)
            self.log.append(f"{enemy.name} attacks for {dmg} damage!")
            if not player.is_alive:
                self.log.append("You were defeated...")
                self.result = "lose"
                self._phase_timer = 2.0
                self.phase = "result"
                return

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")
        ts     = config.tile_size

        # Battle background
        screen.fill((20, 15, 30))
        pygame.draw.rect(screen, (30, 25, 45), (0, 0, config.screen_width, config.screen_height // 2))

        # Enemy sprites
        for i, enemy in enumerate(self.enemies):
            ex = config.screen_width // 2 - 40 + i * 90
            ey = config.screen_height // 4 - ts // 2
            spr = assets.get_sprite(f"enemy_{enemy.npc_id}" if f"enemy_{enemy.npc_id}" in assets._sprites else "enemy")
            screen.blit(spr, (ex, ey))
            # Enemy HP
            hp_pct = enemy.current_hp / max(1, enemy.stats.max_hp)
            pygame.draw.rect(screen, PALETTE["dark_grey"], (ex, ey - 14, ts, 8))
            pygame.draw.rect(screen, PALETTE["red"], (ex, ey - 14, int(ts * hp_pct), 8))
            name_txt = font.render(enemy.name, True, PALETTE["white"])
            screen.blit(name_txt, (ex + ts//2 - name_txt.get_width()//2, ey + ts + 2))

        # Player sprite
        player = self.engine.state.player
        if player:
            px = config.screen_width // 4
            py = config.screen_height // 2
            spr = assets.get_sprite("player")
            screen.blit(spr, (px - ts//2, py - ts//2))

        # Action menu
        menu_x = config.screen_width - 160
        menu_y = config.screen_height // 2
        pygame.draw.rect(screen, PALETTE["dark_grey"], (menu_x-4, menu_y-4, 156, 100))
        pygame.draw.rect(screen, PALETTE["grey"], (menu_x-4, menu_y-4, 156, 100), 1)
        for i, action in enumerate(self.actions):
            color = PALETTE["yellow"] if i == self.selected_action else PALETTE["white"]
            txt = font.render(("▶ " if i == self.selected_action else "  ") + action,
                               True, color)
            screen.blit(txt, (menu_x, menu_y + i * 22))

        # Battle log
        log_y = config.screen_height * 3 // 4
        pygame.draw.rect(screen, PALETTE["dark_grey"],
                         (0, log_y, config.screen_width, config.screen_height - log_y))
        pygame.draw.line(screen, PALETTE["grey"],
                         (0, log_y), (config.screen_width, log_y))
        for i, line in enumerate(self.log[-3:]):
            txt = font.render(line, True, PALETTE["white"])
            screen.blit(txt, (8, log_y + 6 + i * 18))

        # Party stats
        if player:
            hp_txt = large.render(
                f"{player.name}  HP:{player.current_hp}/{player.stats.max_hp}  "
                f"MP:{player.current_mp}/{player.stats.max_mp}",
                True, PALETTE["white"]
            )
            screen.blit(hp_txt, (8, config.screen_height // 2 + 8))


class DialogueScene(Scene):
    """Branching dialogue scene."""

    def __init__(self, engine: "GameEngine", tree, npc):
        super().__init__(engine)
        self.tree    = tree
        self.npc     = npc
        self.current = tree.root_node()
        self.selected_choice = 0
        self._advance_timer  = 0.0

    def handle_event(self, event):
        if not PYGAME_AVAILABLE or not self.current:
            return
        if event.type != pygame.KEYDOWN:
            return

        from .core.world import DialogueNodeType
        if event.key in (pygame.K_RETURN, pygame.K_z, pygame.K_SPACE):
            if self.current.type == DialogueNodeType.TEXT:
                if self.current.next_id:
                    self.current = self.tree.node(self.current.next_id)
                else:
                    self.engine.pop_scene()
            elif self.current.type == DialogueNodeType.CHOICE:
                choice = self.current.choices[self.selected_choice]
                self.current = self.tree.node(choice.next_id)
                if not self.current:
                    self.engine.pop_scene()
            elif self.current.type == DialogueNodeType.SET_FLAG:
                self.engine.state.flags.set(self.current.flag_key, self.current.flag_value)
                if self.current.next_id:
                    self.current = self.tree.node(self.current.next_id)
                else:
                    self.engine.pop_scene()
            else:
                self.engine.pop_scene()

        elif event.key in (pygame.K_UP, pygame.K_w):
            from .core.world import DialogueNodeType
            if self.current.type == DialogueNodeType.CHOICE:
                self.selected_choice = (self.selected_choice - 1) % len(self.current.choices)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            from .core.world import DialogueNodeType
            if self.current.type == DialogueNodeType.CHOICE:
                self.selected_choice = (self.selected_choice + 1) % len(self.current.choices)

    def _check_recruit(self):
        """If this NPC can join the party and conditions are met, recruit them."""
        if not self.npc:
            return
        roster = self.engine.world.npc_roster
        state  = self.engine.state
        config = self.engine.world.config
        if self.npc.id not in roster.party_members:
            return
        if any(c.npc_id == self.npc.id for c in state.party):
            return
        if len(state.party) >= config.party_size:
            return
        unlock_flag = f"recruit_{self.npc.id}"
        if state.flags.is_set(unlock_flag) or len(state.party) < 2:
            pclass_id = self.npc.metadata.get("class_id", config.protagonist_class)
            pclass    = roster.player_classes.get(pclass_id)
            if pclass:
                from engine.core.state import CharacterState
                member = CharacterState(
                    npc_id   = self.npc.id,
                    name     = self.npc.name,
                    class_id = pclass_id,
                    level    = max(1, state.player.level if state.player else 1),
                    current_hp = pclass.base_stats.hp,
                    current_mp = pclass.base_stats.mp,
                    stats    = pclass.base_stats,
                    skills   = [s.id for s in pclass.skills if s.learn_level <= 1],
                )
                state.party.append(member)
                self.engine.push_scene(
                    MessageScene(self.engine,
                                 f"{self.npc.name} joined the party!", duration=2.0)
                )

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")

        # Dim background
        overlay = pygame.Surface((config.screen_width, config.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        # Dialogue box
        box_h   = 140
        box_y   = config.screen_height - box_h - 8
        box_rect = (8, box_y, config.screen_width - 16, box_h)
        pygame.draw.rect(screen, PALETTE["dark_grey"], box_rect)
        pygame.draw.rect(screen, PALETTE["white"], box_rect, 2)

        if not self.current:
            return

        # Speaker name
        speaker = self.current.speaker or (self.npc.name if self.npc else "")
        if speaker:
            name_txt = large.render(speaker, True, PALETTE["yellow"])
            screen.blit(name_txt, (16, box_y + 8))

        # Dialogue text (word-wrapped)
        text   = self.current.text
        words  = text.split()
        lines  = []
        line   = ""
        max_w  = config.screen_width - 40
        for word in words:
            test = (line + " " + word).strip()
            if font.size(test)[0] <= max_w:
                line = test
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)

        text_y = box_y + 32
        for i, ln in enumerate(lines[:4]):
            txt = font.render(ln, True, PALETTE["white"])
            screen.blit(txt, (16, text_y + i * 20))

        # Choices
        from .core.world import DialogueNodeType
        if self.current.type == DialogueNodeType.CHOICE:
            for i, choice in enumerate(self.current.choices):
                color = PALETTE["yellow"] if i == self.selected_choice else PALETTE["grey"]
                txt = font.render(
                    ("▶ " if i == self.selected_choice else "  ") + choice.text,
                    True, color
                )
                screen.blit(txt, (32, text_y + (len(lines) + i + 1) * 20))
        else:
            # Continue prompt
            prompt = font.render("▶", True, PALETTE["grey"])
            screen.blit(prompt, (config.screen_width - 24, box_y + box_h - 20))


class ShopScene(Scene):
    """Item shop."""

    def __init__(self, engine: "GameEngine", shop_id: str):
        super().__init__(engine)
        self.shop_id  = shop_id
        self.items    = []
        self.selected = 0
        self.mode     = "buy"   # "buy" or "sell"

    def on_enter(self):
        world     = self.engine.world
        inventory = world.shop_inventory.shop(self.shop_id)
        for entry in inventory:
            item = world.item_registry.item(entry["item_id"])
            if item:
                price = int(item.price * entry.get("price_mult", 1.0))
                self.items.append({"item": item, "price": price, "stock": entry.get("stock", 99)})

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % max(1, len(self.items))
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % max(1, len(self.items))
        elif event.key in (pygame.K_RETURN, pygame.K_z):
            self._buy()
        elif event.key == pygame.K_ESCAPE:
            self.engine.pop_scene()

    def _buy(self):
        if not self.items:
            return
        entry = self.items[self.selected]
        price = entry["price"]
        inv   = self.engine.state.inventory
        if inv.gold >= price:
            inv.gold -= price
            inv.add(entry["item"].id)
            self.engine.push_scene(
                MessageScene(self.engine, f"Bought {entry['item'].name}!")
            )

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")

        screen.fill(PALETTE["dark_grey"])
        title = large.render("SHOP", True, PALETTE["yellow"])
        screen.blit(title, (config.screen_width // 2 - title.get_width() // 2, 16))

        gold_txt = font.render(f"Gold: {self.engine.state.inventory.gold}", True, PALETTE["yellow"])
        screen.blit(gold_txt, (config.screen_width - gold_txt.get_width() - 16, 16))

        for i, entry in enumerate(self.items):
            color = PALETTE["yellow"] if i == self.selected else PALETTE["white"]
            row = font.render(
                f"{'▶' if i == self.selected else ' '} {entry['item'].name:<20} {entry['price']:>5}G",
                True, color
            )
            screen.blit(row, (32, 60 + i * 24))

        if self.items and self.selected < len(self.items):
            desc = font.render(self.items[self.selected]["item"].description, True, PALETTE["grey"])
            screen.blit(desc, (32, config.screen_height - 60))

        esc_txt = font.render("ESC: Leave", True, PALETTE["grey"])
        screen.blit(esc_txt, (32, config.screen_height - 32))


class MessageScene(Scene):
    """Simple message popup — auto-dismisses or on keypress."""

    def __init__(self, engine: "GameEngine", message: str, duration: float = 0.0):
        super().__init__(engine)
        self.message  = message
        self.duration = duration
        self._timer   = duration

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_z, pygame.K_SPACE, pygame.K_ESCAPE):
                self.engine.pop_scene()

    def update(self, dt: float):
        if self.duration > 0:
            self._timer -= dt
            if self._timer <= 0:
                self.engine.pop_scene()

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        w, h   = 360, 80
        x = config.screen_width  // 2 - w // 2
        y = config.screen_height // 2 - h // 2
        pygame.draw.rect(screen, PALETTE["dark_grey"], (x, y, w, h))
        pygame.draw.rect(screen, PALETTE["white"], (x, y, w, h), 2)
        txt = font.render(self.message, True, PALETTE["white"])
        screen.blit(txt, (x + w // 2 - txt.get_width() // 2,
                           y + h // 2 - txt.get_height() // 2))


class MenuScene(Scene):
    """In-game pause menu."""

    def __init__(self, engine: "GameEngine"):
        super().__init__(engine)
        self.options  = ["Status", "Inventory", "Quests", "Save", "Close"]
        self.selected = 0

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(self.options)
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(self.options)
        elif event.key in (pygame.K_RETURN, pygame.K_z):
            self._select()
        elif event.key == pygame.K_ESCAPE:
            self.engine.pop_scene()

    def _select(self):
        opt = self.options[self.selected]
        if opt == "Close":
            self.engine.pop_scene()
        elif opt == "Save":
            self.engine.save_game("save_01.json")
            self.engine.push_scene(MessageScene(self.engine, "Game saved!", duration=1.5))
        elif opt == "Status":
            self.engine.push_scene(StatusScene(self.engine))
        elif opt == "Inventory":
            self.engine.push_scene(InventoryScene(self.engine))
        elif opt == "Quests":
            self.engine.push_scene(QuestLogScene(self.engine))

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")
        overlay = pygame.Surface((config.screen_width, config.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        w, h = 200, len(self.options) * 36 + 48
        x = config.screen_width  // 2 - w // 2
        y = config.screen_height // 2 - h // 2
        pygame.draw.rect(screen, PALETTE["dark_grey"], (x, y, w, h))
        pygame.draw.rect(screen, PALETTE["white"], (x, y, w, h), 2)
        title = large.render("MENU", True, PALETTE["yellow"])
        screen.blit(title, (x + w//2 - title.get_width()//2, y + 8))
        for i, opt in enumerate(self.options):
            color = PALETTE["yellow"] if i == self.selected else PALETTE["white"]
            txt = font.render(("▶ " if i == self.selected else "  ") + opt, True, color)
            screen.blit(txt, (x + 16, y + 40 + i * 36))


class StatusScene(Scene):
    """Party status screen."""

    def handle_event(self, event):
        if PYGAME_AVAILABLE and event.type == pygame.KEYDOWN:
            self.engine.pop_scene()

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")
        screen.fill(PALETTE["dark_grey"])
        title = large.render("STATUS", True, PALETTE["yellow"])
        screen.blit(title, (16, 16))
        for i, char in enumerate(self.engine.state.party):
            y = 60 + i * 100
            pygame.draw.rect(screen, (30, 30, 50), (8, y, config.screen_width - 16, 90))
            pygame.draw.rect(screen, PALETTE["grey"], (8, y, config.screen_width - 16, 90), 1)
            # Portrait placeholder
            spr = assets.get_sprite(f"portrait_{char.class_id}")
            if spr:
                screen.blit(pygame.transform.scale(spr, (64, 64)), (10, y + 13))
            name_txt = large.render(f"{char.name}  Lv.{char.level}  [{char.class_id}]",
                                    True, PALETTE["yellow"])
            screen.blit(name_txt, (82, y + 6))
            stats_txt = font.render(
                f"HP:{char.current_hp}/{char.stats.max_hp}  "
                f"MP:{char.current_mp}/{char.stats.max_mp}  "
                f"ATK:{char.stats.attack}  DEF:{char.stats.defense}  "
                f"MAG:{char.stats.magic}  SPD:{char.stats.speed}",
                True, PALETTE["white"]
            )
            screen.blit(stats_txt, (82, y + 34))
            # HP/MP bars
            for bi, (val, maxv, col) in enumerate([
                (char.current_hp, char.stats.max_hp, PALETTE["red"]),
                (char.current_mp, char.stats.max_mp, PALETTE["blue"]),
            ]):
                bw = 180
                pct = val / max(1, maxv)
                bx  = 82 + bi * (bw + 10)
                pygame.draw.rect(screen, PALETTE["dark_grey"], (bx, y + 54, bw, 8))
                pygame.draw.rect(screen, col, (bx, y + 54, int(bw * pct), 8))
            exp_txt = font.render(f"EXP: {char.exp}/{char.exp_to_next}", True, PALETTE["cyan"])
            screen.blit(exp_txt, (82, y + 68))
        close_txt = font.render("Press any key to close", True, PALETTE["grey"])
        screen.blit(close_txt, (config.screen_width//2 - close_txt.get_width()//2,
                                 config.screen_height - 28))


class InventoryScene(Scene):
    """Inventory screen."""

    def __init__(self, engine):
        super().__init__(engine)
        self.selected = 0

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type != pygame.KEYDOWN:
            return
        items = list(self.engine.state.inventory.items.items())
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % max(1, len(items))
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % max(1, len(items))
        elif event.key == pygame.K_ESCAPE:
            self.engine.pop_scene()

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")
        screen.fill(PALETTE["dark_grey"])
        title = large.render("INVENTORY", True, PALETTE["yellow"])
        screen.blit(title, (16, 16))
        gold_txt = font.render(f"Gold: {self.engine.state.inventory.gold}", True, PALETTE["yellow"])
        screen.blit(gold_txt, (config.screen_width - gold_txt.get_width() - 16, 16))
        items = list(self.engine.state.inventory.items.items())
        if not items:
            empty = font.render("(empty)", True, PALETTE["grey"])
            screen.blit(empty, (32, 60))
        for i, (item_id, count) in enumerate(items):
            item = self.engine.world.item_registry.item(item_id)
            name = item.name if item else item_id
            color = PALETTE["yellow"] if i == self.selected else PALETTE["white"]
            row = font.render(f"{'▶' if i == self.selected else ' '} {name:<24} x{count}", True, color)
            screen.blit(row, (16, 60 + i * 24))
            if i == self.selected and item:
                desc = font.render(item.description, True, PALETTE["grey"])
                screen.blit(desc, (16, config.screen_height - 50))
        esc_txt = font.render("ESC: Close", True, PALETTE["grey"])
        screen.blit(esc_txt, (16, config.screen_height - 24))


# ─────────────────────────────────────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────────────────────────────────────

class TitleScene(Scene):
    """Title/splash screen — shown at game start."""

    def __init__(self, engine: "GameEngine"):
        super().__init__(engine)
        self._blink  = 0.0
        self._show   = True

    def on_enter(self):
        self.engine.audio.play_music("overworld_theme")

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.engine.pop_scene()   # pop title → goes to map
            elif event.key == pygame.K_ESCAPE:
                self.engine.running = False

    def update(self, dt: float):
        self._blink += dt
        if self._blink >= 0.6:
            self._blink = 0.0
            self._show  = not self._show

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        title_font = assets.get_font("title")
        large      = assets.get_font("large")
        font       = assets.get_font("default")

        # Gradient background
        for y in range(config.screen_height):
            t = y / config.screen_height
            r = int(10  + 30  * t)
            g = int(5   + 20  * t)
            b = int(30  + 60  * t)
            pygame.draw.line(screen, (r,g,b), (0,y), (config.screen_width,y))

        # Title
        title_surf = title_font.render(config.title, True, PALETTE["yellow"])
        tx = config.screen_width  // 2 - title_surf.get_width()  // 2
        ty = config.screen_height // 3 - title_surf.get_height() // 2
        screen.blit(title_surf, (tx, ty))

        # Subtitle / world name
        sub = large.render(f"A tale set in {config.world_name}", True, PALETTE["grey"])
        screen.blit(sub, (config.screen_width//2 - sub.get_width()//2, ty + 50))

        # Blinking prompt
        if self._show:
            prompt = large.render("Press ENTER to begin", True, PALETTE["white"])
            screen.blit(prompt, (config.screen_width//2 - prompt.get_width()//2,
                                  config.screen_height * 2 // 3))

        # Controls reminder
        ctrl = font.render("Arrow keys / WASD: Move   Z/Enter: Action   ESC: Menu",
                            True, PALETTE["dark_grey"])
        screen.blit(ctrl, (config.screen_width//2 - ctrl.get_width()//2,
                            config.screen_height - 24))


class GameOverScene(Scene):
    """Game over screen."""

    def __init__(self, engine: "GameEngine"):
        super().__init__(engine)
        self._timer = 0.0

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Revive party with 1 HP and return to start
                for char in self.engine.state.party:
                    char.current_hp = max(1, char.stats.max_hp // 4)
                start_area = self.engine.world.world_map.start_area
                sx, sy     = self.engine.world.world_map.start_pos
                self.engine.warp_to(start_area, sx, sy)

    def update(self, dt: float):
        self._timer += dt

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        title  = assets.get_font("title")
        large  = assets.get_font("large")
        font   = assets.get_font("default")

        # Dark red vignette
        for y in range(config.screen_height):
            t = y / config.screen_height
            screen.fill((int(30*t), 0, 0),
                        (0, y, config.screen_width, 1))

        go_txt = title.render("GAME OVER", True, PALETTE["red"])
        screen.blit(go_txt, (config.screen_width//2 - go_txt.get_width()//2,
                              config.screen_height//2 - 50))
        if self._timer > 1.5:
            cont = large.render("Press ENTER to continue", True, PALETTE["white"])
            screen.blit(cont, (config.screen_width//2 - cont.get_width()//2,
                                config.screen_height//2 + 20))
        time_txt = font.render(
            f"Played: {int(self.engine.state.play_time//60)}m {int(self.engine.state.play_time%60)}s",
            True, PALETTE["grey"]
        )
        screen.blit(time_txt, (config.screen_width//2 - time_txt.get_width()//2,
                                config.screen_height//2 + 60))


class SkillMenuScene(Scene):
    """In-battle skill selection."""

    def __init__(self, engine: "GameEngine", battle: "BattleScene"):
        super().__init__(engine)
        self.battle   = battle
        self.selected = 0
        # Build skill list from player
        player = engine.state.player
        roster = engine.world.npc_roster
        pclass = roster.player_classes.get(player.class_id) if player else None
        self.skills = []
        if pclass and player:
            skill_map = {s.id: s for s in pclass.skills}
            for sid in player.skills:
                if sid in skill_map:
                    self.skills.append(skill_map[sid])

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % max(1, len(self.skills))
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % max(1, len(self.skills))
        elif event.key in (pygame.K_RETURN, pygame.K_z):
            self._use_skill()
        elif event.key == pygame.K_ESCAPE:
            self.engine.pop_scene()

    def _use_skill(self):
        if not self.skills:
            self.engine.pop_scene()
            return
        skill  = self.skills[self.selected]
        player = self.engine.state.player
        if not player:
            return
        if not player.use_mp(skill.mp_cost):
            self.battle.log.append(f"Not enough MP! (need {skill.mp_cost})")
            self.engine.pop_scene()
            return

        if skill.effect == "damage" and self.battle.enemies:
            target = self.battle.enemies[0]
            power  = skill.power + (player.stats.magic if skill.is_magic else player.stats.attack)
            dmg    = max(1, power - target.stats.defense // 2)
            # Element bonus (placeholder — all elements deal normal damage here)
            target.current_hp -= dmg
            self.battle.log.append(f"{player.name} uses {skill.name} for {dmg} damage!")
            if target.current_hp <= 0:
                self.battle.log.append(f"{target.name} is defeated!")
                self.battle.enemies.remove(target)
            if not self.battle.enemies:
                self.engine.pop_scene()
                self.battle._win()
                return
        elif skill.effect == "heal":
            healed = player.heal(skill.power + player.stats.magic)
            self.battle.log.append(f"{player.name} uses {skill.name}! Restored {healed} HP.")
        elif skill.effect == "status" and self.battle.enemies:
            target = self.battle.enemies[0]
            if skill.status_effect:
                target.status_effects = getattr(target, 'status_effects', [])
                target.status_effects.append(skill.status_effect)
                self.battle.log.append(f"{target.name} is now {skill.status_effect}!")

        self.battle.phase = "enemy_turn"
        self.engine.pop_scene()

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")
        player = self.engine.state.player

        w, h = 280, min(240, 48 + len(self.skills) * 28)
        x = config.screen_width  // 2 - w // 2
        y = config.screen_height // 2 - h // 2
        pygame.draw.rect(screen, PALETTE["dark_grey"], (x, y, w, h))
        pygame.draw.rect(screen, PALETTE["white"],     (x, y, w, h), 2)

        title = large.render("SKILLS", True, PALETTE["yellow"])
        screen.blit(title, (x + w//2 - title.get_width()//2, y + 8))

        mp_txt = font.render(
            f"MP: {player.current_mp}/{player.stats.max_mp}" if player else "",
            True, PALETTE["blue"]
        )
        screen.blit(mp_txt, (x + w - mp_txt.get_width() - 8, y + 10))

        if not self.skills:
            none_txt = font.render("(no skills)", True, PALETTE["grey"])
            screen.blit(none_txt, (x + w//2 - none_txt.get_width()//2, y + 40))
        for i, skill in enumerate(self.skills):
            color  = PALETTE["yellow"] if i == self.selected else PALETTE["white"]
            mp_col = PALETTE["blue"] if (player and player.current_mp >= skill.mp_cost) else PALETTE["red"]
            row = font.render(
                f"{'▶' if i==self.selected else ' '} {skill.name:<16} {skill.mp_cost}MP",
                True, color
            )
            screen.blit(row, (x + 8, y + 40 + i * 28))
        esc = font.render("ESC: Back", True, PALETTE["grey"])
        screen.blit(esc, (x + 8, y + h - 20))


class ItemMenuScene(Scene):
    """In-battle item use menu."""

    def __init__(self, engine: "GameEngine", battle: "BattleScene"):
        super().__init__(engine)
        self.battle   = battle
        self.selected = 0
        # Only usable items (consumables)
        self.usable = [
            (item_id, count, engine.world.item_registry.item(item_id))
            for item_id, count in engine.state.inventory.items.items()
            if engine.world.item_registry.item(item_id) and
               engine.world.item_registry.item(item_id).effect
        ]

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % max(1, len(self.usable))
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % max(1, len(self.usable))
        elif event.key in (pygame.K_RETURN, pygame.K_z):
            self._use_item()
        elif event.key == pygame.K_ESCAPE:
            self.engine.pop_scene()

    def _use_item(self):
        if not self.usable:
            self.engine.pop_scene()
            return
        item_id, count, item = self.usable[self.selected]
        player = self.engine.state.player
        if not player or not item:
            self.engine.pop_scene()
            return
        self.engine.state.inventory.remove(item_id)
        if item.effect == "heal_hp":
            healed = player.heal(item.effect_power)
            self.battle.log.append(f"Used {item.name}! Restored {healed} HP.")
        elif item.effect == "heal_mp":
            restored = player.restore_mp(item.effect_power)
            self.battle.log.append(f"Used {item.name}! Restored {restored} MP.")
        elif item.effect == "full_restore":
            player.current_hp = player.stats.max_hp
            player.current_mp = player.stats.max_mp
            self.battle.log.append(f"Used {item.name}! Fully restored!")
        elif item.effect == "revive":
            # Revive first fallen party member
            fallen = [c for c in self.engine.state.party if not c.is_alive]
            if fallen:
                target = fallen[0]
                target.current_hp = max(1, int(target.stats.max_hp * item.effect_power / 100))
                self.battle.log.append(f"Used {item.name}! {target.name} revived!")
        self.battle.phase = "enemy_turn"
        self.engine.pop_scene()

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")

        w, h = 280, min(240, 48 + max(1,len(self.usable)) * 28)
        x = config.screen_width  // 2 - w // 2
        y = config.screen_height // 2 - h // 2
        pygame.draw.rect(screen, PALETTE["dark_grey"], (x, y, w, h))
        pygame.draw.rect(screen, PALETTE["white"],     (x, y, w, h), 2)

        title = large.render("ITEMS", True, PALETTE["yellow"])
        screen.blit(title, (x + w//2 - title.get_width()//2, y + 8))

        if not self.usable:
            none_txt = font.render("(no usable items)", True, PALETTE["grey"])
            screen.blit(none_txt, (x + w//2 - none_txt.get_width()//2, y + 40))
        for i, (item_id, count, item) in enumerate(self.usable):
            color = PALETTE["yellow"] if i == self.selected else PALETTE["white"]
            row   = font.render(
                f"{'▶' if i==self.selected else ' '} {item.name:<18} x{count}",
                True, color
            )
            screen.blit(row, (x + 8, y + 40 + i * 28))
            if i == self.selected:
                desc = font.render(item.description, True, PALETTE["grey"])
                screen.blit(desc, (x + 8, y + h - 40))
        esc = font.render("ESC: Back", True, PALETTE["grey"])
        screen.blit(esc, (x + 8, y + h - 20))


class QuestLogScene(Scene):
    """Quest journal — shows active and completed quests with objectives."""

    def __init__(self, engine: "GameEngine"):
        super().__init__(engine)
        self.selected = 0
        self.tab      = "active"   # "active" | "completed"

    def _quest_list(self) -> list:
        wq    = self.engine.world.quest_graph
        state = self.engine.state
        if self.tab == "active":
            return [wq.quest(qid) for qid in state.quests.active
                    if wq.quest(qid)]
        else:
            return [wq.quest(qid) for qid in state.quests.completed
                    if wq.quest(qid)]

    def handle_event(self, event):
        if not PYGAME_AVAILABLE:
            return
        if event.type != pygame.KEYDOWN:
            return
        quests = self._quest_list()
        if event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % max(1, len(quests))
        elif event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % max(1, len(quests))
        elif event.key in (pygame.K_LEFT, pygame.K_a):
            self.tab = "active"
            self.selected = 0
        elif event.key in (pygame.K_RIGHT, pygame.K_d):
            self.tab = "completed"
            self.selected = 0
        elif event.key == pygame.K_ESCAPE:
            self.engine.pop_scene()

    def render(self, screen):
        if not PYGAME_AVAILABLE:
            return
        config = self.engine.world.config
        assets = self.engine.assets
        font   = assets.get_font("default")
        large  = assets.get_font("large")

        screen.fill(PALETTE["dark_grey"])

        # Title bar
        title = large.render("QUEST LOG", True, PALETTE["yellow"])
        screen.blit(title, (16, 12))

        # Tab selector
        for i, tab in enumerate(["Active", "Completed"]):
            is_sel = (tab.lower() == self.tab)
            color  = PALETTE["yellow"] if is_sel else PALETTE["grey"]
            txt    = font.render(f"[{tab}]" if is_sel else f" {tab} ", True, color)
            screen.blit(txt, (160 + i * 90, 14))

        esc_txt = font.render("ESC: Back  ◄►: Switch tab", True, PALETTE["grey"])
        screen.blit(esc_txt, (config.screen_width - esc_txt.get_width() - 8, 14))

        pygame.draw.line(screen, PALETTE["grey"], (0, 36), (config.screen_width, 36))

        quests = self._quest_list()
        if not quests:
            empty = font.render(
                "(no active quests)" if self.tab == "active" else "(none completed yet)",
                True, PALETTE["grey"]
            )
            screen.blit(empty, (32, 60))
            return

        # Quest list on left panel
        list_w = config.screen_width // 3
        for i, quest in enumerate(quests[:12]):
            color = PALETTE["yellow"] if i == self.selected else PALETTE["white"]
            icon  = "▶ " if i == self.selected else "  "
            star  = "★ " if quest.is_main_quest else "  "
            txt   = font.render(f"{icon}{star}{quest.name}", True, color)
            screen.blit(txt, (8, 44 + i * 22))

        pygame.draw.line(screen, PALETTE["grey"],
                         (list_w, 36), (list_w, config.screen_height - 36))

        # Detail panel on right
        if self.selected < len(quests):
            q     = quests[self.selected]
            state = self.engine.state
            dx    = list_w + 12
            dy    = 44

            name_txt = large.render(q.name, True, PALETTE["yellow"])
            screen.blit(name_txt, (dx, dy)); dy += 28

            # Description (word-wrap)
            words = q.description.split()
            line, max_w = "", config.screen_width - dx - 12
            lines = []
            for word in words:
                test = (line + " " + word).strip()
                if font.size(test)[0] <= max_w:
                    line = test
                else:
                    if line: lines.append(line)
                    line = word
            if line: lines.append(line)
            for ln in lines[:3]:
                screen.blit(font.render(ln, True, PALETTE["grey"]), (dx, dy))
                dy += 18
            dy += 8

            # Objectives
            screen.blit(font.render("Objectives:", True, PALETTE["white"]), (dx, dy))
            dy += 20
            for obj in q.objectives:
                prog   = state.quests.objective_progress(q.id, obj.id)
                done   = prog >= obj.target_count
                color  = PALETTE["green"] if done else (PALETTE["grey"] if obj.optional else PALETTE["white"])
                check  = "✓" if done else "○"
                count  = f" ({prog}/{obj.target_count})" if obj.target_count > 1 else ""
                opt    = " (optional)" if obj.optional else ""
                row    = font.render(f"  {check} {obj.description}{count}{opt}", True, color)
                screen.blit(row, (dx, dy)); dy += 20

            # Rewards
            if q.reward_gold or q.reward_exp or q.reward_items:
                dy += 8
                screen.blit(font.render("Rewards:", True, PALETTE["white"]), (dx, dy))
                dy += 20
                rewards = []
                if q.reward_gold: rewards.append(f"{q.reward_gold}G")
                if q.reward_exp:  rewards.append(f"{q.reward_exp} EXP")
                for iid in q.reward_items:
                    item = self.engine.world.item_registry.item(iid)
                    rewards.append(item.name if item else iid)
                screen.blit(font.render("  " + "  ".join(rewards), True, PALETTE["cyan"]), (dx, dy))

        # Footer
        pygame.draw.line(screen, PALETTE["grey"],
                         (0, config.screen_height-28), (config.screen_width, config.screen_height-28))
        active_count = len(self.engine.state.quests.active)
        done_count   = len(self.engine.state.quests.completed)
        footer = font.render(
            f"Active: {active_count}   Completed: {done_count}   "
            f"Total: {len(self.engine.world.quest_graph.quests)}",
            True, PALETTE["grey"]
        )
        screen.blit(footer, (8, config.screen_height - 20))


class AudioSystem:
    """Lightweight audio manager using pygame.mixer (optional)."""

    def __init__(self, manifest):
        self.manifest = manifest
        self._music_key: str | None = None
        self._ready     = False
        try:
            if PYGAME_AVAILABLE:
                import pygame.mixer as _m
                _m.init(frequency=44100, size=-16, channels=2, buffer=512)
                self._ready = True
        except Exception:
            pass

    def play_music(self, key: str, loop: bool = True) -> None:
        if not self._ready or key == self._music_key:
            return
        path = self.manifest.music_path(key)
        if not path:
            return
        try:
            import pygame.mixer as _m
            _m.music.load(path)
            _m.music.play(-1 if loop else 0)
            self._music_key = key
        except Exception:
            pass

    def stop_music(self) -> None:
        if not self._ready:
            return
        try:
            import pygame.mixer as _m
            _m.music.stop()
            self._music_key = None
        except Exception:
            pass

    def play_sound(self, key: str) -> None:
        if not self._ready:
            return
        path = self.manifest.sounds.get(key)
        if not path:
            return
        try:
            import pygame.mixer as _m
            snd = _m.Sound(path)
            snd.play()
        except Exception:
            pass


class GameEngine:
    """
    The game engine runtime.

    Usage::
        from engine.runtime import GameEngine
        from procgen_domain import generate_world

        world = generate_world(seed=42, genre="jrpg", tone="heroic")
        engine = GameEngine(world)
        engine.run()
    """

    def __init__(self, world: World, asset_paths: dict | None = None):
        self.world   = world
        self.state   = self._init_state()
        self.scenes: list[Scene] = []
        self.running = False
        self.clock   = None

        # Set up asset manifest with user-provided paths
        if asset_paths:
            for key, path in asset_paths.items():
                if key in world.asset_manifest.sprites:
                    world.asset_manifest.sprites[key] = path
                elif key in world.asset_manifest.music:
                    world.asset_manifest.music[key] = path

        self.assets = AssetManager(world.asset_manifest, world.config.tile_size)
        self.audio  = AudioSystem(world.asset_manifest)

    def _init_state(self) -> GameState:
        """Build initial GameState from the World config."""
        config  = self.world.config
        roster  = self.world.npc_roster
        wm      = self.world.world_map

        # Build player character state
        party   = []
        pclass  = roster.player_classes.get(config.protagonist_class)
        if pclass:
            from .core.state import CharacterState
            pc = CharacterState(
                npc_id   = "player",
                name     = config.protagonist_name,
                class_id = config.protagonist_class,
                level    = config.starting_level,
                current_hp = pclass.base_stats.hp,
                current_mp = pclass.base_stats.mp,
                stats    = pclass.base_stats,
                skills   = [s.id for s in pclass.skills if s.learn_level <= config.starting_level],
            )
            party.append(pc)

        from .core.state import Inventory
        inv = Inventory(gold=config.starting_gold)

        return GameState(
            position = PlayerPosition(
                area_id = wm.start_area,
                x       = wm.start_pos[0],
                y       = wm.start_pos[1],
            ),
            party     = party,
            inventory = inv,
        )

    # ── Scene management ─────────────────────────────────────────────────────

    def push_scene(self, scene: Scene) -> None:
        if self.scenes:
            # Don't call on_exit for map scene — we want to resume it
            pass
        self.scenes.append(scene)
        scene.on_enter()

    def pop_scene(self) -> None:
        if self.scenes:
            self.scenes[-1].on_exit()
            self.scenes.pop()

    def replace_scene(self, scene: Scene) -> None:
        if self.scenes:
            self.scenes[-1].on_exit()
            self.scenes[-1] = scene
            scene.on_enter()

    @property
    def current_scene(self) -> Scene | None:
        return self.scenes[-1] if self.scenes else None

    def warp_to(self, area_id: str, x: int, y: int) -> None:
        # Gate dungeon entry on required key item
        area = self.world.world_map.area(area_id)
        if area:
            from engine.core.world import AreaType
            if area.area_type == AreaType.DUNGEON:
                gate_key = area.metadata.get("requires_item")
                if gate_key and not self.state.inventory.has(gate_key):
                    item = self.world.item_registry.item(gate_key)
                    name = item.name if item else gate_key
                    self.push_scene(
                        MessageScene(self, f"The path is sealed. You need the {name}.")
                    )
                    return
        self.state.position.area_id = area_id
        self.state.position.x       = x
        self.state.position.y       = y
        # Check reach_area quest objectives
        self._check_reach_area(area_id)
        self.replace_scene(MapScene(self, area_id))

    def _check_reach_area(self, area_id: str) -> None:
        """Complete reach_area objectives when warping to a new area."""
        state = self.state
        wq    = self.world.quest_graph
        for q_id in list(state.quests.active):
            quest = wq.quest(q_id)
            if not quest:
                continue
            for obj in quest.objectives:
                if obj.type == "reach_area" and obj.target_id == area_id:
                    current = state.quests.objective_progress(q_id, obj.id)
                    if current < obj.target_count:
                        state.quests.advance(q_id, obj.id, obj.target_count - current)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save_game(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump({
                "world_seed":  self.world.seed,
                "world_genre": self.world.config.genre,
                "world_tone":  self.world.config.tone,
                "protagonist": self.world.config.protagonist_class,
                "antagonist":  self.world.metadata.get("antagonist", "demon_lord"),
                "world_size":  self.world.metadata.get("world_size", "medium"),
                "difficulty":  self.world.config.difficulty,
                "chapters":    self.world.config.chapters,
                "state":       self.state.to_dict(),
            }, f, indent=2)

    @classmethod
    def load_game(cls, path: str) -> "GameEngine":
        """Load a saved game. Regenerates the world from seed + params, then restores state."""
        import json
        with open(path) as f:
            data = json.load(f)
        # Re-generate the world from its seed and parameters
        from procgen.core import Designer
        from procgen_domain import JRPGDomain
        d = Designer(domain=JRPGDomain())
        d.set("genre",       data.get("world_genre", "jrpg"))
        d.set("tone",        data.get("world_tone",  "heroic"))
        d.set("protagonist", data.get("protagonist", "warrior"))
        d.set("antagonist",  data.get("antagonist",  "demon_lord"))
        d.set("world_size",  data.get("world_size",  "medium"))
        d.set("difficulty",  data.get("difficulty",  "normal"))
        d.set("chapters",    data.get("chapters",    3))
        d.set_seed(data["world_seed"])
        world = d.generate()

        engine = cls(world)

        # Restore runtime state
        s = data["state"]
        from engine.core.state import (
            GameState, PlayerPosition, CharacterState, Stats,
            Inventory, QuestState, GameFlags,
        )
        # Rebuild party
        party = []
        for cd in s.get("party", []):
            stats_d = cd.get("stats", {})
            stats   = Stats(
                hp=stats_d.get("hp",10), mp=stats_d.get("mp",5),
                attack=stats_d.get("attack",5), defense=stats_d.get("defense",5),
                magic=stats_d.get("magic",5), speed=stats_d.get("speed",5),
                luck=stats_d.get("luck",5),
                max_hp=stats_d.get("max_hp",10), max_mp=stats_d.get("max_mp",5),
            )
            char = CharacterState(
                npc_id=cd["npc_id"], name=cd["name"], class_id=cd["class_id"],
                level=cd["level"], exp=cd["exp"], exp_to_next=cd["exp_to_next"],
                current_hp=cd["current_hp"], current_mp=cd["current_mp"],
                stats=stats, skills=cd.get("skills",[]),
                equipment=cd.get("equipment",{}),
                status_effects=cd.get("status_effects",[]),
            )
            party.append(char)

        inv_d = s.get("inventory", {})
        inv   = Inventory(items=dict(inv_d.get("items",{})), gold=inv_d.get("gold",0))

        q_d   = s.get("quests", {})
        qs    = QuestState(
            active=set(q_d.get("active",[])),
            completed=set(q_d.get("completed",[])),
            failed=set(q_d.get("failed",[])),
            progress=q_d.get("progress",{}),
        )

        pos_d = s.get("position", {})
        pos   = PlayerPosition(
            area_id=pos_d.get("area_id", world.world_map.start_area),
            x=pos_d.get("x", world.world_map.start_pos[0]),
            y=pos_d.get("y", world.world_map.start_pos[1]),
            facing=pos_d.get("facing","down"),
        )

        engine.state = GameState(
            position=pos, party=party, inventory=inv, quests=qs,
            flags=GameFlags(flags=s.get("flags",{})),
            chapter=s.get("chapter",1),
            play_time=s.get("play_time",0.0),
            step_count=s.get("step_count",0),
            opened_chests=set(s.get("opened_chests",[])),
            triggered=set(s.get("triggered",[])),
        )
        return engine

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        if not PYGAME_AVAILABLE:
            print("pygame not installed — run:  pip install pygame")
            return

        pygame.init()
        config = self.world.config
        screen = pygame.display.set_mode((config.screen_width, config.screen_height))
        pygame.display.set_caption(config.title)
        self.clock = pygame.time.Clock()

        self.assets.load_all(screen)

        # Start on the world map, with title screen on top
        start_area = self.world.world_map.start_area
        self.push_scene(MapScene(self, start_area))
        self.push_scene(TitleScene(self))

        self.running = True
        prev_time    = time.time()
        _area_msg_timer = 0.0
        _last_area      = ""

        while self.running:
            now    = time.time()
            dt     = min(now - prev_time, 0.05)
            prev_time = now

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif self.current_scene:
                    self.current_scene.handle_event(event)

            # Update (only top scene)
            if self.current_scene:
                self.current_scene.update(dt)
                self.state.play_time += dt

            # Game over detection
            if self.state.is_game_over and not isinstance(self.current_scene, GameOverScene):
                # Clear battle scenes
                while self.scenes and not isinstance(self.scenes[-1], MapScene):
                    self.scenes.pop()
                self.push_scene(GameOverScene(self))

            # Area transition notification
            cur_area = self.state.position.area_id
            if cur_area != _last_area and _last_area:
                area = self.world.world_map.area(cur_area)
                if area:
                    self.audio.play_music(area.music_key)
                    _area_msg_timer = 2.5
            _last_area = cur_area
            if _area_msg_timer > 0:
                _area_msg_timer -= dt

            # Render (bottom-up so map shows through dialogue/menus)
            for scene in self.scenes:
                scene.render(screen)

            # Area name overlay (fades after 2.5 sec)
            if _area_msg_timer > 0:
                area_obj  = self.world.world_map.area(cur_area)
                area_font = self.assets.get_font("large")
                if area_obj and area_font:
                    alpha  = min(255, int(_area_msg_timer / 2.5 * 255 * 2))
                    surf   = area_font.render(area_obj.name, True, PALETTE["white"])
                    surf.set_alpha(alpha)
                    screen.blit(surf, (config.screen_width//2 - surf.get_width()//2, 30))

            pygame.display.flip()
            self.clock.tick(config.target_fps)

        pygame.quit()
