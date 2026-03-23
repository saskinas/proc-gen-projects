"""
TerrainDomain: Procedural world/terrain map generation.

Uses Wave Function Collapse for tile consistency and layered
value noise for elevation/moisture.

Designer knobs:
    biome       — str   : "island" | "continent" | "arctic" | "desert" | "volcanic"
    width       — int   : 8–64
    height      — int   : 8–64
    roughness   — float : 0.0 (smooth rolling hills) → 1.0 (jagged peaks)
    moisture    — float : 0.0 (arid) → 1.0 (lush)
    sea_level   — float : 0.0–1.0 (fraction of map covered by water)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math

from .base import DomainBase
from ..core import Intent, GenerationPlan
from ..generators import GeneratorBase
from ..wfc import WFCGrid, dungeon_tiles, terrain_tiles, DIRECTIONS, Tile


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class TerrainMap:
    width: int
    height: int
    biome: str
    elevation: list[list[float]]    # 0.0–1.0
    moisture:  list[list[float]]    # 0.0–1.0
    tile_grid: list[list[str]]      # tile name per cell
    legend: dict[str, str]          # tile → ascii symbol
    metadata: dict = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            f"🌍 Terrain Map",
            f"   Biome    : {self.biome}",
            f"   Size     : {self.width}×{self.height}",
            f"   Legend   : {self.legend}",
            "",
        ]
        for row in self.tile_grid:
            line = "   " + "".join(self.legend.get(cell, "?") for cell in row)
            lines.append(line)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Noise utilities (pure Python, no deps)
# ---------------------------------------------------------------------------

def _value_noise(x: float, y: float, seed: int) -> float:
    """Cheap deterministic 2D noise, returns [0, 1]."""
    xi, yi = int(x), int(y)
    xf, yf = x - xi, y - yi

    def _hash(a, b):
        n = a * 1619 + b * 31337 + seed * 1000003
        n = (n ^ (n >> 13)) * 1376312589
        return ((n ^ (n >> 15)) & 0x7FFFFFFF) / 0x7FFFFFFF

    v00 = _hash(xi,   yi)
    v10 = _hash(xi+1, yi)
    v01 = _hash(xi,   yi+1)
    v11 = _hash(xi+1, yi+1)

    # Smooth interpolation
    sx = xf * xf * (3 - 2 * xf)
    sy = yf * yf * (3 - 2 * yf)

    top    = v00 + sx * (v10 - v00)
    bottom = v01 + sx * (v11 - v01)
    return top + sy * (bottom - top)


def _fractal_noise(x: float, y: float, seed: int, octaves: int = 4, roughness: float = 0.5) -> float:
    """Layered value noise (fBm). Returns [0, 1]."""
    value = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    for _ in range(octaves):
        value    += _value_noise(x * frequency, y * frequency, seed) * amplitude
        max_val  += amplitude
        amplitude *= roughness
        frequency *= 2.0
    return value / max_val


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

_BIOME_CONFIGS = {
    "island": {
        "island_shape": True,
        "sea_bias": 0.5,
        "tiles": terrain_tiles,
        "legend": {"ocean": "~", "beach": ".", "plains": ",", "forest": "T", "mountain": "^"},
        "elevation_octaves": 5,
    },
    "continent": {
        "island_shape": False,
        "sea_bias": 0.35,
        "tiles": terrain_tiles,
        "legend": {"ocean": "~", "beach": ".", "plains": ",", "forest": "T", "mountain": "^"},
        "elevation_octaves": 6,
    },
    "arctic": {
        "island_shape": False,
        "sea_bias": 0.3,
        "tiles": [
            Tile("ice_ocean", weight=2.0, allowed={d: ["ice_ocean","ice_shelf"] for d in DIRECTIONS}),
            Tile("ice_shelf", weight=1.5, allowed={d: ["ice_ocean","ice_shelf","tundra"] for d in DIRECTIONS}),
            Tile("tundra",    weight=3.0, allowed={d: ["ice_shelf","tundra","glacier"] for d in DIRECTIONS}),
            Tile("glacier",   weight=1.0, allowed={d: ["tundra","glacier"] for d in DIRECTIONS}),
        ],
        "legend": {"ice_ocean": "~", "ice_shelf": "=", "tundra": ".", "glacier": "*"},
        "elevation_octaves": 4,
    },
    "desert": {
        "island_shape": False,
        "sea_bias": 0.15,
        "tiles": [
            Tile("dune",   weight=4.0, allowed={d: ["dune","rocky","oasis"] for d in DIRECTIONS}),
            Tile("rocky",  weight=2.0, allowed={d: ["dune","rocky","mesa"] for d in DIRECTIONS}),
            Tile("mesa",   weight=1.0, allowed={d: ["rocky","mesa"] for d in DIRECTIONS}),
            Tile("oasis",  weight=0.4, allowed={d: ["dune","oasis"] for d in DIRECTIONS}),
        ],
        "legend": {"dune": "~", "rocky": ".", "mesa": "^", "oasis": "O"},
        "elevation_octaves": 3,
    },
    "volcanic": {
        "island_shape": True,
        "sea_bias": 0.55,
        "tiles": [
            Tile("sea",    weight=3.0, allowed={d: ["sea","shore"] for d in DIRECTIONS}),
            Tile("shore",  weight=1.0, allowed={d: ["sea","shore","ash"] for d in DIRECTIONS}),
            Tile("ash",    weight=3.0, allowed={d: ["shore","ash","lava","obsidian"] for d in DIRECTIONS}),
            Tile("lava",   weight=1.0, allowed={d: ["ash","lava","obsidian"] for d in DIRECTIONS}),
            Tile("obsidian",weight=0.5,allowed={d: ["lava","obsidian","ash"] for d in DIRECTIONS}),
        ],
        "legend": {"sea": "~", "shore": ".", "ash": ":", "lava": "@", "obsidian": "#"},
        "elevation_octaves": 5,
    },
}


class ElevationGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> list[list[float]]:
        rng = self.rng()
        w, h = params["width"], params["height"]
        roughness = params["roughness"]
        biome_cfg = _BIOME_CONFIGS[params["biome"]]
        octaves = biome_cfg["elevation_octaves"]
        seed = rng.randint(0, 999999)

        scale = 3.0 / max(w, h)
        grid = []
        for y in range(h):
            row = []
            for x in range(w):
                v = _fractal_noise(x * scale, y * scale, seed, octaves=octaves, roughness=0.3 + roughness * 0.5)
                # Island shape: distance falloff from centre
                if biome_cfg.get("island_shape"):
                    cx, cy = (w - 1) / 2, (h - 1) / 2
                    dist = math.sqrt((x - cx)**2 + (y - cy)**2) / (min(w, h) / 2)
                    falloff = max(0.0, 1.0 - dist * 1.2)
                    v = v * falloff
                row.append(v)
            grid.append(row)

        # Normalise to [0, 1]
        flat = [v for row in grid for v in row]
        lo, hi = min(flat), max(flat)
        if hi > lo:
            grid = [[(v - lo) / (hi - lo) for v in row] for row in grid]
        return grid


class MoistureGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> list[list[float]]:
        rng = self.rng()
        w, h = params["width"], params["height"]
        moisture_level = params["moisture"]
        seed = rng.randint(0, 999999)
        scale = 4.0 / max(w, h)

        grid = []
        for y in range(h):
            row = []
            for x in range(w):
                v = _fractal_noise(x * scale, y * scale, seed, octaves=3, roughness=0.5)
                # Bias toward target moisture level
                v = v * 0.5 + moisture_level * 0.5
                row.append(max(0.0, min(1.0, v)))
            grid.append(row)
        return grid


class TileGridGenerator(GeneratorBase):
    def generate(self, params: dict, context: dict) -> list[list[str]]:
        rng = self.rng()
        w, h = params["width"], params["height"]
        biome = params["biome"]
        sea_level = params["sea_level"]
        biome_cfg = _BIOME_CONFIGS[biome]
        elevation = context.get("elevation", [[0.5] * w for _ in range(h)])

        tiles = biome_cfg["tiles"]() if callable(biome_cfg["tiles"]) else biome_cfg["tiles"]

        # For terrain biomes: map elevation directly to tiles
        tile_names = [t.name for t in tiles]

        # Try WFC — fall back to elevation-based mapping on contradiction
        wfc = WFCGrid(width=w, height=h, tiles=tiles, max_retries=3)
        result = wfc.solve(rng)

        # If WFC left holes (None), fill with elevation mapping
        for y in range(h):
            for x in range(w):
                if result[y][x] is None:
                    elev = elevation[y][x]
                    idx = min(int(elev * len(tile_names)), len(tile_names) - 1)
                    result[y][x] = tile_names[idx]

        return result


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class TerrainDomain(DomainBase):
    def __init__(self):
        from ..constraints import ConstraintSet, Rule
        self.schema = ConstraintSet([
            Rule("biome",     type=str,   required=True,
                 choices=list(_BIOME_CONFIGS.keys()),
                 description="World biome type"),
            Rule("width",     type=int,   default=24, range=(8, 64)),
            Rule("height",    type=int,   default=16, range=(8, 64)),
            Rule("roughness", type=float, default=0.5, range=(0.0, 1.0),
                 description="Terrain roughness / fractal complexity"),
            Rule("moisture",  type=float, default=0.5, range=(0.0, 1.0),
                 description="Average moisture level"),
            Rule("sea_level", type=float, default=0.4, range=(0.0, 1.0),
                 description="Fraction of map below sea"),
        ])

    def build_plan(self, intent: Intent) -> GenerationPlan:
        from ..core import GenerationPlan
        plan = GenerationPlan()
        shared = {
            "biome":     intent.get("biome"),
            "width":     intent.get("width"),
            "height":    intent.get("height"),
            "roughness": intent.get("roughness"),
            "moisture":  intent.get("moisture"),
            "sea_level": intent.get("sea_level"),
        }
        plan.add_step("elevation", shared)
        plan.add_step("moisture_map", shared)
        plan.add_step("tile_grid",  shared)
        return plan

    def assemble(self, outputs: dict[str, Any], plan: GenerationPlan) -> TerrainMap:
        intent = plan.metadata["intent_summary"]
        biome = intent["biome"]
        cfg = _BIOME_CONFIGS[biome]
        return TerrainMap(
            width     = intent["width"],
            height    = intent["height"],
            biome     = biome,
            elevation = outputs["elevation"],
            moisture  = outputs["moisture_map"],
            tile_grid = outputs["tile_grid"],
            legend    = cfg["legend"],
            metadata  = {"intent": intent},
        )

    @property
    def generators(self):
        return {
            "elevation":   ElevationGenerator,
            "moisture_map": MoistureGenerator,
            "tile_grid":   TileGridGenerator,
        }
