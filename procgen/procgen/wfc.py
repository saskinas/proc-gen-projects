"""
procgen.wfc — Wave Function Collapse (WFC) for grid-based generation.

A simplified but genuine WFC implementation suitable for:
- Dungeon tile maps
- Terrain biome grids
- Musical rhythm grids (allowed adjacencies)
- Pattern-consistent world generation

Concepts:
    Tile        — a named cell type with allowed neighbours per direction
    WFCGrid     — the grid being solved; each cell starts as a superposition
                  of all tiles, then collapses via constraint propagation

Usage::

    from procgen.wfc import WFCGrid, Tile

    tiles = [
        Tile("wall",  {"N": ["wall","corner"], "S": ["wall","corner"],
                       "E": ["wall","corner"], "W": ["wall","corner"]}),
        Tile("floor", {"N": ["floor","wall"],  "S": ["floor","wall"],
                       "E": ["floor","wall"],  "W": ["floor","wall"]}),
        Tile("corner",{"N": ["wall"],          "S": ["floor"],
                       "E": ["floor"],         "W": ["wall"]}),
    ]
    grid = WFCGrid(width=10, height=10, tiles=tiles)
    result = grid.solve(rng)   # list[list[str | None]]
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from typing import NamedTuple


DIRECTIONS = ("N", "S", "E", "W")
OPPOSITES  = {"N": "S", "S": "N", "E": "W", "W": "E"}
DELTAS     = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


@dataclass
class Tile:
    """
    A tile type with adjacency rules.

    Args:
        name        — identifier string
        allowed     — dict mapping direction → list of allowed neighbour tile names
        weight      — relative probability of this tile being chosen (default 1.0)
    """
    name: str
    allowed: dict[str, list[str]]
    weight: float = 1.0

    def can_neighbour(self, direction: str, other_name: str) -> bool:
        return other_name in self.allowed.get(direction, [])


class Cell:
    """A grid cell that holds a superposition of possible tiles."""

    def __init__(self, tiles: list[Tile]):
        self._possible: set[str] = {t.name for t in tiles}
        self._weights:  dict[str, float] = {t.name: t.weight for t in tiles}
        self.collapsed: str | None = None

    @property
    def entropy(self) -> float:
        """Shannon entropy of the current distribution."""
        if len(self._possible) == 1:
            return 0.0
        total = sum(self._weights[n] for n in self._possible)
        h = 0.0
        for name in self._possible:
            p = self._weights[name] / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def is_collapsed(self) -> bool:
        return self.collapsed is not None

    def is_contradiction(self) -> bool:
        return len(self._possible) == 0 and self.collapsed is None

    def collapse(self, rng: random.Random) -> str:
        """Pick one tile from the remaining possibilities."""
        names = list(self._possible)
        weights = [self._weights[n] for n in names]
        chosen = rng.choices(names, weights=weights, k=1)[0]
        self._possible = {chosen}
        self.collapsed = chosen
        return chosen

    def constrain(self, allowed: set[str]) -> bool:
        """
        Remove possibilities not in `allowed`.
        Returns True if the cell's possibilities changed.
        """
        before = len(self._possible)
        self._possible &= allowed
        return len(self._possible) < before

    def possibilities(self) -> set[str]:
        return set(self._possible)


class WFCGrid:
    """
    Wave Function Collapse grid solver.

    Args:
        width, height — grid dimensions
        tiles         — list of Tile definitions
        max_retries   — how many times to retry on contradiction
    """

    def __init__(self, width: int, height: int, tiles: list[Tile], max_retries: int = 10):
        self.width  = width
        self.height = height
        self.tiles  = {t.name: t for t in tiles}
        self.max_retries = max_retries
        self._tile_list = tiles

    def _make_grid(self) -> list[list[Cell]]:
        return [[Cell(self._tile_list) for _ in range(self.width)]
                for _ in range(self.height)]

    def solve(self, rng: random.Random) -> list[list[str | None]]:
        """
        Run WFC until fully collapsed or max_retries exhausted.
        Returns a 2D list of tile names (or None on contradiction).
        """
        for attempt in range(self.max_retries):
            grid = self._make_grid()
            success = self._run(grid, rng)
            if success:
                return [[cell.collapsed for cell in row] for row in grid]
        # Give up: return partial result
        return [[cell.collapsed for cell in row] for row in grid]

    def _run(self, grid: list[list[Cell]], rng: random.Random) -> bool:
        while True:
            # Find uncollapsed cell with lowest entropy
            cell, x, y = self._lowest_entropy_cell(grid)
            if cell is None:
                return True  # all collapsed
            if cell.is_contradiction():
                return False

            # Collapse it
            cell.collapse(rng)

            # Propagate constraints
            if not self._propagate(grid, x, y):
                return False

    def _lowest_entropy_cell(self, grid) -> tuple[Cell | None, int, int]:
        min_entropy = float("inf")
        result = (None, -1, -1)
        for y in range(self.height):
            for x in range(self.width):
                cell = grid[y][x]
                if not cell.is_collapsed():
                    if cell.is_contradiction():
                        return cell, x, y
                    e = cell.entropy
                    if e < min_entropy:
                        min_entropy = e
                        result = (cell, x, y)
        return result

    def _propagate(self, grid, start_x: int, start_y: int) -> bool:
        """Arc-consistency propagation via BFS."""
        queue = [(start_x, start_y)]
        visited = set()

        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))

            cell = grid[cy][cx]

            for direction in DIRECTIONS:
                dx, dy = DELTAS[direction]
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue

                neighbour = grid[ny][nx]
                if neighbour.is_collapsed():
                    continue

                # Which tile names are allowed from the current cell's possibilities?
                allowed: set[str] = set()
                for tile_name in cell.possibilities():
                    tile = self.tiles[tile_name]
                    allowed |= set(tile.allowed.get(direction, []))

                changed = neighbour.constrain(allowed)
                if neighbour.is_contradiction():
                    return False
                if changed:
                    queue.append((nx, ny))

        return True

    def render_ascii(self, result: list[list[str | None]], symbols: dict[str, str] | None = None) -> str:
        """Render the solved grid as ASCII art."""
        symbols = symbols or {}
        rows = []
        for row in result:
            line = ""
            for cell in row:
                if cell is None:
                    line += "?"
                else:
                    line += symbols.get(cell, cell[0].upper())
            rows.append(line)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Pre-built tile sets
# ---------------------------------------------------------------------------

def dungeon_tiles() -> list[Tile]:
    """Classic dungeon tileset: wall, floor, door."""
    return [
        Tile("wall",  weight=3.0, allowed={
            "N": ["wall", "door"],
            "S": ["wall", "door"],
            "E": ["wall", "door"],
            "W": ["wall", "door"],
        }),
        Tile("floor", weight=5.0, allowed={
            "N": ["floor", "wall", "door"],
            "S": ["floor", "wall", "door"],
            "E": ["floor", "wall", "door"],
            "W": ["floor", "wall", "door"],
        }),
        Tile("door",  weight=0.3, allowed={
            "N": ["floor", "wall"],
            "S": ["floor", "wall"],
            "E": ["floor", "wall"],
            "W": ["floor", "wall"],
        }),
    ]


def terrain_tiles() -> list[Tile]:
    """Outdoor terrain: ocean, beach, plains, forest, mountain."""
    return [
        Tile("ocean",    weight=2.0, allowed={d: ["ocean", "beach"] for d in DIRECTIONS}),
        Tile("beach",    weight=1.0, allowed={d: ["ocean", "beach", "plains"] for d in DIRECTIONS}),
        Tile("plains",   weight=3.0, allowed={d: ["beach", "plains", "forest"] for d in DIRECTIONS}),
        Tile("forest",   weight=2.0, allowed={d: ["plains", "forest", "mountain"] for d in DIRECTIONS}),
        Tile("mountain", weight=1.0, allowed={d: ["forest", "mountain"] for d in DIRECTIONS}),
    ]
