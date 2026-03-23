#!/usr/bin/env python3
"""
story_generator demo.

Usage:
    python demo.py
    python demo.py fantasy dark long
    python demo.py horror tragic medium seed=99
"""

import sys
from procgen import Designer
from story_generator import NarrativeDomain

genre  = "fantasy"
tone   = "hopeful"
length = "medium"
seed   = None

for arg in sys.argv[1:]:
    if arg.startswith("seed="):
        seed = int(arg.split("=")[1])
    elif arg in ("fantasy", "horror", "mystery", "scifi", "romance"):
        genre = arg
    elif arg in ("dark", "hopeful", "comedic", "tragic"):
        tone = arg
    elif arg in ("short", "medium", "long"):
        length = arg

designer = (Designer(domain=NarrativeDomain(), seed=seed)
    .set("genre",       genre)
    .set("tone",        tone)
    .set("length",      length)
    .set("protagonist", "hero")
    .set("stakes",      0.7))

outline = designer.generate()
print(outline.describe())
