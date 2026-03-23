"""
audio_listener._paths — sys.path bootstrap.

Adds music_generator and procgen package roots to sys.path so audio_listener
works regardless of the working directory or installation state.
"""

import sys
import pathlib


def ensure_paths() -> None:
    here     = pathlib.Path(__file__).resolve().parent.parent  # .../audio_listener/
    projects = here.parent                                       # .../proc gen projects/
    for p in [
        projects / "music_generator",
        projects / "procgen",
    ]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
