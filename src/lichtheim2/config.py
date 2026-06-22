from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    sound_input_size: int
    motor_output_size: int
    iSMG_hidden_size: int
    mSTG_hidden_size: int
    aSTG_hidden_size: int
    triangularis_hidden_size: int
    vATL_size: int
    repetition_ticks: int
    comprehension_ticks: int
    speaking_ticks: int
    sound_proj_size: int | None = None  # None = no projection (paper pathway) [Phase 3h]
    dorsal_motor_only: bool = False     # diagnostic: exclude triangularis_to_motor from motor_net [Phase 3j]


def load_config(path: str | Path) -> ModelConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    m = raw["model"]
    t = raw.get("tasks", {})
    return ModelConfig(
        sound_input_size=m["sound_input_size"],
        motor_output_size=m["motor_output_size"],
        iSMG_hidden_size=m["iSMG_hidden_size"],
        mSTG_hidden_size=m["mSTG_hidden_size"],
        aSTG_hidden_size=m["aSTG_hidden_size"],
        triangularis_hidden_size=m["triangularis_hidden_size"],
        vATL_size=m["vATL_size"],
        repetition_ticks=t.get("repetition_ticks", 6),
        comprehension_ticks=t.get("comprehension_ticks", 3),
        speaking_ticks=t.get("speaking_ticks", 3),
        sound_proj_size=m.get("sound_proj_size", None),
        dorsal_motor_only=m.get("dorsal_motor_only", False),
    )
