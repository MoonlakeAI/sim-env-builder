"""Task definitions: one registered Arena environment per entry.

This module is pure data and pre-Kit safe. `environments.register_task_
environments()` turns each spec into a first-class Arena environment named
after the task, with the spec's values as its typed-config defaults, so
Arena's environment registry is the task registry.

Adding a task for another bundled asset means adding a TaskSpec here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    """One evaluation task over a bundled library asset."""

    asset: str
    instruction: str
    # Overrides for OpenAssetEnvironmentCfg fields (placement, thresholds).
    env_overrides: dict = field(default_factory=dict)


# One task per bundled asset. Success is graded on the
# asset's openable joint (see articulation.pick_openable_joint) at >= 50% of
# its range. Floor assets override asset_x so the asset's front face lands
# roughly 0.4 m from the robot regardless of asset depth. Table assets must
# keep clear of the gripper's home pose (fingertips near x 0.5, z 0.3, see
# environments.PLACEMENT_PRESETS): tall assets keep their front face behind
# x 0.55, short ones (top below the fingertips) may sit closer.
TASK_SPECS: dict[str, TaskSpec] = {
    # -- floor placement -----------------------------------------------------
    "open_dishwasher": TaskSpec(
        asset="dishwasher",
        instruction="open the dishwasher",
        env_overrides={"asset_x": 0.72},
    ),
    "open_toilet": TaskSpec(
        asset="toilet",
        instruction="open the toilet lid",
        env_overrides={"asset_x": 0.60},
    ),
    "open_vending_machine": TaskSpec(
        asset="vending_machine",
        instruction="open the vending machine door",
        env_overrides={"asset_x": 0.90},
    ),
    "open_popcorn_machine": TaskSpec(
        asset="popcorn_machine",
        instruction="open the popcorn machine door",
        env_overrides={"asset_x": 0.68},
    ),
    "open_slushie_machine": TaskSpec(
        asset="slushie_machine",
        instruction="lift the slushie machine lid",
        env_overrides={"asset_x": 0.72},
    ),
    "open_spreader": TaskSpec(
        asset="broadcast_spreader",
        instruction="open the spreader lid",
        env_overrides={"asset_x": 0.85},
    ),
    "open_toolbox": TaskSpec(
        asset="stacking_modular_toolbox",
        instruction="open the toolbox top lid",
        env_overrides={"asset_x": 0.60},
    ),
    "rotate_compressor_knob": TaskSpec(
        asset="pancake_air_compressor",
        instruction="rotate the air compressor regulator knob",
        env_overrides={"asset_x": 0.62},
    ),
    # -- table placement -----------------------------------------------------
    "open_printer": TaskSpec(
        asset="office_printer",
        instruction="open the printer lid",
        env_overrides={"placement": "table"},
    ),
    "tilt_mixer": TaskSpec(
        asset="stand_mixer",
        instruction="tilt the mixer head up",
        env_overrides={"placement": "table", "asset_x": 0.73},
    ),
    "open_kettle": TaskSpec(
        asset="glass_electric_kettle",
        instruction="open the kettle lid",
        env_overrides={"placement": "table", "asset_x": 0.60},
    ),
    "open_multi_cooker": TaskSpec(
        asset="multi_cooker_crisper_lid",
        instruction="open the multi cooker lid",
        env_overrides={"placement": "table", "asset_x": 0.72},
    ),
    "open_soda_maker": TaskSpec(
        asset="soda_maker",
        instruction="open the soda maker gas cylinder door",
        env_overrides={"placement": "table", "asset_x": 0.69},
    ),
    "lift_blender_lid": TaskSpec(
        asset="blender",
        instruction="lift the blender lid",
        env_overrides={"placement": "table", "asset_x": 0.65},
    ),
    "lift_spinner_lid": TaskSpec(
        asset="salad_spinner",
        instruction="lift the salad spinner lid",
        env_overrides={"placement": "table", "asset_x": 0.62},
    ),
    "rotate_espresso_dial": TaskSpec(
        asset="espresso_machine",
        instruction="rotate the espresso machine steam dial",
        env_overrides={"placement": "table", "asset_x": 0.73},
    ),
    "lift_guillotine_blade": TaskSpec(
        asset="paper_guillotine_cutter",
        instruction="lift the paper cutter blade",
        env_overrides={"placement": "table", "asset_x": 0.68},
    ),
    "press_hole_punch": TaskSpec(
        asset="hole_punch",
        instruction="press down the hole punch lever",
        env_overrides={"placement": "table", "asset_x": 0.55},
    ),
    "twist_pill_cap": TaskSpec(
        asset="pill_bottle",
        instruction="twist open the pill bottle cap",
        env_overrides={"placement": "table", "asset_x": 0.50},
    ),
    "rotate_multimeter_dial": TaskSpec(
        asset="multimeter_probes",
        instruction="rotate the multimeter dial",
        env_overrides={"placement": "table", "asset_x": 0.50},
    ),
}
