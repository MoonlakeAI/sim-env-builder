"""Arena environment factories for library assets.

Import only after the Isaac SimulationApp is running. The rollout runner
loads this module via ``--external_environment_class_path`` (or directly),
which fires the ``@register_environment`` / ``@register_asset`` decorators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentCfg,
    ArenaEnvironmentFactory,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

# Lighting presets, applied to Arena's registered light assets ("light" dome +
# "directional_light").
LIGHTING_PRESETS: dict[str, dict] = {
    "default": {"dome_intensity": 1500.0, "directional_intensity": 1000.0},
    "dim": {"dome_intensity": 300.0, "directional_intensity": 0.0},
    "bright": {"dome_intensity": 2800.0, "directional_intensity": 3000.0},
}

# Placement patterns, following the two scene arrangements Arena's DROID
# environments use. Values fill the pose fields left unset in the config.
# - "floor": the asset stands on the (invisible) ground plane; the robot is
#   lowered relative to it, mirroring upstream's franka_put_and_close_door,
#   so a bottom-hinged part can swing past the robot's tabletop plane.
# - "table": the asset sits on Arena's maple table with the robot at the
#   embodiment's authored origin, mirroring upstream's tabletop arrangement
#   (droid_table_multi_object_placement / pick_and_place_maple_table). The
#   maple-table scene authors its tabletop at z = 0.003 (top spans
#   x 0.20-0.90, y -0.48-0.52; legs reach the ground at z = -0.697), so the
#   asset base lands on the surface with 2 mm of clearance. Library assets
#   author their origin at the geometry base (bbox min z = 0).
#   The DROID home pose parks the gripper fingertips near x 0.5, z 0.3; a
#   tall asset spawned through that spot gets depenetrated violently enough
#   to tear the gripper joints apart. The default x keeps a tall asset's
#   front face behind the fingertips; short assets (top below ~0.28 m, under
#   the fingertips) may override asset_x to sit closer (see tasks.TASK_SPECS).
PLACEMENT_PRESETS: dict[str, dict] = {
    "floor": {"asset_x": 0.60, "asset_z": -0.70, "robot_z": -0.35},
    "table": {"asset_x": 0.70, "asset_z": 0.005, "robot_z": 0.0},
}


@dataclass
class OpenAssetEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the Franka(DROID) open-asset environment."""

    # Bundled library asset to evaluate (assets/library/<asset>.usdz).
    asset: str = ""
    embodiment: str = "droid_abs_joint_pos"
    # HDRI environment map for the dome light (registered name: home_office,
    # hotel_room, garage, or any Arena-registered HDR). Empty string = plain
    # grey dome.
    hdr: str = "home_office"
    # Lighting preset: default | dim | bright (see LIGHTING_PRESETS).
    lighting: str = "default"
    # Placement pattern: floor | table (see PLACEMENT_PRESETS).
    placement: str = "floor"
    # Asset root pose. Library assets face -Y; the build rotation turns the
    # front toward the robot. Pose fields left as None resolve from the
    # placement preset; per-asset overrides belong in the task configuration
    # (see tasks.TASK_SPECS).
    asset_x: float | None = None
    asset_y: float = 0.0
    asset_z: float | None = None
    # z of the robot base (the top of the DROID stand).
    robot_z: float | None = None
    ground_z: float = -0.70
    # Success threshold on the openable joint's normalized openness.
    openness_threshold: float = 0.5
    reset_openness: float = 0.0
    episode_length_s: float = 25.0


class OpenAssetEnvironment(ArenaEnvironmentFactory[OpenAssetEnvironmentCfg]):
    """Open a library asset's openable part with a DROID Franka."""

    name: str = "open_asset"
    _legacy_argparse_cfg_type = OpenAssetEnvironmentCfg
    # Default language instruction; task subclasses set it from their spec.
    instruction: str = ""

    def build(self, cfg: OpenAssetEnvironmentCfg) -> IsaacLabArenaEnvironment:
        from isaaclab.envs.common import ViewerCfg
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        import sim_env_builder.arena_assets  # noqa: F401  (registers HDRs)
        from sim_env_builder.arena_assets import register_library_asset

        assert cfg.asset, "--asset is required (a bundled library asset name)"
        preset = PLACEMENT_PRESETS[cfg.placement]
        asset_x = preset["asset_x"] if cfg.asset_x is None else cfg.asset_x
        asset_z = preset["asset_z"] if cfg.asset_z is None else cfg.asset_z
        robot_z = preset["robot_z"] if cfg.robot_z is None else cfg.robot_z

        # The returned class is authoritative: Arena's registry may hold an
        # unrelated upstream asset under the same name (e.g. "stand_mixer").
        asset = register_library_asset(cfg.asset)()
        asset.set_initial_pose(
            Pose(
                position_xyz=(asset_x, cfg.asset_y, asset_z),
                rotation_xyzw=(0.0, 0.0, -0.7071068, 0.7071068),
            )
        )

        if cfg.placement == "table":
            # Arena's maple-table scene (tabletop plus invisible ground
            # collision), the background of the upstream DROID tabletop
            # environments.
            support = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        else:
            # The ground plane is collision-only, so the HDRI dome renders
            # below the horizon and provides the visible floor.
            import isaaclab.sim as sim_utils

            support = self.asset_registry.get_asset_by_name("ground_plane")(
                spawner_cfg=sim_utils.GroundPlaneCfg(visible=False),
            )
            support.set_initial_pose(Pose(position_xyz=(0.0, 0.0, cfg.ground_z)))

        lighting = LIGHTING_PRESETS[cfg.lighting]
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(lighting["dome_intensity"])
        if cfg.hdr:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()
        directional_light.set_intensity(lighting["directional_intensity"])

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
        )
        # The composed robot+stand USD keeps the gripper meshes instanceable,
        # which the camera render path draws frozen at their authored pose
        # (see paths.disable_usd_instancing).
        from sim_env_builder.paths import disable_usd_instancing

        embodiment.scene_config.robot.spawn.usd_path = disable_usd_instancing(
            embodiment.scene_config.robot.spawn.usd_path
        )
        embodiment.set_initial_pose(
            Pose(position_xyz=(0.0, 0.0, robot_z), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )

        task = OpenDoorTask(
            openable_object=asset,
            openness_threshold=cfg.openness_threshold,
            reset_openness=cfg.reset_openness,
            episode_length_s=cfg.episode_length_s,
            task_description=self.instruction or f"open the {cfg.asset.replace('_', ' ')}",
        )

        # Near-level pitch keeps the HDRI background above the ground plane's
        # horizon in viewport videos; the table view is raised to tabletop height.
        viewer = {
            "floor": ViewerCfg(eye=(-1.6, -1.3, 0.45), lookat=(0.9, 0.2, 0.1)),
            "table": ViewerCfg(eye=(-1.4, -1.2, 0.55), lookat=(0.8, 0.2, 0.15)),
        }[cfg.placement]

        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = viewer
            return env_cfg

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[support, light, directional_light, asset]),
            task=task,
            env_cfg_callback=_set_viewer_cfg,
        )


def register_task_environments() -> None:
    """Register one Arena environment per entry in tasks.TASK_SPECS.

    Each task becomes a first-class registered environment (Arena's native
    mechanism): its name is the task name and its typed config carries the
    spec's asset and placement as defaults.
    """
    import dataclasses

    from isaaclab_arena.assets.register import register_environment
    from isaaclab_arena.assets.registries import EnvironmentRegistry

    from .tasks import TASK_SPECS

    registry = EnvironmentRegistry()
    for task_name, spec in TASK_SPECS.items():
        if registry.is_registered(task_name):
            continue
        defaults = {"asset": spec.asset, **spec.env_overrides}
        unknown = set(defaults) - {f.name for f in dataclasses.fields(OpenAssetEnvironmentCfg)}
        assert not unknown, f"task '{task_name}' overrides unknown cfg fields: {unknown}"
        cfg_cls = dataclasses.make_dataclass(
            f"{task_name.title().replace('_', '')}Cfg",
            [
                (f.name, f.type, dataclasses.field(default=defaults.get(f.name, f.default)))
                for f in dataclasses.fields(OpenAssetEnvironmentCfg)
            ],
            bases=(OpenAssetEnvironmentCfg,),
        )
        # The registry reads the concrete cfg type from a direct generic base.
        import types

        def _body(ns, task_name=task_name, cfg_cls=cfg_cls, spec=spec):
            ns["name"] = task_name
            ns["_legacy_argparse_cfg_type"] = cfg_cls
            ns["instruction"] = spec.instruction

        env_cls = types.new_class(
            f"{task_name.title().replace('_', '')}Environment",
            (OpenAssetEnvironment, ArenaEnvironmentFactory[cfg_cls]),
            exec_body=_body,
        )
        register_environment(env_cls)
