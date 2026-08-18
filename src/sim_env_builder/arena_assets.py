"""Arena asset wrappers for the bundled asset library.

IMPORTANT: import this module only after the Isaac SimulationApp is running
(inside an environment factory's ``build()`` or after ``SimulationAppContext``
starts): Arena asset classes pull in pxr/omni at import time.

Nothing hardcodes the openable joint or threshold: `parse_articulation`
reads them from the articulation authored in the asset USD. For a
door-bearing asset this resolves to its door joint, and the default 0.5
openness threshold means "open at least half of the authored range".
"""

from __future__ import annotations

from pathlib import Path

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.assets.hdr_image import HDRImage
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset, register_hdr

from .articulation import parse_articulation, pick_openable_joint
from .paths import ASSETS_DIR


class _LocalHDR(HDRImage):
    """Repo-bundled Poly Haven CC0 HDRIs (see assets/backgrounds/*.txt)."""

    name: str
    tags: list[str]
    filename: str

    def __init__(self):
        super().__init__(
            name=self.name,
            texture_file=str(ASSETS_DIR / "backgrounds" / self.filename),
            tags=self.tags,
            texture_format="latlong",
        )


@register_hdr
class HomeOfficeHDR(_LocalHDR):
    name = "home_office"
    tags = ["indoor"]
    filename = "home_office.exr"


@register_hdr
class HotelRoomHDR(_LocalHDR):
    name = "hotel_room"
    tags = ["indoor"]
    filename = "hotel_room_4k.hdr"


@register_hdr
class GarageHDR(_LocalHDR):
    name = "garage"
    tags = ["indoor", "industrial"]
    filename = "garage_2k.hdr"


class ArticulatedLibraryAsset(LibraryObject, Openable):
    """Base class for articulated library assets."""

    object_type = ObjectType.ARTICULATION
    tags = ["object", "openable", "articulated", "library"]
    openable_joint_name: str
    openable_threshold = 0.5  # progress fraction; 0.5 of a 0-90 deg door == 45 deg

    def __init__(self, instance_name=None, prim_path=None, initial_pose=None):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_threshold=self.openable_threshold,
        )

    def _generate_articulation_cfg(self):
        # Library exports ship frictionless joints; without passive friction
        # gravity alone moves parts (a bottom-hinged door swings tens of
        # degrees, measured with a zero-action probe). Model joint friction
        # so parts stay put until the robot moves them.
        import isaaclab.sim as sim_utils  # noqa: PLC0415
        from isaaclab.actuators import ImplicitActuatorCfg  # noqa: PLC0415

        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "passive_friction": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=5.0,
                friction=12.0,
            ),
        }
        # Upstream spawns the raw UsdFileCfg, inheriting whatever physics the
        # export authored. Parts that rest in contact (a mixer head on its
        # bowl, a toilet lid on its seat) then fight their own colliders: the
        # depenetration kick swings joints open at spawn, and a robot nudge
        # tears them past their limits (observed at >1000 deg). Disable
        # self-collisions, as Isaac Lab robot configs do, and resolve any
        # remaining penetration gently with enough solver iterations to hold
        # the joint limits.
        cfg.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        )
        cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=1.0,
        )
        return cfg

    def _joint_frames(self, env, asset_cfg):
        from isaaclab.managers import SceneEntityCfg  # noqa: PLC0415
        from isaaclab_arena.utils import joint_utils  # noqa: PLC0415

        if asset_cfg is None:
            asset_cfg = SceneEntityCfg(self.name)
        asset_cfg = self._add_joint_name_to_scene_entity_cfg(asset_cfg)
        articulation = joint_utils.get_articulation_from_asset_cfg(env, asset_cfg)
        joint_index = joint_utils.get_joint_index_from_asset_cfg(env, asset_cfg)
        lower, upper = joint_utils.get_joint_position_limits_from_articulation(
            articulation, joint_index
        )
        rest = articulation.data.default_joint_pos[0, joint_index]
        return asset_cfg, lower, upper, rest

    def get_openness(self, env, asset_cfg=None):
        """Openness as displacement from the authored rest pose.

        Upstream normalizes the raw joint position within [lower, upper],
        which reads "already open" for joints whose rest pose sits at a limit
        or inside a signed range (a +-150 deg dial rests at 0). Displacement
        from rest over the largest available excursion mirrors the milestone
        grading in articulation.JointSpec.fraction, so Arena's success
        termination and the milestone verdict agree.
        """
        from isaaclab_arena.utils import joint_utils  # noqa: PLC0415

        asset_cfg, lower, upper, rest = self._joint_frames(env, asset_cfg)
        position = joint_utils.get_unnormalized_joint_position(env, asset_cfg)
        excursion = max(float(upper - rest), float(rest - lower), 1e-9)
        return ((position - rest).abs() / excursion).clamp(0.0, 1.0)

    # Sim steps the openness must stay above threshold before the success
    # termination fires (50 Hz control -> 0.3 s).
    SUCCESS_HOLD_STEPS = 15

    def is_open(self, env, asset_cfg=None, threshold=None):
        """Success only after openness holds above threshold for a few steps.

        A physics blowup (a joint torn past its limit by contact, or a spawn
        depenetration impulse) corrupts the articulation's joint readings for
        a step or two. Upstream's instantaneous check turns such spikes into
        false success terminations, and the reset inside env.step() hides the
        spike from the milestone tracker, so the run logs success while the
        milestone verdict says fail. Requiring the openness to hold filters
        one-step garbage while leaving genuine opens (which persist) intact.
        """
        import torch  # noqa: PLC0415

        used = self.openable_threshold if threshold is None else threshold
        open_now = self.get_openness(env, asset_cfg) > used
        streak = getattr(self, "_open_streak", None)
        if streak is None or streak.shape != open_now.shape:
            streak = torch.zeros_like(open_now, dtype=torch.int32)
        self._open_streak = torch.where(open_now, streak + 1, torch.zeros_like(streak))
        return self._open_streak >= self.SUCCESS_HOLD_STEPS

    def rotate_revolute_joint(self, env, env_ids, asset_cfg=None, percentage=0.0):
        """Drive the joint from rest (0.0) toward its widest limit (1.0).

        Keeps the reset event consistent with the rest-relative openness:
        upstream's percentage-of-range reset would slam a signed-range joint
        to one of its limits instead of leaving it at rest.
        """
        assert 0.0 <= percentage <= 1.0, "Percentage must be between 0.0 and 1.0"
        from isaaclab_arena.utils import joint_utils  # noqa: PLC0415

        asset_cfg, lower, upper, rest = self._joint_frames(env, asset_cfg)
        open_limit = upper if float(upper - rest) >= float(rest - lower) else lower
        target = float(rest) + percentage * (float(open_limit) - float(rest))
        joint_utils.set_unnormalized_joint_position(env, asset_cfg, target, env_ids)


# Registered library assets: Arena asset name -> extracted USD. The tracker
# uses this to resolve a scene articulation back to its source package.
REGISTERED_LIBRARY_ASSETS: dict[str, "Path"] = {}

# Library asset name -> registered Arena asset class (idempotency cache).
_LIBRARY_ASSET_CLASSES: dict[str, type] = {}


def register_library_asset(asset_name: str) -> type:
    """Register a bundled library asset as an Arena articulation (idempotent).

    The openable joint is picked from the articulation authored in the USD.
    The Arena asset name equals the library asset name, unless upstream Arena
    already registered a different asset under it (it ships its own
    "stand_mixer", for example); then the library asset registers as
    "<asset_name>_library". Callers must use the returned class rather than
    looking the name up in Arena's registry.
    """
    from isaaclab_arena.assets.registries import AssetRegistry  # noqa: PLC0415

    from .paths import ensure_library_extracted  # noqa: PLC0415

    if asset_name in _LIBRARY_ASSET_CLASSES:
        return _LIBRARY_ASSET_CLASSES[asset_name]

    arena_name = asset_name
    if AssetRegistry().is_registered(arena_name):
        arena_name = f"{asset_name}_library"
        assert not AssetRegistry().is_registered(arena_name), (
            f"both '{asset_name}' and '{arena_name}' are taken in Arena's registry"
        )
    usd = ensure_library_extracted(asset_name)
    joints = parse_articulation(usd)
    openable = pick_openable_joint(joints)
    cls = type(
        f"{asset_name.title().replace('_', '')}Asset",
        (ArticulatedLibraryAsset,),
        {
            "name": arena_name,
            "usd_path": str(usd),
            "openable_joint_name": openable.name,
        },
    )
    REGISTERED_LIBRARY_ASSETS[arena_name] = usd
    _LIBRARY_ASSET_CLASSES[asset_name] = cls
    register_asset(cls)
    return cls

