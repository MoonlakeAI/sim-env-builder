"""Articulation introspection for sim-ready USD assets.

Generated asset packages ship physics articulations (revolute / prismatic
joints with limits) inside their USD. This library treats that metadata as
its ground truth:

- milestone definitions ("open the cabinet door" == door_pivot >= 45 deg)
  come from the joint inventory, never hardcoded per task;
- suggested instruction prompts come from joint names and ranges;
- rollout scoring reads sim joint state and grades it against joint limits,
  with no VLM judge or human labeling.

Parsing works in two modes:
- `pxr` (OpenUSD) when available: authoritative, used inside Isaac.
- a text fallback for flat ``.usda`` layers, so prompt
  suggestion and milestone listing work in a plain venv without Isaac.
"""

from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

REVOLUTE = "revolute"
PRISMATIC = "prismatic"

# Prismatic joints with a full range below this (meters) read as "press"
# rather than "pull"; typical button joints travel a few millimeters.
PRESS_TRAVEL_MAX_M = 0.02


@dataclass
class JointSpec:
    """One degree of freedom read from the asset USD."""

    name: str                 # joint prim name, e.g. "door_pivot"
    joint_type: str           # REVOLUTE | PRISMATIC
    lower: float              # deg (revolute, USD convention) or m (prismatic)
    upper: float
    body0: str = ""           # parent body prim path
    body1: str = ""           # child body prim path
    prim_path: str = ""
    axis: str = "Z"           # USD physics:axis in the joint local frame
    # Joint frames relative to body0 / body1 (USD quat is wxyz).
    local_pos0: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rot0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    local_pos1: tuple[float, float, float] = (0.0, 0.0, 0.0)
    local_rot1: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    # Joint value in the closed/idle state. The USD does not author a rest
    # state, so the default assumes 0 (clamped into the limits); during a
    # rollout the tracker overwrites it with the first observed sim value.
    rest: float | None = None

    def __post_init__(self):
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower

    @property
    def range(self) -> float:
        return self.upper - self.lower

    @property
    def units(self) -> str:
        return "deg" if self.joint_type == REVOLUTE else "m"

    @property
    def rest_value(self) -> float:
        if self.rest is not None:
            return min(max(self.rest, self.lower), self.upper)
        return min(max(0.0, self.lower), self.upper)

    @property
    def max_excursion(self) -> float:
        """Largest displacement available from the rest state."""
        rest = self.rest_value
        return max(self.upper - rest, rest - self.lower)

    def fraction(self, value: float) -> float:
        """Normalized progress: displacement from rest over the largest
        available excursion. Direction-agnostic, so joints that open by
        decreasing (or in either direction of a signed range) grade the
        same way as joints that open by increasing."""
        if self.max_excursion <= 0:
            return 0.0
        return max(0.0, min(1.0, abs(value - self.rest_value) / self.max_excursion))

    def displacement(self, value: float) -> float:
        """Absolute displacement from the rest state, in joint units."""
        return abs(value - self.rest_value)

    @property
    def open_value(self) -> float:
        """Joint value at the largest available excursion from rest."""
        rest = self.rest_value
        if self.upper - rest >= rest - self.lower:
            return self.upper
        return self.lower


@dataclass
class Milestone:
    """A checkable sub-goal derived from one joint."""

    joint: JointSpec
    verb: str                 # open / pull out / press / slide ...
    part: str                 # human name of the moving part, e.g. "door"
    threshold: float          # displacement from rest that counts as achieved (joint units)
    threshold_fraction: float # threshold as fraction of the max excursion from rest

    @property
    def name(self) -> str:
        return f"{self.verb} the {self.part}"

    def instruction(self, asset_name: str) -> str:
        """DROID-vernacular instruction for this milestone."""
        if self.part.startswith(asset_name):
            return f"{self.verb} the {self.part}"
        return f"{self.verb} the {asset_name} {self.part}"

    def grade(self, peak_value: float) -> dict:
        """Grade a rollout given the joint value at peak displacement from
        rest. Reports progress metrics (progress fraction and peak
        displacement) alongside the achieved boolean.
        """
        progress = self.joint.fraction(peak_value)
        out = {
            "milestone": self.name,
            "joint": self.joint.name,
            "achieved": bool(progress >= self.threshold_fraction),
            "progress": round(progress, 4),
            "threshold": self.threshold,
        }
        key = (
            f"max_{self.joint.name.removesuffix('_pivot')}_"
            + ("angle_deg" if self.joint.joint_type == REVOLUTE else "travel_m")
        )
        out[key] = round(self.joint.displacement(peak_value), 4)
        return out


def _humanize(joint_name: str) -> str:
    """'door_pivot' -> 'door', 'pivot_door' -> 'door', 'lower_rack_pivot' -> 'lower rack'."""
    stem = re.sub(r"_?(pivot|joint|hinge)(_\d+)?$", "", joint_name)
    stem = re.sub(r"^(pivot|joint|hinge)_", "", stem)
    return stem.replace("_", " ").strip() or joint_name


def _verb_for(joint: JointSpec) -> str:
    """Pick the action verb from joint type, range, and name hints."""
    name = joint.name.lower()
    if joint.joint_type == REVOLUTE:
        if re.search(r"swivel|yaw|twist|spin|rotate|knob|dial|caster", name):
            return "rotate"
        if re.search(r"tilt|fold|recline", name):
            return "tilt"
        if re.search(r"lever|trigger", name):
            return "flip"
        return "open"
    if joint.range <= PRESS_TRAVEL_MAX_M or re.search(r"button|press", name):
        return "press"
    if re.search(r"lift|raise", name):
        return "lift"
    return "pull out"


def derive_milestones(
    joints: list[JointSpec],
    threshold_fraction: float = 0.5,
    overrides: dict[str, float] | None = None,
) -> list[Milestone]:
    """Turn the joint inventory into milestones.

    `threshold_fraction`: fraction of the joint range that counts as
    achieved (default 50%: a 0-90 deg door hinge passes at 45 deg).
    `overrides`: per-joint absolute thresholds in joint units, e.g.
    ``{"door_pivot": 45.0}``.
    """
    milestones = []
    for j in joints:
        if j.range <= 0:
            continue
        if j.max_excursion <= 0:
            continue
        if overrides and j.name in overrides:
            threshold = abs(overrides[j.name])
            frac = threshold / j.max_excursion
        else:
            frac = threshold_fraction
            # A button either gets pressed or it doesn't.
            if _verb_for(j) == "press":
                frac = 0.8
            threshold = frac * j.max_excursion
        milestones.append(
            Milestone(
                joint=j,
                verb=_verb_for(j),
                part=_humanize(j.name),
                threshold=threshold,
                threshold_fraction=frac,
            )
        )
    # Big motions first: doors before buttons.
    order = {REVOLUTE: 0, PRISMATIC: 1}
    milestones.sort(key=lambda m: (order[m.joint.joint_type], -m.joint.range))
    return milestones


def pick_openable_joint(joints: list[JointSpec]) -> JointSpec:
    """Choose the joint that represents "opening" the asset.

    Preference order: a joint named like a door/lid/hatch, then the
    largest-range revolute joint, then the largest-range joint overall.
    When several lids share a range, prefer the top/upper one so a stacked
    toolbox opens the lid that is not load-bearing.
    """
    if not joints:
        raise ValueError("asset has no articulation joints")
    for pattern in ("door", "lid", "hatch", "cover"):
        # Whole-word match on name parts: "slide" must not read as "lid".
        named = [j for j in joints if pattern in j.name.lower().split("_")]
        if named:
            for hint in ("top", "upper"):
                hinted = [j for j in named if hint in j.name.lower().split("_")]
                if hinted:
                    return max(hinted, key=lambda j: j.range)
            return max(named, key=lambda j: j.range)
    revolute = [j for j in joints if j.joint_type == REVOLUTE]
    return max(revolute or joints, key=lambda j: j.range)


_INSTRUCTION_STOP = frozenset(
    {"the", "a", "an", "and", "then", "it", "to", "up", "down", "out", "open"}
)


def match_joint_for_instruction(
    joints: list[JointSpec], instruction: str, asset_name: str = ""
) -> JointSpec:
    """Pick the joint a task instruction is referring to.

    Scores each joint by how many of its name tokens appear in the
    instruction (after dropping the asset name and stopwords). Falls back
    to `pick_openable_joint` when nothing distinctive matches — so
    "open the dishwasher" still resolves to the door.
    """
    if not joints:
        raise ValueError("asset has no articulation joints")
    instr_tokens = set(re.findall(r"[a-z0-9]+", instruction.lower()))
    instr_tokens -= _INSTRUCTION_STOP
    instr_tokens -= set(re.findall(r"[a-z0-9]+", asset_name.lower().replace("-", "_")))

    def score(joint: JointSpec) -> tuple[int, float]:
        name_tokens = set(re.findall(r"[a-z0-9]+", joint.name.lower()))
        name_tokens -= {"pivot", "joint", "hinge"}
        name_tokens |= set(_humanize(joint.name).split())
        return (len(instr_tokens & name_tokens), joint.range)

    ranked = max(joints, key=score)
    if score(ranked)[0] == 0:
        return pick_openable_joint(joints)
    return ranked


def moving_body_paths(joint: JointSpec, joints: list[JointSpec]) -> list[str]:
    """Prim paths that must move with `joint`: its child body plus anything
    whose parent chain goes through that body (buttons on a door, …)."""
    children: dict[str, list[str]] = {}
    for spec in joints:
        if spec.body0 and spec.body1:
            children.setdefault(spec.body0, []).append(spec.body1)
    out: list[str] = []
    stack = [joint.body1] if joint.body1 else []
    seen: set[str] = set()
    while stack:
        path = stack.pop()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
        stack.extend(children.get(path, []))
    return out


# --------------------------------------------------------------------------
# USD parsing
# --------------------------------------------------------------------------

_JOINT_BLOCK = re.compile(
    r'def\s+Physics(Revolute|Prismatic)Joint\s+"([^"]+)"\s*(?:\([^)]*\))?\s*\{(.*?)\n\s*\}',
    re.DOTALL,
)
_FLOAT_ATTR = re.compile(r"float\s+physics:(lowerLimit|upperLimit)\s*=\s*([-\d.eE+]+)")
_REL_ATTR = re.compile(r"rel\s+physics:(body0|body1)\s*=\s*<([^>]*)>")
_AXIS_ATTR = re.compile(r'token\s+physics:axis\s*=\s*"([XYZ])"')
_POINT_ATTR = re.compile(
    r"point3f\s+physics:(localPos0|localPos1)\s*=\s*\(\s*([^)]+?)\s*\)"
)
_QUAT_ATTR = re.compile(
    r"quatf\s+physics:(localRot0|localRot1)\s*=\s*\(\s*([^)]+?)\s*\)"
)


def _floats(csv: str) -> tuple[float, ...]:
    return tuple(float(p.strip()) for p in csv.split(",") if p.strip())


def _parse_usda_text(text: str) -> list[JointSpec]:
    joints = []
    for match in _JOINT_BLOCK.finditer(text):
        kind, name, body = match.groups()
        attrs = dict(_FLOAT_ATTR.findall(body))
        rels = dict(_REL_ATTR.findall(body))
        points = {key: _floats(val) for key, val in _POINT_ATTR.findall(body)}
        quats = {key: _floats(val) for key, val in _QUAT_ATTR.findall(body)}
        axis_match = _AXIS_ATTR.search(body)
        joints.append(
            JointSpec(
                name=name,
                joint_type=REVOLUTE if kind == "Revolute" else PRISMATIC,
                lower=float(attrs.get("lowerLimit", "-inf")),
                upper=float(attrs.get("upperLimit", "inf")),
                body0=rels.get("body0", ""),
                body1=rels.get("body1", ""),
                axis=axis_match.group(1) if axis_match else "Z",
                local_pos0=points.get("localPos0", (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
                local_rot0=quats.get("localRot0", (1.0, 0.0, 0.0, 0.0)),  # type: ignore[arg-type]
                local_pos1=points.get("localPos1", (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
                local_rot1=quats.get("localRot1", (1.0, 0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            )
        )
    return [j for j in joints if math.isfinite(j.lower) and math.isfinite(j.upper)]


def _pxr_xyz(value) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    return (float(value[0]), float(value[1]), float(value[2]))


def _pxr_quat_wxyz(value) -> tuple[float, float, float, float]:
    if value is None:
        return (1.0, 0.0, 0.0, 0.0)
    imag = value.GetImaginary()
    return (float(value.GetReal()), float(imag[0]), float(imag[1]), float(imag[2]))


def _parse_with_pxr(usd_path: Path) -> list[JointSpec]:
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    stage = Usd.Stage.Open(str(usd_path))
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            api, kind = UsdPhysics.RevoluteJoint(prim), REVOLUTE
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            api, kind = UsdPhysics.PrismaticJoint(prim), PRISMATIC
        else:
            continue
        lower = api.GetLowerLimitAttr().Get()
        upper = api.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            continue
        b0 = api.GetBody0Rel().GetTargets()
        b1 = api.GetBody1Rel().GetTargets()
        axis = api.GetAxisAttr().Get() or "Z"
        joints.append(
            JointSpec(
                name=prim.GetName(),
                joint_type=kind,
                lower=float(lower),
                upper=float(upper),
                body0=str(b0[0]) if b0 else "",
                body1=str(b1[0]) if b1 else "",
                prim_path=str(prim.GetPath()),
                axis=str(axis),
                local_pos0=_pxr_xyz(api.GetLocalPos0Attr().Get()),
                local_rot0=_pxr_quat_wxyz(api.GetLocalRot0Attr().Get()),
                local_pos1=_pxr_xyz(api.GetLocalPos1Attr().Get()),
                local_rot1=_pxr_quat_wxyz(api.GetLocalRot1Attr().Get()),
            )
        )
    return joints


def parse_articulation(usd_path: str | Path) -> list[JointSpec]:
    """Read the joint inventory from a USD asset package.

    Accepts .usd/.usda/.usdc/.usdz. Uses pxr when importable, otherwise
    falls back to scanning the usda text layer (works for flat
    Blender-exported packages).
    """
    usd_path = Path(usd_path)
    try:
        return _parse_with_pxr(usd_path)
    except ImportError:
        pass
    if usd_path.suffix == ".usdz":
        with zipfile.ZipFile(usd_path) as z:
            names = [n for n in z.namelist() if n.endswith(".usda")]
            if not names:
                raise ValueError(
                    f"{usd_path} has no usda text layer; install OpenUSD (pxr) to parse it"
                )
            text = z.read(names[0]).decode("utf-8", errors="replace")
    else:
        text = usd_path.read_text(errors="replace")
    return _parse_usda_text(text)
