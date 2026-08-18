"""Per-step articulation tracking during Arena rollouts.

Bridges the sim to the pure scoring layer: reads the tracked asset's joint
positions from the Isaac Lab articulation view every step and feeds one
`EpisodeRecord` per env. Everything graded here comes from joint state and
the limits authored in the asset USD, with no VLM judge or labeling.
"""

from __future__ import annotations

import json
from pathlib import Path

from .articulation import derive_milestones, parse_articulation
from .scoring import EpisodeRecord, summarize, write_episode_json


def _asset_usd(scene_entity_name: str) -> Path | None:
    """Resolve a scene articulation to its source USD, if it is a registered
    library asset."""
    from .arena_assets import REGISTERED_LIBRARY_ASSETS  # noqa: PLC0415

    return REGISTERED_LIBRARY_ASSETS.get(scene_entity_name)


class MilestoneTracker:
    """Tracks articulation-derived milestones for one asset across all envs."""

    def __init__(self, env, asset_name: str, usd_path: Path, instruction: str, output_dir: Path):
        self.asset_name = asset_name
        self.instruction = instruction
        self.output_dir = Path(output_dir) / "milestones"
        self.articulation = env.unwrapped.scene.articulations[asset_name]
        self.num_envs = env.unwrapped.num_envs

        joints = parse_articulation(usd_path)
        self.milestones = derive_milestones(joints)
        self.label_name = asset_name.replace("_", " ")

        # Success = the milestone on the asset's openable joint (the task's
        # own success criterion), phrased like an instruction.
        from .articulation import pick_openable_joint  # noqa: PLC0415

        target_joint = pick_openable_joint(joints)
        target = next(m for m in self.milestones if m.joint.name == target_joint.name)
        self.target_milestone = target.instruction(self.label_name)
        # Long-horizon mode: an instruction like "open the cabinet and then
        # close it" additionally requires the joint to return near closed.
        self.require_close = "close" in instruction.lower()

        sim_names = list(self.articulation.joint_names)
        self.joint_indices = {
            m.joint.name: sim_names.index(m.joint.name)
            for m in self.milestones
            if m.joint.name in sim_names
        }

        self._episode_counter = 0
        self._records = [self._new_record(env_id) for env_id in range(self.num_envs)]
        self.results: list[dict] = []

    def _new_record(self, env_id: int) -> EpisodeRecord:
        record = EpisodeRecord(
            milestones=self.milestones,
            instruction=self.instruction,
            episode=self._episode_counter,
            asset_name=self.label_name,
        )
        self._episode_counter += 1
        return record

    def update(self) -> None:
        """Call once per env.step()."""
        joint_pos = self.articulation.data.joint_pos.detach().cpu().numpy()
        for env_id in range(self.num_envs):
            self._records[env_id].update(
                {name: joint_pos[env_id, idx] for name, idx in self.joint_indices.items()}
            )

    def finish_episodes(self, env_ids) -> None:
        """Grade and persist episodes for the envs that just terminated."""
        for env_id in env_ids:
            env_id = int(env_id)
            record = self._records[env_id]
            if record.steps == 0:
                continue
            result = record.result(
                target_milestone=self.target_milestone, require_close=self.require_close
            )
            result["env_id"] = env_id
            self.results.append(result)
            path = write_episode_json(result, self.output_dir)
            if result["limit_violations"]:
                status = f"invalid, joint limits exceeded: {sorted(result['limit_violations'])}"
            else:
                status = "success" if result["success"] else "no success"
            print(f"[sim-env-builder] episode {result['episode']} ({status}) -> {path}")
            self._records[env_id] = self._new_record(env_id)

    def finalize(self) -> str:
        """Write the run summary and return the pass/fail line."""
        line = summarize(self.results) if self.results else "[FAIL] no completed episodes"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "asset": self.asset_name,
                    "instruction": self.instruction,
                    "target_milestone": self.target_milestone,
                    "episodes": self.results,
                    "summary": line,
                },
                indent=2,
            )
            + "\n"
        )
        return line


def maybe_create_tracker(env, instruction: str, output_dir) -> MilestoneTracker | None:
    """Attach a tracker if the scene contains a registered library asset."""
    for asset_name in env.unwrapped.scene.articulations:
        usd_path = _asset_usd(asset_name)
        if usd_path is not None:
            return MilestoneTracker(env, asset_name, usd_path, instruction, Path(output_dir))
    return None
