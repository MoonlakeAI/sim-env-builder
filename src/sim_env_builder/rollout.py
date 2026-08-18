"""Policy rollout runner: Isaac Lab Arena + PI-0.5 + articulation milestones.

Adapted from isaaclab_arena.evaluation.policy_runner, with one addition: a
MilestoneTracker that reads the tracked asset's joint state every step and
emits per-episode milestone JSON + a pass/fail line alongside Arena's own
videos, episode JSONL, metrics, and HTML report.

Run through the friendly CLI (`sim-env-builder rollout ...`) or directly:

    OMNI_KIT_ACCEPT_EULA=yes uv run --group arena python -m sim_env_builder.rollout \
        --device cuda:0 --headless --enable_cameras --num_envs 1 --num_episodes 3 \
        --policy_type isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy \
        --policy_variant pi05 --remote_host 127.0.0.1 --remote_port 8000 \
        --record_camera_video --record_viewport_video \
        --language_instruction "open the <asset>" \
        <task_name> --embodiment droid_abs_joint_pos
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def main() -> None:
    # Only AppLauncher-safe imports may happen before SimulationAppContext.
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, unknown = args_parser.parse_known_args()

    # --record_camera_video requires cameras enabled at sim startup.
    if "--record_camera_video" in unknown:
        args_cli.enable_cameras = True

    with SimulationAppContext(args_cli):
        _run(args_parser)


def main_batch(runs: list[tuple[str, list[str]]]) -> None:
    """Run several task rollouts inside one SimulationApp session.

    Kit boot (~2-3 min) dominates short evaluations, so the batch boots Isaac
    once and only rebuilds the environment between tasks. Each task keeps its
    own argv, output dir, milestone JSON, and [PASS]/[FAIL] verdict. A crash
    in one task is logged and the batch moves on to the next.
    """
    import sys
    import time
    import traceback

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    # Boot-time flags (device, cameras, headless) are shared across the batch.
    # Copy: AppLauncher appends kit settings tokens to sys.argv in place, and
    # aliasing runs[0][1] here would pollute the first task's argv.
    sys.argv = list(runs[0][1])
    args_cli, unknown = get_isaaclab_arena_cli_parser().parse_known_args()
    if "--record_camera_video" in unknown:
        args_cli.enable_cameras = True

    failures: list[str] = []
    with SimulationAppContext(args_cli):
        for i, (task_name, argv) in enumerate(runs, 1):
            print(f"[sim-env-builder] ==== task {i}/{len(runs)}: {task_name} ====", flush=True)
            start = time.monotonic()
            # Boot-only tokens: --kit_args is consumed at startup, and bare
            # --/path=value kit settings must not reach the task parser.
            sys.argv = [
                a for a in argv
                if not a.startswith("--kit_args") and not a.startswith("--/")
            ]
            try:
                _run(get_isaaclab_arena_cli_parser())
            except (Exception, SystemExit):
                traceback.print_exc()
                failures.append(task_name)
                print(f"[sim-env-builder] task '{task_name}' crashed; continuing", flush=True)
            print(
                f"[sim-env-builder] ==== task {task_name} done in "
                f"{time.monotonic() - start:.0f}s ====",
                flush=True,
            )

    if failures:
        print(f"[sim-env-builder] batch finished with {len(failures)} crashed "
              f"task(s): {', '.join(failures)}")
    else:
        print("[sim-env-builder] batch finished: all tasks ran")


def _run(args_parser) -> None:
    import torch
    import tqdm
    from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
    from isaaclab_arena.video.video_recording import (
        VideoRecordingCfg,
        timestamped_run_dir,
        wrap_env_for_video,
    )
    from isaaclab_arena_environments.cli import (
        get_arena_builder_from_cli,
        get_isaaclab_arena_environments_cli_parser,
    )

    import sim_env_builder.arena_assets  # noqa: F401  (register HDRs/assets for all envs)
    from sim_env_builder.environments import register_task_environments
    from sim_env_builder.tracking import maybe_create_tracker

    register_task_environments()

    add_policy_runner_arguments(args_parser)
    args_cli, _ = args_parser.parse_known_args()

    assert args_cli.policy_type is not None, "--policy_type is required."
    policy_cls = get_policy_cls(args_cli.policy_type)

    args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
    args_parser = add_policy_cli_args(args_parser, policy_cls)
    args_cli, hydra_overrides = args_parser.parse_known_args()
    assert_hydra_overrides(hydra_overrides, args_parser)

    if args_cli.record_camera_video:
        args_cli.enable_cameras = True

    arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)

    output_dir = timestamped_run_dir(args_cli.output_base_dir)
    print(f"[sim-env-builder] output dir: {output_dir}")
    video_cfg = VideoRecordingCfg(
        record_viewport_video=args_cli.record_viewport_video,
        record_camera_video=args_cli.record_camera_video,
        video_base_dir=output_dir,
    )
    env_cfg, env_kwargs = arena_builder.compose_manager_cfg()
    # Keep the metric recorder's HDF5 with the run artifacts (the default
    # /tmp/isaaclab/logs is shared machine-wide and may not be writable).
    if getattr(env_cfg, "recorders", None) is not None:
        env_cfg.recorders.dataset_export_dir_path = os.path.join(output_dir, "metrics_data")
    env = arena_builder.make_registered(env_cfg, env_kwargs, render_mode=video_cfg.render_mode)

    env.unwrapped.episode_recorder.set_job_name("sim_env_builder")
    env.unwrapped.episode_recorder.set_output_path(
        os.path.join(output_dir, "episode_results.jsonl")
    )

    policy = build_policy_from_cli(policy_cls, args_cli)

    if args_cli.num_steps is not None:
        num_steps, num_episodes = args_cli.num_steps, None
    elif args_cli.num_episodes is not None:
        num_steps, num_episodes = None, args_cli.num_episodes
    else:
        raise ValueError("Either --num_steps or --num_episodes must be provided")

    env = wrap_env_for_video(env, video_cfg, num_steps, num_episodes)

    instruction = env.unwrapped.get_language_instruction()
    tracker = maybe_create_tracker(env, instruction, output_dir)
    if tracker is not None:
        print(
            f"[sim-env-builder] tracking articulation of '{tracker.asset_name}': "
            f"{len(tracker.joint_indices)} joints, target milestone "
            f"'{tracker.target_milestone}'"
        )

    obs, _ = env.reset()
    policy.reset()
    policy.set_task_description(instruction)
    print(f"[sim-env-builder] instruction: {instruction!r}")

    pbar = tqdm.tqdm(total=num_episodes or num_steps, unit="episode" if num_episodes else "step")
    episodes_done = 0
    steps_done = 0
    try:
        while True:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

            if tracker is not None:
                tracker.update()

            if terminated.any() or truncated.any():
                env_ids = (terminated | truncated).nonzero().flatten()
                if tracker is not None:
                    tracker.finish_episodes(env_ids.tolist())
                policy.reset(env_ids=env_ids)
                episodes_done += env_ids.shape[0]
                if num_episodes is not None:
                    pbar.update(env_ids.shape[0])
                    if episodes_done >= num_episodes:
                        break

            steps_done += 1
            if num_steps is not None:
                pbar.update(1)
                if steps_done >= num_steps:
                    break
    finally:
        pbar.close()

    if hasattr(env.unwrapped.cfg, "metrics") and env.unwrapped.cfg.metrics is not None:
        metrics = env.unwrapped.compute_metrics()
        print(f"[sim-env-builder] arena metrics: {metrics_to_plain_python_types(metrics)}")

    if policy.is_remote:
        policy.shutdown_remote(kill_server=getattr(args_cli, "remote_kill_on_exit", False))
    env.close()

    verdict = tracker.finalize() if tracker is not None else None
    build_report(output_dir)

    print(f"[sim-env-builder] artifacts in {output_dir}: rollout MP4s, episode_results.jsonl, "
          "index.html" + (", milestones/" if tracker is not None else ""))
    if verdict is not None:
        # Print the pass/fail line last. It appears even when the policy
        # scored zero, because the milestone progress metrics are the product.
        print(verdict)



# ---------------------------------------------------------------------------
# Vendored from isaaclab_arena.evaluation, which the isaaclab-arena wheel does
# not ship. Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
# Arena imports stay inside the functions: they must not load before Kit.
# ---------------------------------------------------------------------------

_FIELDS_PROVIDED_BY_SHARED_PARSER = {"device", "num_envs"}


def get_policy_cls(policy_type: str) -> type:
    """Resolve a registered policy name or a dotted 'module.ClassName' path."""
    from importlib import import_module

    import isaaclab_arena.assets.registries

    policy_registry = isaaclab_arena.assets.registries.PolicyRegistry()
    if policy_registry.is_registered(policy_type):
        return policy_registry.get_policy(policy_type)
    assert "." in policy_type, (
        "policy_type must be a registered name or a dotted import path "
        f"'module.submodule.ClassName', got: {policy_type}"
    )
    module_path, class_name = policy_type.rsplit(".", 1)
    return getattr(import_module(module_path), class_name)


def add_policy_cli_args(
    parser: argparse.ArgumentParser, policy_type: type
) -> argparse.ArgumentParser:
    import dataclasses

    import isaaclab_arena.assets.registries
    import isaaclab_arena.cli.dataclass_cli

    registry = isaaclab_arena.assets.registries.PolicyRegistry()
    policy_cfg_type = registry.get_policy_cfg_type(policy_type)
    cfg_field_names = {config_field.name for config_field in dataclasses.fields(policy_cfg_type)}
    shared_fields = cfg_field_names.intersection(_FIELDS_PROVIDED_BY_SHARED_PARSER)
    isaaclab_arena.cli.dataclass_cli.assert_cli_defaults_match_dataclass(
        parser, policy_cfg_type, shared_fields
    )
    isaaclab_arena.cli.dataclass_cli.add_dataclass_cli_args(
        parser, policy_cfg_type, excluded_fields=shared_fields
    )
    return parser


def build_policy_from_cli(policy_type: type, args_cli: argparse.Namespace):
    import isaaclab_arena.assets.registries
    import isaaclab_arena.cli.dataclass_cli

    registry = isaaclab_arena.assets.registries.PolicyRegistry()
    policy_cfg_type = registry.get_policy_cfg_type(policy_type)
    cfg = isaaclab_arena.cli.dataclass_cli.dataclass_from_cli(policy_cfg_type, args_cli)
    return policy_type(cfg)


def add_policy_runner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy_type", type=str, default=None,
                        help="Registered policy name or dotted policy class path.")
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--language_instruction", type=str, default=None)
    parser.add_argument("--record_viewport_video", action="store_true", default=False)
    parser.add_argument("--record_camera_video", action="store_true", default=False)
    parser.add_argument("--output_base_dir", type=str, default="outputs")


# ---------------------------------------------------------------------------
# Static HTML report for one run directory.
# ---------------------------------------------------------------------------


def build_report(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    videos = sorted(p.name for p in out.glob("*.mp4"))
    episodes = []
    for p in sorted((out / "milestones").glob("episode_*_milestones.json")):
        episodes.append(json.loads(p.read_text()))
    summary = {}
    summary_path = out / "milestones" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    rows = []
    for e in episodes:
        marks = "".join(
            f"<li>{'&#9989;' if ok else '&#10060;'} {html.escape(name)}</li>"
            for name, ok in e.get("milestones", {}).items()
        )
        progress = ", ".join(f"{k}={v}" for k, v in e.get("progress", {}).items())
        rows.append(
            f"<tr><td>{e['episode']}</td>"
            f"<td>{'PASS' if e.get('success') else 'fail'}</td>"
            f"<td><ul>{marks}</ul></td><td><code>{html.escape(progress)}</code></td></tr>"
        )
    vids = "".join(
        f"<figure><figcaption>{html.escape(v)}</figcaption>"
        f'<video src="{html.escape(v)}" controls width="480"></video></figure>'
        for v in videos
    )
    verdict = html.escape(summary.get("summary", ""))
    instruction = html.escape(summary.get("instruction", ""))
    page = f"""<!doctype html><meta charset="utf-8"><title>sim-env-builder rollout</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:.4rem .6rem;vertical-align:top}}
ul{{margin:0;padding-left:1.2em}}
figure{{display:inline-block;margin:.5rem}}</style>
<h1>sim-env-builder rollout</h1>
<p><b>instruction:</b> {instruction}</p>
<p><b>{verdict}</b></p>
<h2>Milestones (from articulation state)</h2>
<table><tr><th>episode</th><th>result</th><th>milestones</th><th>progress</th></tr>
{''.join(rows)}</table>
<h2>Videos</h2>{vids}
"""
    path = out / "index.html"
    path.write_text(page)
    return path

if __name__ == "__main__":
    main()
