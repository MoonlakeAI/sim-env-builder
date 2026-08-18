"""sim-env-builder CLI. Pre-Kit safe: no Isaac imports at module level.

    sim-env-builder suggest [usd]         # articulation -> milestones & prompts
    sim-env-builder check-asset [asset]   # static quality checks -> report.json
    sim-env-builder rollout [options]     # one command -> videos + milestone JSON + pass/fail
    sim-env-builder dashboard [options]   # review rollout videos + milestone timelines
    sim-env-builder preview-gifs          # joint-driven GIFs of each default task

Rollout artifacts land in the output dir:
  - rollout MP4s (per-episode robot-camera videos + one viewport video)
  - milestones/episode_XXX_milestones.json (from joint state; includes
    progress metrics like max_door_angle_deg even when the policy scores zero)
  - a final [PASS]/[FAIL] line, plus Arena's episode_results.jsonl and
    an index.html report
"""

from __future__ import annotations

import argparse
import os
import sys

POLICY_CLASS = "isaaclab_arena_openpi.policy.pi0_remote_policy.Pi0RemotePolicy"
DEFAULT_KIT_ARGS = "--/renderer/multiGpu/enabled=false"


def _cmd_suggest(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .articulation import derive_milestones, parse_articulation

    usd = args.usd
    asset = args.asset_name or Path(usd).stem.replace("_", " ")
    joints = parse_articulation(usd)
    milestones = derive_milestones(joints)

    print(f"Articulation of {usd}:")
    for j in joints:
        print(f"  {j.name:24s} {j.joint_type:10s} range [{j.lower:g}, {j.upper:g}] {j.units}")
    print("\nDerived milestones (threshold = displacement from rest that counts as achieved):")
    for m in milestones:
        print(f"  {m.instruction(asset):45s} {m.joint.name} >= {m.threshold:g} {m.joint.units}")
    print("\nSuggested prompts to test:")
    for m in milestones:
        print(f'  "{m.instruction(asset)}"')


def _cmd_check_asset(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .asset_checks.cli import main as check_asset_main
    from .paths import OUTPUTS_DIR

    out = Path(args.out) if args.out else OUTPUTS_DIR / "asset_checks" / Path(args.asset).stem
    argv = [args.asset, "--out", str(out), "--render", args.render]
    if args.thresholds:
        argv += ["--thresholds", args.thresholds]
    if args.blender:
        argv += ["--blender", args.blender]
    if args.verbose:
        argv.append("--verbose")
    raise SystemExit(check_asset_main(argv))


def _cmd_rollout(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .paths import OUTPUTS_DIR
    from .tasks import TASK_SPECS

    if args.task == "all":
        task_names = list(TASK_SPECS)
    else:
        task_names = [t.replace("-", "_") for t in args.task.split(",")]
    unknown = [t for t in task_names if t not in TASK_SPECS]
    if unknown:
        known = sorted(TASK_SPECS)
        raise SystemExit(f"unknown tasks {unknown}; known tasks: {', '.join(known)}, all")
    if args.instruction and len(task_names) > 1:
        raise SystemExit("--instruction only applies to a single --task")

    runs: list[tuple[str, list[str]]] = []
    for task_name in task_names:
        env_name = task_name
        instruction = args.instruction or TASK_SPECS[task_name].instruction
        if args.output:
            base = Path(args.output)
            output_dir = str(base / task_name) if len(task_names) > 1 else str(base)
        else:
            output_dir = str(OUTPUTS_DIR / task_name)

        argv = [
            "sim-env-builder-rollout",
            "--device", f"cuda:{args.gpu}",
            "--enable_cameras",
            "--num_envs", str(args.num_envs),
            "--seed", str(args.seed),
            "--policy_type", POLICY_CLASS,
            "--policy_variant", "pi05",
            "--remote_host", args.host,
            "--remote_port", str(args.port),
            "--record_camera_video",
            "--record_viewport_video",
            "--output_base_dir", output_dir,
            "--language_instruction", instruction,
            f"--kit_args={DEFAULT_KIT_ARGS}",
        ]
        if not args.viz:
            argv.append("--headless")
        if args.steps is not None:
            argv += ["--num_steps", str(args.steps)]
        else:
            argv += ["--num_episodes", str(args.episodes)]
        # Environment subcommand must be the trailing positional.
        argv.append(env_name)
        argv += ["--embodiment", args.embodiment]
        argv += args.extra
        runs.append((task_name, argv))

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    if len(runs) == 1:
        from .rollout import main as rollout_main

        sys.argv = runs[0][1]
        rollout_main()
    else:
        # Kit boot dominates short runs; batch all tasks into one session.
        from .rollout import main_batch

        main_batch(runs)


def _cmd_dashboard(args: argparse.Namespace) -> None:
    from .dashboard import serve
    from .paths import OUTPUTS_DIR

    serve(args.dir or OUTPUTS_DIR, args.port, args.host)


def _cmd_preview_gifs(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .preview import generate_task_gifs

    tasks = [t.replace("-", "_") for t in args.task] if args.task else None
    generate_task_gifs(
        tasks=tasks,
        out_dir=Path(args.output) if args.output else None,
        blender=args.blender,
        width=args.width,
        height=args.height,
        n_frames=args.frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="sim-env-builder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_suggest = sub.add_parser("suggest", help="derive milestones & prompts from an asset USD")
    p_suggest.add_argument("usd", help="path to a .usd/.usda/.usdz asset package")
    p_suggest.add_argument("--asset-name", default=None,
                           help="name used in prompts (default: from the filename)")
    p_suggest.set_defaults(func=_cmd_suggest)

    p_check = sub.add_parser(
        "check-asset",
        help="run static quality checks on an asset package (USD or glTF)",
    )
    p_check.add_argument("asset", help="path to a .usd/.usda/.usdc/.usdz/.glb/.gltf package")
    p_check.add_argument("--out", default=None,
                         help="report directory (default outputs/asset_checks/<asset>)")
    p_check.add_argument("--render", choices=["off", "preview", "loop"], default="off",
                         help="render a turntable via Blender; 'loop' takes minutes")
    p_check.add_argument("--thresholds", default=None,
                         help="JSON file overriding threshold defaults")
    p_check.add_argument("--blender", default=None, help="Blender binary (default: PATH)")
    p_check.add_argument("--verbose", action="store_true")
    p_check.set_defaults(func=_cmd_check_asset)

    p_roll = sub.add_parser("rollout", help="roll out PI-0.5 and emit videos + milestones")
    p_roll.add_argument("--task", required=True,
                        help="task name (see tasks.py), comma-separated list, or 'all'; "
                             "multiple tasks share one Isaac Sim boot")
    p_roll.add_argument("--episodes", type=int, default=5, help="episodes to run (default 5)")
    p_roll.add_argument("--steps", type=int, default=None,
                        help="run a fixed number of env steps instead of episodes")
    p_roll.add_argument("--instruction", default=None,
                        help="language instruction (default: task-specific DROID phrasing)")
    p_roll.add_argument("--host", default="127.0.0.1", help="PI-0.5 policy server host")
    p_roll.add_argument("--port", type=int, default=8000, help="PI-0.5 policy server port")
    p_roll.add_argument("--gpu", type=int, default=0, help="CUDA device index for the sim")
    p_roll.add_argument("--num-envs", type=int, default=1)
    p_roll.add_argument("--seed", type=int, default=42)
    p_roll.add_argument("--embodiment", default="droid_abs_joint_pos")
    p_roll.add_argument("--output", default=None, help="output base dir (default outputs/<task>)")
    p_roll.add_argument("--viz", action="store_true",
                        help="open the Kit viewport (default headless)")
    p_roll.add_argument("extra", nargs="*", help="extra args passed through to the arena runner")
    p_roll.set_defaults(func=_cmd_rollout)

    p_dash = sub.add_parser("dashboard", help="serve a review page for rollout outputs")
    p_dash.add_argument("--dir", default=None,
                        help="output directory to browse (default outputs/)")
    p_dash.add_argument("--port", type=int, default=8090, help="HTTP port (default 8090)")
    p_dash.add_argument("--host", default="0.0.0.0",
                        help="bind address (default 0.0.0.0, reachable from other machines)")
    p_dash.set_defaults(func=_cmd_dashboard)

    p_prev = sub.add_parser(
        "preview-gifs",
        help="render looping GIFs of each default task's intended joint motion",
    )
    p_prev.add_argument(
        "--task",
        action="append",
        default=None,
        help="task name (repeatable). Default: all bundled tasks",
    )
    p_prev.add_argument("--output", default=None, help="GIF output directory")
    p_prev.add_argument("--blender", default=None, help="blender executable (default: PATH)")
    p_prev.add_argument("--width", type=int, default=320)
    p_prev.add_argument("--height", type=int, default=180)
    p_prev.add_argument("--frames", type=int, default=28)
    p_prev.set_defaults(func=_cmd_preview_gifs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
