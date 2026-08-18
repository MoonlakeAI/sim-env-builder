"""Command line entry point."""

import argparse
import logging
import pathlib
import sys

from sim_env_builder.asset_checks import config, ingest, report
from sim_env_builder.asset_checks.checks import Context, run_all
from sim_env_builder.asset_checks.render import turntable

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="quality-evals",
        description="Evaluate the quality of a packaged 3D asset.",
    )
    parser.add_argument("asset", help="USD or glTF file to evaluate")
    parser.add_argument(
        "--out", required=True, type=pathlib.Path, help="directory for the report"
    )
    parser.add_argument(
        "--render",
        choices=["off", "preview", "loop"],
        default="off",
        help="render a turntable; 'loop' takes minutes",
    )
    parser.add_argument(
        "--thresholds", type=pathlib.Path, help="JSON file overriding threshold defaults"
    )
    parser.add_argument("--blender", help="path to the Blender binary")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    thresholds = config.load(args.thresholds)
    logger.info("loading %s", args.asset)
    asset = ingest.load(args.asset)
    logger.info(
        "loaded %d part(s), %d joint(s), %d triangle(s)",
        len(asset.parts),
        len(asset.joints),
        asset.triangle_count,
    )

    results = run_all(Context(asset, thresholds))
    rendered = turntable.render(
        args.asset, args.out / "render", args.render, args.blender
    )

    args.out.mkdir(parents=True, exist_ok=True)
    document = report.build(asset, results, rendered)
    report.write(args.out / "report.json", document)

    print(report.summarize(document))
    logger.info("report written to %s", args.out / "report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
