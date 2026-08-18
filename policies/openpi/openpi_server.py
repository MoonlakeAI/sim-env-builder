"""Serve an openpi (PI-0 / PI-0.5) checkpoint over the openpi websocket protocol.

Equivalent to openpi's scripts/serve_policy.py, defaulting to the DROID
joint-position PI-0.5 checkpoint that IsaacLab-Arena's Pi0RemotePolicy
(droid adapter) expects. Fully generic: point --config/--checkpoint at any
openpi TrainConfig and checkpoint directory.

    uv run openpi-server                       # pi05_droid_jointpos on :8000
    uv run openpi-server --port 8001 --config pi05_droid_jointpos_polaris
"""

from __future__ import annotations

import argparse
import logging

DEFAULT_CONFIG = "pi05_droid_jointpos_polaris"
DEFAULT_CHECKPOINT = "gs://openpi-assets-simeval/pi05_droid_jointpos"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="openpi TrainConfig name")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="checkpoint dir (local path or gs:// URI)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.shared import download
    from openpi.training import config as openpi_train_config

    config = openpi_train_config.get_config(args.config)
    checkpoint_dir = download.maybe_download(args.checkpoint)
    logging.info("Loading policy %s from %s", args.config, checkpoint_dir)

    policy = policy_config.create_trained_policy(
        config, checkpoint_dir, default_prompt=args.default_prompt
    )
    logging.info("Policy metadata: %s", policy.metadata)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy.metadata
    )
    logging.info("PI-0.5 policy server listening on ws://%s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
