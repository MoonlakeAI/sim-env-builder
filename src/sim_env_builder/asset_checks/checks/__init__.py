"""Check modules.

Importing a module registers its checks; `registry.SECTIONS` fixes the order
they run and are reported in.
"""

from sim_env_builder.asset_checks.checks import (  # noqa: F401
    articulation,
    materials,
    mesh,
    proxy,
    sim_ready,
    uv,
)
from sim_env_builder.asset_checks.checks.registry import CheckResult, Context, run_all  # noqa: F401
