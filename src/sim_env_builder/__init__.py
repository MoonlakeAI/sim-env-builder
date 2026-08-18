"""sim-env-builder: preview policy rollouts on articulated, sim-ready assets.

Compose an articulated asset into an Isaac Lab Arena scene, roll out a PI-0.5
policy against it, and grade progress with milestones derived from the
asset's own articulation metadata, with no VLM judge or labeling.
"""

from .articulation import JointSpec, Milestone, derive_milestones, parse_articulation
from .scoring import EpisodeRecord, summarize, write_episode_json

__all__ = [
    "JointSpec",
    "Milestone",
    "derive_milestones",
    "parse_articulation",
    "EpisodeRecord",
    "summarize",
    "write_episode_json",
]

__version__ = "0.1.0"
