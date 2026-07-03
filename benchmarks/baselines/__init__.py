"""Baseline implementations for comparison.

These are reference implementations that any adaptive agent should
be compared against:

- `static`: Pick one config and stick with it. Tests if the workload
  is homogeneous enough that a fixed config is good enough.
- `random`: Pick a random config each step. Tests if the agent is
  actually learning, or if just being "different" each time helps.
- `oracle`: Always pick the best config for the workload class.
  Tests the upper bound on what any agent can achieve.

The oracle is intentionally not realistic — it knows the optimal
workload class → config mapping. The real question is how close
adaptive agents get to the oracle.
"""

from .static import StaticConfigBaseline
from .random import RandomConfigBaseline
from .oracle import OracleBaseline

__all__ = [
    "StaticConfigBaseline",
    "RandomConfigBaseline",
    "OracleBaseline",
]
