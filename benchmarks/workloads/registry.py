"""Workload registry.

A central place to register, look up, and list all available workloads.
Adding a new workload is as simple as calling register_workload() at
module import time.
"""

from __future__ import annotations

from typing import Iterator

from .base import WorkloadSpec

_REGISTRY: dict[str, WorkloadSpec] = {}


def register_workload(spec: WorkloadSpec) -> None:
    """Register a workload spec in the global registry."""
    key = f"{spec.name}@{spec.version}"
    if key in _REGISTRY:
        raise ValueError(f"Workload {key} already registered")
    _REGISTRY[key] = spec


def get_workload(name: str, version: str | None = None) -> WorkloadSpec:
    """Look up a workload by name and optional version.

    If version is None, returns the latest registered version.
    """
    if version is not None:
        key = f"{name}@{version}"
        if key not in _REGISTRY:
            raise KeyError(f"No workload registered as {key}")
        return _REGISTRY[key]

    matches = [s for k, s in _REGISTRY.items() if k.startswith(f"{name}@")]
    if not matches:
        raise KeyError(f"No workload registered with name {name!r}")
    # Return the latest version (highest semver)
    return max(matches, key=lambda s: tuple(int(x) for x in s.version.split(".")))


def list_workloads() -> list[WorkloadSpec]:
    """Return all registered workloads, sorted by name then version."""
    return sorted(_REGISTRY.values(), key=lambda s: (s.name, s.version))


def iter_workloads() -> Iterator[WorkloadSpec]:
    """Iterate over all registered workloads."""
    return iter(list_workloads())


def clear_registry() -> None:
    """Clear the registry (for testing)."""
    _REGISTRY.clear()
