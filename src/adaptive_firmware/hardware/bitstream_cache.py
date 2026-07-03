"""Bitstream cache model.

Simulates the on-device bitstream cache. When the agent requests a config
switch, the cache determines whether the bitstream is already loaded (fast)
or needs to be fetched from cold storage (incurs reconfig_time_ms).
"""

from __future__ import annotations

from collections import OrderedDict

from .configs import AcceleratorConfig


class BitstreamCache:
    """LRU cache for partial bitstreams.

    A cache hit means the bitstream is already in the active PRR — no
    reconfiguration needed. A miss means we must load it, incurring
    the config's reconfig_time_ms penalty.
    """

    def __init__(self, capacity: int = 2) -> None:
        """Initialize cache.

        Args:
            capacity: Number of PRRs (partially reconfigurable regions).
                      Each PRR can hold one bitstream at a time.
        """
        self.capacity = capacity
        self._loaded: OrderedDict[int, AcceleratorConfig] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def loaded_config_ids(self) -> list[int]:
        """IDs of currently loaded configs."""
        return list(self._loaded.keys())

    def request(self, config: AcceleratorConfig) -> float:
        """Request a config from the cache.

        Returns the reconfiguration time incurred (0.0 on hit).
        """
        if config.config_id in self._loaded:
            self.hits += 1
            self._loaded.move_to_end(config.config_id)
            return 0.0

        self.misses += 1
        # Evict LRU if at capacity
        while len(self._loaded) >= self.capacity:
            self._loaded.popitem(last=False)

        self._loaded[config.config_id] = config
        return config.reconfig_time_ms

    def reset(self) -> None:
        self._loaded.clear()
        self.hits = 0
        self.misses = 0
