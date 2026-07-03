"""Tests for the hardware simulator."""

import pytest
from adaptive_firmware.hardware.configs import CONFIG_PRESETS, AcceleratorConfig
from adaptive_firmware.hardware.simulator import HardwareSimulator
from adaptive_firmware.hardware.bitstream_cache import BitstreamCache


class TestBitstreamCache:
    def test_cache_hit(self):
        cache = BitstreamCache(capacity=2)
        config = CONFIG_PRESETS[0]
        # First request: miss
        time1 = cache.request(config)
        assert time1 == config.reconfig_time_ms
        assert cache.misses == 1
        # Second request: hit
        time2 = cache.request(config)
        assert time2 == 0.0
        assert cache.hits == 1

    def test_cache_eviction(self):
        cache = BitstreamCache(capacity=2)
        c0, c1, c2 = CONFIG_PRESETS[0], CONFIG_PRESETS[1], CONFIG_PRESETS[2]
        # Fill cache
        cache.request(c0)
        cache.request(c1)
        assert cache.loaded_config_ids == [0, 1]
        # Add third: evicts LRU (c0)
        cache.request(c2)
        assert 0 not in cache.loaded_config_ids
        assert cache.loaded_config_ids == [1, 2]

    def test_reset(self):
        cache = BitstreamCache(capacity=2)
        cache.request(CONFIG_PRESETS[0])
        cache.reset()
        assert cache.loaded_config_ids == []
        assert cache.hits == 0
        assert cache.misses == 0


class TestHardwareSimulator:
    def test_execute_compute_bound(self):
        sim = HardwareSimulator(CONFIG_PRESETS)
        # Compute-bound: high FLOPs, low memory
        result = sim.execute(flops=1e9, memory_bytes=1e7, config_id=0)  # HIGH_COMPUTE
        assert result.exec_time_ms > 0
        assert result.throughput_gops > 0
        assert result.config_id == 0

    def test_roofline_model(self):
        sim = HardwareSimulator(CONFIG_PRESETS)
        # Compute-bound workload should be faster on HIGH_COMPUTE than HIGH_BANDWIDTH
        result_compute = sim.execute(flops=1e9, memory_bytes=1e7, config_id=0)
        sim.reset()
        result_bandwidth = sim.execute(flops=1e9, memory_bytes=1e7, config_id=1)
        assert result_compute.exec_time_ms < result_bandwidth.exec_time_ms

    def test_memory_bound_faster_on_bandwidth_config(self):
        sim = HardwareSimulator(CONFIG_PRESETS)
        # Memory-bound: low FLOPs, high memory
        result_compute = sim.execute(flops=1e8, memory_bytes=1e10, config_id=0)
        sim.reset()
        result_bandwidth = sim.execute(flops=1e8, memory_bytes=1e10, config_id=1)
        assert result_bandwidth.exec_time_ms < result_compute.exec_time_ms

    def test_workload_classification(self):
        assert HardwareSimulator.classify_workload(1e9, 1e7) == "compute_bound"
        assert HardwareSimulator.classify_workload(1e8, 1e10) == "memory_bound"
        assert HardwareSimulator.classify_workload(1e9, 1e8) == "balanced"

    def test_reconfig_time(self):
        sim = HardwareSimulator(CONFIG_PRESETS, cache_capacity=1)
        # First load: reconfig needed
        r1 = sim.execute(flops=1e8, memory_bytes=1e8, config_id=0)
        assert r1.reconfig_time_ms > 0
        # Same config again: cache hit
        r2 = sim.execute(flops=1e8, memory_bytes=1e8, config_id=0)
        assert r2.reconfig_time_ms == 0.0
        assert r2.cache_hit

    def test_reset(self):
        sim = HardwareSimulator(CONFIG_PRESETS)
        sim.execute(flops=1e9, memory_bytes=1e7, config_id=0)
        sim.reset()
        assert sim.total_exec_time_ms == 0.0
        assert sim.current_config_id is None
