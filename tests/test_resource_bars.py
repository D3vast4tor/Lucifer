"""Current-only resource gauges, cached telemetry and honest missing readings."""
import copy

import pytest

from lucifer.tui import LuciferApp


class Panel:
    def __init__(self):
        self.value = None
        self.updates = 0

    def update(self, value):
        self.value = value
        self.updates += 1


class ResourceView:
    """Exercise production rendering/cache methods without mounting/network I/O."""
    metric = staticmethod(LuciferApp.metric)
    meter = staticmethod(LuciferApp.meter)
    render_gpu = LuciferApp.render_gpu
    update_panel = LuciferApp.update_panel
    update_session = LuciferApp.update_session

    def __init__(self, gpu):
        self._gpu = gpu
        self._gpu_summary = ""
        self._panel_values = {}
        self.conversation = None
        self.panels = {"#gpus": Panel(), "#session": Panel()}

    def main_query(self, selector, widget_type):
        return self.panels[selector]

    @property
    def text(self):
        return self.panels["#gpus"].value.plain


def snapshot(**changes):
    card = {"index": 0, "name": "Lucifer GPU", "backend": "cuda", "utilization_pct": 12,
            "used_mb": 3000, "total_mb": 12000, "power_w": 20, "temperature_c": 42}
    card.update(changes)
    return {"gpus": [card], "sampled_at": "2026-11-12T12:00:00Z", "stale": False, "errors": []}


def test_utilization_and_vram_show_current_percentages_and_themed_bars_only():
    view = ResourceView(snapshot())
    view.render_gpu()
    assert "GPU  12% █░░░░░░░░░" in view.text
    assert "VRAM 25% ██░░░░░░░░" in view.text
    assert "3,000 / 12,000 MiB" in view.text and "42°C // 20 W" in view.text
    assert "▁" not in view.text and "▂" not in view.text and "▃" not in view.text
    assert not hasattr(view, "_gpu_history")
    assert "GPU 12%" in view._gpu_summary and "VRAM 25%" in view._gpu_summary
    assert "█" in view._gpu_summary and "░" in view._gpu_summary
    view._gpu["gpus"][0].update(utilization_pct=87, used_mb=10000)
    view.render_gpu()
    assert "GPU  87%" in view.text and "VRAM 83%" in view.text
    assert "12%" not in view.text and "25%" not in view.text


def test_known_zero_and_missing_counters_remain_distinguishable():
    view = ResourceView(snapshot(utilization_pct=0, used_mb=0))
    view.render_gpu()
    assert "GPU  0% ░░░░░░░░░░" in view.text and "VRAM 0% ░░░░░░░░░░" in view.text
    view._gpu["gpus"][0].update(utilization_pct=None, used_mb=None)
    view.render_gpu()
    assert "GPU  --% ··········" in view.text and "VRAM --% ··········" in view.text
    assert "--% = unavailable" in view.text and "0%" not in view.text


def test_stale_reading_and_connection_failure_keep_last_values_explicitly_stale():
    view = ResourceView(snapshot(utilization_pct=67, used_mb=6000))
    view._gpu.update(stale=True, errors=["Counter sample expired"])
    view.render_gpu()
    assert "GPU  67%" in view.text and "VRAM 50%" in view.text
    assert "STALE" in view.text and "LIVE" not in view.text
    assert "Counter sample expired" in view.text and "STALE" in view._gpu_summary
    view.render_gpu(connection_error="Backend disconnected")
    assert "67%" in view.text and "Backend disconnected" in view.text and "STALE" in view._gpu_summary


@pytest.mark.parametrize("changes", [
    {"utilization_pct": float("nan"), "used_mb": -2},
    {"utilization_pct": float("inf"), "total_mb": 0},
    {"utilization_pct": 101, "used_mb": 12001},
    {"utilization_pct": -1, "total_mb": -100},
    {"utilization_pct": True, "used_mb": True},
])
def test_invalid_percent_or_memory_readings_are_unknown(changes):
    view = ResourceView(snapshot(**changes))
    view.render_gpu()
    assert "GPU  --%" in view.text and "VRAM --%" in view.text
    assert "nan%" not in view.text and "inf%" not in view.text and "101%" not in view.text


def test_memory_percentage_can_use_current_total_and_free_but_never_invents_capacity():
    view = ResourceView(snapshot(used_mb=None, free_mb=3000))
    view.render_gpu()
    assert "VRAM 75%" in view.text and "9,000 / 12,000 MiB" in view.text
    view._gpu["gpus"][0]["total_mb"] = None
    view.render_gpu()
    assert "VRAM --%" in view.text and "MiB" not in view.text


def test_render_reuses_cached_telemetry_and_updates_widgets_only_when_changed(monkeypatch):
    import lucifer.gpu
    import lucifer.telemetry
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("Resource rendering must never discover or sample hardware")
    monkeypatch.setattr(lucifer.gpu, "discover_gpus", unexpected_probe)
    monkeypatch.setattr(lucifer.telemetry, "sample_gpus", unexpected_probe)
    view = ResourceView(snapshot())
    original = copy.deepcopy(view._gpu)
    view.render_gpu()
    view.render_gpu()
    assert view._gpu == original
    assert view.panels["#gpus"].updates == view.panels["#session"].updates == 1
    view._gpu["gpus"][0].update(utilization_pct=99, used_mb=9000)
    view.render_gpu()
    assert view.panels["#gpus"].updates == view.panels["#session"].updates == 2
    assert "99%" in view.text and "75%" in view.text
    view._gpu["sampled_at"] = "2026-11-12T12:00:01Z"
    view.render_gpu()
    assert view.panels["#gpus"].updates == 3 and view.panels["#session"].updates == 2


def test_every_device_has_its_own_current_percentages_and_missing_time_is_unknown():
    gpu = snapshot()
    gpu["gpus"].append({"index": 1, "name": "Second device", "utilization_pct": 80, "memory_used_mb": 2,
                        "memory_total_mb": 4})
    gpu["sampled_at"] = None
    view = ResourceView(gpu)
    view.render_gpu()
    assert "[0] Lucifer GPU" in view.text and "[1] Second device" in view.text
    assert "GPU  80%" in view.text and "VRAM 50%" in view.text
    assert "UNKNOWN // sample time unavailable" in view.text


def test_meter_keeps_unknown_distinct_from_empty_or_full():
    assert LuciferApp.meter(None, 5) == "·····"
    assert LuciferApp.meter(0, 5) == "░░░░░"
    assert LuciferApp.meter(100, 5) == "█████"
    assert LuciferApp.meter(float("nan"), 5) == "·····"
