"""探头缺报或读数不可信：模温判定拒收、提示复检、复检通过后重判。"""

from __future__ import annotations

import pytest

from rubbercure.app import CycleInputs
from rubbercure.errors import MoldError
from rubbercure.sequence.cure import CureRunPlan, CureSequence
from rubbercure.temp.read import read_mold_temp


def _cycle_inputs(platform, raw=None):
    return CycleInputs(
        inlet_c=26.0,
        steam_target_kpa=platform.config.steam.target_pressure_kpa,
        dt=1.0,
        raw_readings=raw,
    )


def _press_entry(cycle_report, press_id):
    return [item for item in cycle_report["presses"] if item["press_id"] == press_id][0]


def test_read_mold_temp_rejects_implausible_values(platform):
    platform.register_mold("mold-p", cavity_count=2)
    mapping = platform.maps.latest_for("mold-p")
    plausible = platform.config.temp.plausible_range()

    reading = read_mold_temp(
        "mold-p", mapping, {"mold-p-T1": 150.0, "mold-p-T2": 151.0}, 0.0,
        plausible_range=plausible,
    )
    assert reading.by_cavity == {0: 150.0, 1: 151.0}

    for bad in (9999.0, -273.0, float("nan"), float("inf")):
        with pytest.raises(MoldError):
            read_mold_temp(
                "mold-p", mapping, {"mold-p-T1": bad, "mold-p-T2": 151.0}, 0.0,
                plausible_range=plausible,
            )


def test_cycle_rejects_judgment_when_probe_silent(platform):
    platform.run_cure("mold-A", cavity_count=4)
    unit = platform.press("press-01")
    cycles_before = unit.snapshot()["cycles"]

    result = platform.cycle(_cycle_inputs(platform, raw={"mold-A": {"mold-A-T1": 151.0}}))

    rejected = _press_entry(result, "press-01")
    assert rejected["temp_judgment"] == "rejected"
    assert rejected["recheck_required"] is True
    assert rejected["mold_id"] == "mold-A"
    # 没有拿别的腔凑数再判：拒收的机台不产出任何判定值
    assert "temperature" not in rejected
    assert unit.snapshot()["cycles"] == cycles_before
    # 其余机台不受影响，照常判定
    for item in result["presses"]:
        if item["press_id"] != "press-01":
            assert "temperature" in item
    # 提示复检：告警级别事件已入审计
    alarms = platform.audit.alarms()
    assert any(
        item["action"] == "temp_judgment_rejected" and item["subject"] == "mold-A"
        for item in alarms
    )


def test_cycle_rejects_judgment_when_temperature_impossible(platform):
    platform.run_cure("mold-B", cavity_count=4)
    raw = {f"mold-B-T{index}": 151.0 for index in range(1, 5)}
    raw["mold-B-T3"] = 9999.0

    result = platform.cycle(_cycle_inputs(platform, raw={"mold-B": raw}))

    rejected = _press_entry(result, "press-01")
    assert rejected["temp_judgment"] == "rejected"
    assert "9999" in rejected["reason"]


def test_judgment_resumes_after_probe_recheck_passes(platform):
    platform.run_cure("mold-C", cavity_count=4)
    bad = {f"mold-C-T{index}": 151.0 for index in range(1, 5)}
    del bad["mold-C-T2"]
    rejected = platform.cycle(_cycle_inputs(platform, raw={"mold-C": bad}))
    assert _press_entry(rejected, "press-01")["temp_judgment"] == "rejected"

    # 探头复检通过、重新完整上报后，下个周期自动重判
    good = {f"mold-C-T{index}": 151.0 for index in range(1, 5)}
    recovered = platform.cycle(_cycle_inputs(platform, raw={"mold-C": good}))
    entry = _press_entry(recovered, "press-01")
    assert "temperature" in entry
    assert entry["temperature"]["mold_id"] == "mold-C"
    assert entry["temperature"]["spread_ok"] is True


def test_cure_sequence_rejects_implausible_reading(platform):
    platform.register_mold("mold-D", cavity_count=2)
    mapping = platform.maps.latest_for("mold-D")
    sequence = CureSequence(
        clock=platform.clock,
        bus=platform.bus,
        config=platform.config,
        registry=platform.registry,
        maps=platform.maps,
        curves=platform.curves,
        press=platform.press("press-02"),
    )
    plan = CureRunPlan(
        mold_id="mold-D",
        product_code="RB-215",
        cavity_count=2,
        thickness_mm=6.0,
        sensors={cavity: sensor for cavity, sensor in mapping.sensors},
        raw_readings={"mold-D-T1": 151.0, "mold-D-T2": 8888.0},
        target_kpa=platform.config.steam.target_pressure_kpa,
        warmup_start_c=30.0,
        temp_hold_s=platform.config.temp.confirm_hold_s,
    )
    with pytest.raises(MoldError):
        sequence.run(plan)
    # 判定被拒收：模温确认没有落盘，硫化计时没有启动
    assert not platform.press("press-02").temp_loop.is_confirmed()
    assert platform.press("press-02").timer.phase.value != "curing"
