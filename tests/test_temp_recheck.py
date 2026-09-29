"""模温拒收与复检：探头缺报或读数不可能时不顶替、不判定，复检通过再重判。"""

from __future__ import annotations

import pytest

from rubbercure.app import CycleInputs
from rubbercure.errors import (
    MoldError,
    RubberCureError,
    SensorMissingError,
    SensorReadError,
    SensorValueError,
)
from rubbercure.temp.read import read_mold_temp, sensor_problems
from rubbercure.temp.recheck import RecheckBook

STEP = CycleInputs(inlet_c=26.0, steam_target_kpa=640.0, dt=1.0)


def _book_and_mapping(platform, mold_id="mold-r", cavity_count=3):
    platform.register_mold(mold_id, cavity_count=cavity_count)
    mapping = platform.maps.latest_for(mold_id)
    return mapping, {cavity: sensor for cavity, sensor in mapping.sensors}


# ---------- 读取层 ----------

def test_missing_sensor_raises_structured_error(platform):
    mapping, sensors = _book_and_mapping(platform)
    raw = {sensors[0]: 151.0, sensors[1]: 150.0}
    with pytest.raises(SensorMissingError) as caught:
        read_mold_temp("mold-r", mapping, raw, 1.0)
    problems = caught.value.problems
    assert [item["cavity"] for item in problems] == [2]
    assert problems[0]["reason"] == "missing"


def test_impossible_temperature_is_rejected_not_passed_through(platform):
    mapping, sensors = _book_and_mapping(platform)
    for bad, reason in (
        (9999.0, "out_of_range"),
        (-196.0, "out_of_range"),
        (float("nan"), "not_finite"),
        (float("inf"), "not_finite"),
        ("151", "not_a_number"),
        (None, "not_a_number"),
    ):
        raw = {sensors[0]: 151.0, sensors[1]: 150.0, sensors[2]: bad}
        with pytest.raises(SensorValueError) as caught:
            read_mold_temp("mold-r", mapping, raw, 1.0)
        assert caught.value.problems[0]["reason"] == reason


def test_sensor_problems_collects_every_failed_cavity(platform):
    mapping, sensors = _book_and_mapping(platform, cavity_count=4)
    # 腔 0、2 缺报，腔 1 越界，只有腔 3 正常
    problems = sensor_problems(
        mapping,
        {
            sensors[1]: 9999.0,
            sensors[3]: 151.0,
        },
    )
    by_cavity = {item["cavity"]: item["reason"] for item in problems}
    assert by_cavity == {0: "missing", 1: "out_of_range", 2: "missing"}

    with pytest.raises(SensorReadError) as caught:
        read_mold_temp(
            "mold-r",
            mapping,
            {sensors[0]: 151.0, sensors[1]: 151.0},
            1.0,
        )
    assert {item["cavity"] for item in caught.value.problems} == {2, 3}


def test_missing_sensor_error_is_still_a_mold_error(platform):
    mapping, sensors = _book_and_mapping(platform, cavity_count=2)
    with pytest.raises(MoldError):
        read_mold_temp("mold-r", mapping, {sensors[0]: 151.0}, 1.0)


def test_in_range_reading_still_maps(platform):
    mapping, sensors = _book_and_mapping(platform, cavity_count=2)
    reading = read_mold_temp(
        "mold-r", mapping, {sensors[0]: 150.0, sensors[1]: 152.0}, 2.0
    )
    assert reading.average_c() == 151.0
    assert sensor_problems(mapping, {sensors[0]: 150.0, sensors[1]: 152.0}) == []


# ---------- 复检台账 ----------

def test_recheck_book_keeps_pending_until_explicit_pass():
    book = RecheckBook()
    record = book.reject("m1", "press-01", 1, [{"cavity": 0, "sensor": "s0", "reason": "missing"}], 10.0)
    assert record.pending
    assert book.is_pending("m1")

    failed = book.record_attempt("m1", False, 11.0, [{"cavity": 0, "sensor": "s0", "reason": "out_of_range"}])
    assert failed.attempts == 1
    assert book.is_pending("m1")

    passed = book.record_attempt("m1", True, 12.0)
    assert passed.attempts == 2
    assert not book.is_pending("m1")
    assert book.snapshot()["recent_resolved"][0]["mold_id"] == "m1"


def test_recheck_book_attempt_without_pending_is_an_error():
    book = RecheckBook()
    with pytest.raises(KeyError):
        book.record_attempt("ghost", True, 1.0)


# ---------- 控制周期：拒收不顶替 ----------

def test_cycle_rejects_bad_sampling_without_substitution(platform):
    platform.run_cure("mold-c", cavity_count=3)
    mapping = platform.maps.latest_for("mold-c")
    sensors = [sensor for _, sensor in sorted(mapping.sensors)]

    frame = platform.report_sensor_readings(
        "mold-c", {sensors[0]: 151.0, sensors[1]: 9999.0}
    )
    assert not frame["acceptable"]

    result = platform.cycle(STEP)
    press = result["presses"][0]
    assert press["temp_judgement"] == "rejected"
    assert press["temperature"] is None
    assert press["cure"]["elapsed_s"] == 0.0
    assert [item["mold_id"] for item in result["temp_rejections"]] == ["mold-c"]
    assert result["temp_recheck_pending"][0]["cavities"] == [1, 2]

    # 坏帧未撤之前，下一周期继续拒收（绝不顶替、绝不自动恢复判定）
    result2 = platform.cycle(STEP)
    for each in (result["presses"][0], result2["presses"][0]):
        assert each["temp_judgement"] == "rejected"
        assert each["temperature"] is None
        assert each["cure"]["elapsed_s"] == 0.0
    assert platform.rechecks.pending_for("mold-c").attempts == 0


def test_pending_recheck_does_not_auto_recover_on_clean_frame(platform):
    platform.run_cure("mold-c2", cavity_count=2)
    sensors = [s for _, s in sorted(platform.maps.latest_for("mold-c2").sensors)]
    platform.report_sensor_readings("mold-c2", {sensors[0]: 151.0})
    platform.cycle(STEP)
    assert platform.rechecks.is_pending("mold-c2")

    platform.clear_sensor_readings("mold-c2")
    result = platform.cycle(STEP)
    assert result["presses"][0]["temp_judgement"] == "pending_recheck"
    assert result["presses"][0]["temperature"] is None


def test_recheck_failure_keeps_judgement_rejected(platform):
    platform.run_cure("mold-c3", cavity_count=2)
    sensors = [s for _, s in sorted(platform.maps.latest_for("mold-c3").sensors)]
    platform.report_sensor_readings("mold-c3", {sensors[0]: 151.0, sensors[1]: 9999.0})
    platform.cycle(STEP)

    outcome = platform.recheck_mold_temp(
        "mold-c3", readings={sensors[0]: 151.0, sensors[1]: 9999.0}
    )
    assert outcome["ok"] is False
    assert outcome["recheck"]["attempts"] == 1
    assert platform.rechecks.is_pending("mold-c3")

    missing = platform.recheck_mold_temp("mold-c3", readings={sensors[0]: 151.0})
    assert missing["ok"] is False
    assert [p["reason"] for p in missing["recheck"]["problems"]] == ["missing"]


def test_recheck_pass_rejudges_and_resumes_cycle(platform):
    platform.run_cure("mold-c4", cavity_count=2)
    sensors = [s for _, s in sorted(platform.maps.latest_for("mold-c4").sensors)]
    platform.report_sensor_readings("mold-c4", {sensors[0]: 151.0, sensors[1]: 9999.0})
    platform.cycle(STEP)

    outcome = platform.recheck_mold_temp(
        "mold-c4", readings={sensors[0]: 151.0, sensors[1]: 151.4}
    )
    assert outcome["ok"] is True
    assert outcome["in_band"] is True
    assert outcome["spread_ok"] is True
    assert outcome["judgement"] == "qualified"
    assert not platform.rechecks.is_pending("mold-c4")

    result = platform.cycle(STEP)
    assert result["presses"][0]["temp_judgement"] == "ok"
    assert result["presses"][0]["temperature"] is not None


def test_recheck_pass_can_still_judge_out_of_band(platform):
    platform.run_cure("mold-c5", cavity_count=2)
    sensors = [s for _, s in sorted(platform.maps.latest_for("mold-c5").sensors)]
    platform.report_sensor_readings("mold-c5", {sensors[0]: 151.0, sensors[1]: 9999.0})
    platform.cycle(STEP)

    # 值真实可信（通过可信窗口），但仍明显低于工艺目标带
    outcome = platform.recheck_mold_temp(
        "mold-c5", readings={sensors[0]: 60.0, sensors[1]: 60.5}
    )
    assert outcome["ok"] is True
    assert outcome["in_band"] is False
    assert outcome["judgement"] == "rechecked_out_of_band"


def test_recheck_without_pending_record_is_rejected(platform):
    platform.register_mold("mold-c6", cavity_count=2)
    with pytest.raises(Exception) as caught:
        platform.recheck_mold_temp("mold-c6", readings={})
    assert isinstance(caught.value, RubberCureError)
    assert "没有待复检" in caught.value.message


# ---------- 硫化顺序 ----------

def test_cure_sequence_rejects_impossible_reading_without_confirmation(platform):
    report = platform.run_cure(
        "mold-s",
        cavity_count=2,
        raw_readings={"mold-s-T1": 151.0, "mold-s-T2": -100.0},
    )
    assert report["ok"] is False
    assert report["temp_judgement"] == "rejected"
    assert report["recheck_required"] is True
    assert report["steps"][-1] == "steam_open"
    unit = platform.press("press-01")
    assert unit.timer.phase.value == "idle"
    assert unit.temp_loop.is_confirmed() is False
    assert platform.rechecks.is_pending("mold-s")

    alarms = platform.audit.alarms()
    assert any(item["action"] == "mold_temp_rejected" for item in alarms)


def test_cure_rejected_then_recheck_passes(platform):
    platform.run_cure(
        "mold-s2",
        cavity_count=2,
        raw_readings={"mold-s2-T1": 151.0, "mold-s2-T2": 9999.0},
    )
    outcome = platform.recheck_mold_temp(
        "mold-s2",
        readings={"mold-s2-T1": 151.0, "mold-s2-T2": 151.2},
    )
    assert outcome["ok"] is True
    assert outcome["judgement"] == "qualified"


# ---------- 控制台 ----------

def test_console_recheck_endpoint_roundtrip(platform):
    from rubbercure.console.api import ConsoleAPI

    api = ConsoleAPI(platform)
    platform.run_cure("mold-http", cavity_count=2)
    injected = api.sensor_readings(
        {"mold_id": "mold-http", "readings": {"mold-http-T1": 151.0}}
    )
    assert injected["ok"] is True
    assert injected["frame"]["acceptable"] is False

    api.step({"dt": 1.0})
    failed = api.temp_recheck(
        {"mold_id": "mold-http", "readings": {"mold-http-T1": 151.0, "mold-http-T2": 9999.0}}
    )
    assert failed["ok"] is False

    passed = api.temp_recheck(
        {"mold_id": "mold-http", "readings": {"mold-http-T1": 151.0, "mold-http-T2": 151.0}}
    )
    assert passed["ok"] is True
    assert passed["judgement"] == "qualified"

    status = api.status()
    assert status["temp_rechecks"]["pending"] == []
