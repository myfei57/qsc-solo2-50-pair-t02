"""平台装配与机群控制循环。

``Platform`` 把模具台账、传感器映射、曲线基线、机群、调度器与四条工艺顺序
按真实调用链装起来，并暴露三件事：工艺顺序入口、机群控制循环与状态快照。
控制循环每个周期对每台硫化机做四件事：模温闭环结算压力目标、蒸汽母管充压、
硫化计时推进、冷却水采样。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from .audit.log import AuditLog
from .audit.record import AuditRecord
from .batch.batch import BatchScheduler
from .batch.profile import BatchProfile
from .clock import Clock, SystemClock
from .config import PlatformConfig, default_config
from .errors import MoldError, RubberCureError, SensorReadError
from .eventbus import Event, EventBus
from .mold.mapping import SensorMapBook
from .mold.registry import MoldRegistry
from .mold.summary import summarize_molds
from .ns.registry import NamespaceRegistry
from .ns.topology import Topology, build_default_topology
from .press.fleet import PressFleet
from .press.summary import summarize_fleet
from .sequence.cure import CureRunPlan, CureSequence
from .sequence.dispatch import DispatchSequence
from .sequence.openmold import OpenSequence
from .sequence.selfcheck import verify_topology
from .sequence.stop import StopSequence
from .store.journal import JsonlJournal
from .store.jsonfile import JsonStore
from .telemetry import TrendBuffer
from .temp.read import (
    MoldTempReading,
    read_mold_temp,
    sensor_problems,
)
from .temp.recheck import RecheckBook
from .temp.tolerance import band_summary, spread_ok
from .timer.curve import CurveBook
from .timer.latch import CureLatch

DEFAULT_UNIT = "shop1"
DEFAULT_PRESS_ID = "press-01"
REFERENCE_TEMPERATURE_C = 30.0


@dataclass(frozen=True)
class CycleInputs:
    """一个机群控制周期的工况输入。"""

    inlet_c: float
    steam_target_kpa: float
    dt: float

    def clamped(self) -> "CycleInputs":
        return CycleInputs(
            inlet_c=max(-20.0, float(self.inlet_c)),
            steam_target_kpa=max(0.0, float(self.steam_target_kpa)),
            dt=max(0.001, float(self.dt)),
        )


class Platform:
    """橡胶硫化机群与模具调度控制平台。"""

    def __init__(
        self,
        config: PlatformConfig | None = None,
        clock: Clock | None = None,
        unit: str = DEFAULT_UNIT,
    ) -> None:
        self.config = config or default_config()
        self.clock = clock or SystemClock()
        self.bus = EventBus()

        self.store = JsonStore(self.config.storage_root)
        self.audit = AuditLog(
            self.clock, self.bus, JsonlJournal(self.store.root, "audit", segment_limit=512)
        )
        self.namespaces = NamespaceRegistry(unit)
        self.topology: Topology = build_default_topology()
        self.trend = TrendBuffer(period_s=5.0)

        self.registry = MoldRegistry()
        self.maps = SensorMapBook()
        self.curves = CurveBook()
        self.latch = CureLatch()
        self.rechecks = RecheckBook()
        # 现场上报的真实探头读数：{mold_id: {sensor: value}}。没有注入时
        # 控制循环按蒸汽压力折算一份仿真读数；缺报/越界一律走拒收复检。
        self._sensor_inputs: dict[str, dict[str, object]] = {}
        self.fleet = PressFleet.from_config(
            self.clock, self.bus, self.store.root, self.config
        )
        self.scheduler = BatchScheduler(self.bus, self.config.batch, self.curves)
        self._seed_reference_curve()

        self._cycles = 0
        self._lock = threading.RLock()
        self._register_namespaces()

    def _seed_reference_curve(self) -> None:
        """登记一份出厂参考曲线，供默认制品型号直接投产。"""

        self.curves.register(
            self.config.press.default_product_code,
            self.config.timer.default_nominal_s,
            self.clock.now(),
        )

    def _register_namespaces(self) -> None:
        """登记现场设备命名空间（控制台与审计共用同一套名字）。"""

        for area, device, kind in (
            ("mold", "registry", "mold-ledger"),
            ("mold", "sensor_map", "sensor-map"),
            ("lift", "clamp", "clamp-stroke"),
            ("steam", "header", "steam-header"),
            ("steam", "trap", "drain-trap"),
            ("steam", "valve", "steam-valve"),
            ("temp", "loop", "mold-temp-loop"),
            ("timer", "cure", "cure-timer"),
            ("timer", "latch", "cure-latch"),
            ("cool", "pump", "coolant-pump"),
            ("batch", "queue", "dispatch-queue"),
            ("batch", "scheduler", "fleet-scheduler"),
            ("audit", "journal", "production-audit"),
        ):
            self.namespaces.register(area, device, kind)

    def press(self, press_id: str = DEFAULT_PRESS_ID):
        """按编号取硫化机。"""

        return self.fleet.press(press_id)

    def register_mold(
        self,
        mold_id: str,
        product_code: str | None = None,
        cavity_count: int = 4,
        sensors: dict[int, str] | None = None,
    ) -> dict:
        """登记一台模具及其传感器映射。"""

        code = product_code or self.config.press.default_product_code
        if cavity_count > self.config.mold.max_cavities:
            raise RubberCureError(
                f"模具 {mold_id} 腔数 {cavity_count} 超过上限 {self.config.mold.max_cavities}",
                component="mold",
            )
        now = self.clock.now()
        record = self.registry.register(mold_id, code, cavity_count, now)
        layout = sensors or {
            cavity: f"{mold_id}-T{cavity + 1}" for cavity in range(cavity_count)
        }
        mapping = self.maps.register(mold_id, layout, now)
        return {
            "mold": record.as_dict(),
            "mapping": mapping.as_dict(),
        }

    def remap_sensor(self, mold_id: str, sensors: dict[int, str]) -> dict:
        """重新登记传感器映射（换探头或换模之后的维护动作）。"""

        mapping = self.maps.register(mold_id, sensors, self.clock.now())
        self.bus.publish(
            Event.of(
                "mold",
                "mold",
                "sensor_remapped",
                subject=mold_id,
                severity="notice",
                generation=mapping.generation,
                cavities=len(mapping.sensors),
            )
        )
        return mapping.as_dict()

    def retire_mold(self, mold_id: str, reason: str = "模具检修报废") -> dict:
        """把一台模具标记为报废，并从调度队列中撤出。"""

        record = self.registry.retire(mold_id, reason)
        self.scheduler.queue.remove(mold_id)
        self.rechecks.release(mold_id)
        self._sensor_inputs.pop(mold_id, None)
        self.bus.publish(
            Event.of(
                "mold",
                "mold",
                "mold_retired",
                subject=mold_id,
                severity="warning",
                reason=reason,
            )
        )
        return record.as_dict()

    def enqueue_mold(self, mold_id: str, thickness_mm: float | None = None, priority: int = 50) -> dict:
        """把一台已登记模具排进调度队列。"""

        thickness = (
            self.config.press.default_thickness_mm if thickness_mm is None else float(thickness_mm)
        )
        profile = BatchProfile.resolve(mold_id, thickness, self.registry, self.curves)
        entry = self.scheduler.enqueue(profile, self.clock.now(), priority=priority)
        return {"profile": profile.as_dict(), "entry": entry.as_dict()}

    def run_cure(
        self,
        mold_id: str,
        product_code: str | None = None,
        cavity_count: int = 4,
        thickness_mm: float | None = None,
        sensors: dict[int, str] | None = None,
        raw_readings: dict[str, object] | None = None,
        press_id: str = DEFAULT_PRESS_ID,
    ) -> dict:
        """在一台硫化机上跑完整的硫化顺序。

        模温采样缺探头或出现不可能的温度时，顺序在模温确认前被拒收：不确认
        模温、不启动硫化计时，挂待复检账并提示复检，复检通过后再重判。
        """

        thickness = (
            self.config.press.default_thickness_mm if thickness_mm is None else float(thickness_mm)
        )
        registered = self.register_mold(
            mold_id,
            product_code=product_code,
            cavity_count=cavity_count,
            sensors=sensors,
        )
        mapping = self.maps.latest_for(mold_id)
        if raw_readings is None:
            raw_readings = {
                sensor: self.config.temp.setpoint_c + (cavity - (cavity_count - 1) / 2.0) * 0.4
                for cavity, sensor in mapping.sensors
            }
        plan = CureRunPlan(
            mold_id=mold_id,
            product_code=registered["mold"]["product_code"],
            cavity_count=cavity_count,
            thickness_mm=thickness,
            sensors={cavity: sensor for cavity, sensor in mapping.sensors},
            raw_readings=raw_readings,
            target_kpa=self.config.steam.target_pressure_kpa,
            warmup_start_c=REFERENCE_TEMPERATURE_C,
            temp_hold_s=self.config.temp.confirm_hold_s,
        )
        with self._lock:
            sequence = CureSequence(
                clock=self.clock,
                bus=self.bus,
                config=self.config,
                registry=self.registry,
                maps=self.maps,
                curves=self.curves,
                press=self.press(press_id),
            )
            try:
                report = sequence.run(plan)
            except SensorReadError as error:
                record = self.rechecks.reject(
                    mold_id,
                    press_id,
                    mapping.generation,
                    error.problems,
                    self.clock.now(),
                )
                self.note_error(error)
                self.bus.publish(
                    Event.of(
                        "temp",
                        "temp",
                        "mold_temp_rejected",
                        subject=mold_id,
                        severity="alarm",
                        press_id=press_id,
                        generation=mapping.generation,
                        problems=list(record.problems),
                        message=error.message,
                    )
                )
                return {
                    "mold_id": mold_id,
                    "ok": False,
                    "temp_judgement": "rejected",
                    "recheck_required": True,
                    "message": error.message,
                    "recheck": record.as_dict(),
                    "steps": sequence.recorder.names(),
                }
            report["ok"] = True
            report["temp_judgement"] = "ok"
            report["topology_missing"] = sequence.verify_topology(self.topology)
            return report

    def run_open(self, press_id: str = DEFAULT_PRESS_ID) -> dict:
        """在一台硫化机上跑完整的开模顺序。"""

        with self._lock:
            sequence = OpenSequence(
                clock=self.clock,
                bus=self.bus,
                latch=self.latch,
                press=self.press(press_id),
            )
            return sequence.run()

    def run_dispatch(self, press_id: str = DEFAULT_PRESS_ID) -> dict:
        """在一台硫化机上跑一次机群调度顺序。"""

        with self._lock:
            sequence = DispatchSequence(
                clock=self.clock,
                bus=self.bus,
                registry=self.registry,
                scheduler=self.scheduler,
                fleet=self.fleet,
                press=self.press(press_id),
            )
            return sequence.run()

    def run_stop(self, press_id: str = DEFAULT_PRESS_ID) -> dict:
        """在一台硫化机上跑停运顺序。"""

        with self._lock:
            sequence = StopSequence(
                clock=self.clock, bus=self.bus, press=self.press(press_id)
            )
            return sequence.run()

    def cycle(self, inputs: CycleInputs) -> dict:
        """推进一个机群控制周期，返回本周期摘要。"""

        state = inputs.clamped()
        with self._lock:
            self._cycles += 1
            moment = self.clock.now()
            per_press: list[dict] = []
            rejections: list[dict] = []
            for unit in self.fleet.all():
                outcome = self._reading_for(unit, state)
                mold_id = unit.mold_id
                if isinstance(outcome, SensorReadError):
                    assert mold_id is not None
                    mapping = self.maps.latest_for(mold_id)
                    record = self.rechecks.reject(
                        mold_id,
                        unit.press_id,
                        mapping.generation,
                        outcome.problems,
                        moment,
                    )
                    summary = unit.step(state.dt, None)
                    summary["temp_judgement"] = "rejected"
                    summary["recheck"] = record.as_dict()
                    per_press.append(summary)
                    rejections.append(record.as_dict())
                    self.bus.publish(
                        Event.of(
                            "temp",
                            "temp",
                            "mold_temp_rejected",
                            subject=record.mold_id,
                            severity="alarm",
                            press_id=unit.press_id,
                            generation=record.generation,
                            problems=list(record.problems),
                            message=outcome.message,
                        )
                    )
                    continue
                reading = outcome
                # 已挂待复检账时，只有显式复检通过才能销账；这期间哪怕新
                # 一帧读数正常，也不自动恢复判定，避免顶替/蒙混过去。
                if mold_id is not None and self.rechecks.is_pending(mold_id):
                    record = self.rechecks.pending_for(mold_id)
                    summary = unit.step(state.dt, None)
                    summary["temp_judgement"] = "pending_recheck"
                    summary["recheck"] = record.as_dict()
                    per_press.append(summary)
                    continue
                summary = unit.step(state.dt, reading)
                summary["temp_judgement"] = "ok"
                per_press.append(summary)
                self.trend.append("mold_temp", moment, reading.average_c())
                self.trend.append("steam_pressure", moment, unit.steam.pressure_kpa)
            self.trend.append("fleet_load", moment, len(self.fleet.idle_presses()) / self.fleet.size)
            return {
                "cycle": self._cycles,
                "dt": state.dt,
                "presses": per_press,
                "idle": [unit.press_id for unit in self.fleet.idle_presses()],
                "temp_rejections": rejections,
                "temp_recheck_pending": [
                    item.as_dict() for item in self.rechecks.pending()
                ],
            }

    def _reading_for(self, unit, state: CycleInputs):
        """取本周期模温读数；探头缺报或读数不可能时返回拒收异常，绝不顶替。

        返回 :class:`MoldTempReading` 表示整模采样可信；返回
        :class:`SensorReadError` 表示这一模模温判定必须拒收并挂复检。
        """

        mold_id = unit.mold_id
        if mold_id is None or mold_id not in self.maps.known_molds():
            return MoldTempReading(
                mold_id=mold_id or unit.press_id,
                generation=0,
                sampled_at=self.clock.now(),
                by_cavity={0: REFERENCE_TEMPERATURE_C + state.inlet_c * 0.05},
            )
        mapping = self.maps.latest_for(mold_id)
        injected = self._sensor_inputs.get(mold_id)
        if injected is None:
            base = REFERENCE_TEMPERATURE_C + unit.steam.pressure_kpa * 0.19
            raw = {
                sensor: base + (cavity - (len(mapping.sensors) - 1) / 2.0) * 0.6
                for cavity, sensor in mapping.sensors
            }
        else:
            raw = injected
        try:
            return read_mold_temp(
                mold_id,
                mapping,
                raw,
                self.clock.now(),
                plausible_min_c=self.config.temp.plausible_min_c,
                plausible_max_c=self.config.temp.plausible_max_c,
            )
        except SensorReadError as error:
            return error

    def report_sensor_readings(self, mold_id: str, raw: dict[str, object]) -> dict:
        """录入现场上报的一帧探头读数，供下一控制周期判定。"""

        if mold_id not in self.maps.known_molds():
            raise MoldError(f"模具 {mold_id} 尚未登记传感器映射", component="temp")
        values = {str(sensor): value for sensor, value in dict(raw).items()}
        self._sensor_inputs[mold_id] = values
        mapping = self.maps.latest_for(mold_id)
        problems = sensor_problems(
            mapping,
            values,
            self.config.temp.plausible_min_c,
            self.config.temp.plausible_max_c,
        )
        return {
            "mold_id": mold_id,
            "generation": mapping.generation,
            "sensors": values,
            "problems": problems,
            "acceptable": not problems,
        }

    def clear_sensor_readings(self, mold_id: str) -> dict:
        """撤掉手工录入的读数（回到按蒸汽压力折算的仿真采样）。"""

        self._sensor_inputs.pop(mold_id, None)
        return {"mold_id": mold_id, "cleared": True}

    def recheck_mold_temp(
        self,
        mold_id: str,
        readings: dict[str, object] | None = None,
        press_id: str = DEFAULT_PRESS_ID,
    ) -> dict:
        """对拒收的一模提交复检值并重新判定。

        复检值通过可信窗口校验才销账：整模每腔都有可信值时重判平均温度与
        腔间温差；仍有缺报或越界则继续拒收并累计复检次数。不传 ``readings``
        时，用最近一次录入（或当前周期折算）的读数重判。
        """

        record = self.rechecks.pending_for(mold_id)
        if record is None:
            raise RubberCureError(
                f"模具 {mold_id} 没有待复检的模温记录", component="temp"
            )
        mapping = self.maps.latest_for(mold_id)
        if readings is None:
            readings = self._sensor_inputs.get(mold_id)
        if readings is None:
            unit = self.fleet.press(press_id)
            base = REFERENCE_TEMPERATURE_C + unit.steam.pressure_kpa * 0.19
            readings = {
                sensor: base + (cavity - (len(mapping.sensors) - 1) / 2.0) * 0.6
                for cavity, sensor in mapping.sensors
            }
        raw = {str(sensor): value for sensor, value in dict(readings).items()}
        moment = self.clock.now()
        problems = sensor_problems(
            mapping,
            raw,
            self.config.temp.plausible_min_c,
            self.config.temp.plausible_max_c,
        )
        if problems:
            updated = self.rechecks.record_attempt(mold_id, False, moment, problems)
            self.bus.publish(
                Event.of(
                    "temp",
                    "temp",
                    "mold_temp_recheck_failed",
                    subject=mold_id,
                    severity="alarm",
                    press_id=record.press_id,
                    attempts=updated.attempts,
                    problems=problems,
                )
            )
            return {"ok": False, "recheck": updated.as_dict()}

        reading = read_mold_temp(mold_id, mapping, raw, moment)
        # 复检通过意味着现场已核对探头：复检值成为新的当前帧，下一周期按它判定。
        self._sensor_inputs[mold_id] = raw
        updated = self.rechecks.record_attempt(mold_id, True, moment)
        band = band_summary(
            reading.average_c(),
            self.config.temp.setpoint_c,
            self.config.temp.tolerance_c,
        )
        spread = reading.spread_c()
        verdict = {
            "ok": True,
            "recheck": updated.as_dict(),
            "reading": reading.as_dict(),
            "in_band": band["in_band"],
            "spread_ok": spread_ok(spread, self.config.temp.max_cavity_spread_c),
            "judgement": (
                "qualified"
                if band["in_band"]
                and spread_ok(spread, self.config.temp.max_cavity_spread_c)
                else "rechecked_out_of_band"
            ),
        }
        self.bus.publish(
            Event.of(
                "temp",
                "temp",
                "mold_temp_recheck_passed",
                subject=mold_id,
                severity="notice",
                press_id=record.press_id,
                attempts=updated.attempts,
                average_c=band["measured_c"],
                in_band=band["in_band"],
                spread_c=round(spread, 3),
            )
        )
        return verdict

    def concurrency_setpoint_probe(self, rounds: int = 24) -> dict:
        """用真实并行调用探测蒸汽压力目标值的仲裁结果。

        两个线程分别代表模温回路与蒸汽压力调节器，同时向母管提交目标压力。
        探测只读取最终提交值，用来确认共享写入方始终收敛到唯一目标值。
        """

        if rounds < 1:
            raise ValueError("探测轮数必须为正")
        unit = self.press(DEFAULT_PRESS_ID)
        submitted: list[float] = []

        def temp_writer() -> None:
            for _ in range(rounds):
                unit.steam.submit_setpoint("temp_loop", self.config.steam.target_pressure_kpa + 18.0, 70)
                submitted.append(unit.steam.arbiter.resolve().target_kpa)

        def pressure_writer() -> None:
            for _ in range(rounds):
                unit.steam.submit_setpoint("header_loop", self.config.steam.target_pressure_kpa - 12.0, 60)
                submitted.append(unit.steam.arbiter.resolve().target_kpa)

        threads = [
            threading.Thread(target=temp_writer),
            threading.Thread(target=pressure_writer),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        outcome = unit.steam.arbiter.resolve()
        return {
            "rounds": rounds,
            "winner": outcome.winner,
            "committed_kpa": round(outcome.target_kpa, 3),
            "samples": len(submitted),
            "distinct_targets": len({round(value, 3) for value in submitted}),
        }

    def concurrency_report_probe(self, rounds: int = 12) -> dict:
        """用真实并行上报探测完成上报的排序结果。

        多台硫化机在不同线程里并发上报完成，调度必须把同一批上报排成可复现
        顺序。探测比较两次排序的键序列是否完全一致。
        """

        if rounds < 1:
            raise ValueError("探测轮数必须为正")
        presses = self.fleet.all()

        def reporter(index: int, unit) -> None:
            for step in range(rounds):
                self.scheduler.record_report(
                    unit.press_id,
                    f"probe-{unit.press_id}-{step}",
                    self.config.press.default_product_code,
                    float(step),
                    1,
                )

        threads = [
            threading.Thread(target=reporter, args=(index, unit))
            for index, unit in enumerate(presses)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        reports = self.scheduler.recent_reports(limit=rounds * max(1, len(presses)))
        first = [
            (item["mold_id"], item["press_id"], item["sequence"])
            for item in self.scheduler.ordered_reports(limit=len(reports), window_s=0.0)
        ]
        second = [
            (item["mold_id"], item["press_id"], item["sequence"])
            for item in self.scheduler.ordered_reports(limit=len(reports), window_s=0.0)
        ]
        return {
            "reports": len(reports),
            "stable": first == second,
            "ordered": first[:5],
        }

    def note_error(self, error: RubberCureError) -> dict:
        """把平台异常写入审计，返回落盘记录。"""

        payload = error.as_record()
        record = self.audit.record(
            AuditRecord.create(
                self.clock.now(),
                payload["component"],
                "error",
                payload["type"],
                severity="warning",
                message=payload["message"],
            )
        )
        return record.as_dict()

    def reset_state(self) -> dict:
        """清掉所有硫化机的落盘状态（检修后重新标定用）。"""

        removed: list[str] = []
        for unit in self.fleet.all():
            removed.append(unit.store.root if unit.store.remove("steam_state") else "")
            unit.store.remove("cure_timer")
            unit.store.remove("temp_confirmation")
        self.latch.release_all()
        return {"storage_root": self.store.root, "cleared": [item for item in removed if item]}

    def status(self) -> dict:
        """汇总状态快照，供控制台与命令行读取。"""

        return {
            "platform": self.config.name,
            "project_id": self.config.project_id,
            "unit": self.namespaces.unit,
            "cycles": self._cycles,
            "now": round(self.clock.now(), 3),
            "utc": self.clock.utc_iso(),
            "molds": summarize_molds(self.registry),
            "sensor_maps": self.maps.snapshot(),
            "temp_rechecks": self.rechecks.snapshot(),
            "curves": self.curves.snapshot(),
            "latch": self.latch.snapshot(),
            "fleet": summarize_fleet(self.fleet),
            "scheduler": self.scheduler.snapshot(),
            "presses": self.fleet.snapshot(),
            "trend": self.trend.snapshot(),
            "namespaces_by_area": self.namespaces_by_area(),
            "audit": self.audit.summary(),
            "endpoints": self.endpoints(),
        }

    def namespaces_by_area(self) -> dict:
        """按工位分组返回命名空间，控制台据此渲染分区清单。"""

        areas = sorted({entry["area"] for entry in self.namespaces.describe()})
        return {
            area: [entry.as_dict() for entry in self.namespaces.in_area(area)]
            for area in areas
        }

    def endpoints(self) -> dict:
        """返回本平台对外暴露的页面与接口清单。"""

        return {
            "pages": list(self.config.pages),
            "api": [
                "/api/status",
                "/api/trend",
                "/api/alarms",
                "/api/audit",
                "/api/journal",
                "/api/readiness",
                "/api/cure",
                "/api/open",
                "/api/dispatch",
                "/api/stop",
                "/api/step",
                "/api/enqueue",
                "/api/remap",
                "/api/retire",
                "/api/sensor_readings",
                "/api/temp_recheck",
            ],
        }

    def topology_summary(self) -> dict:
        """链路拓扑加上每个节点的上下游关系。"""

        summary = self.topology.as_dict()
        summary["upstream"] = {
            name: self.topology.upstream_of(name) for name in self.topology.nodes
        }
        summary["downstream"] = {
            name: self.topology.downstream_of(name) for name in self.topology.nodes
        }
        summary["missing"] = verify_topology(
            self.topology,
            (
                ("mold", "lift"),
                ("lift", "steam"),
                ("steam", "temp"),
                ("temp", "timer"),
                ("timer", "press"),
                ("press", "batch"),
            ),
        )
        return summary


def build_platform(
    config: PlatformConfig | None = None,
    clock: Clock | None = None,
    storage_root: str | None = None,
    unit: str = DEFAULT_UNIT,
) -> Platform:
    """构造平台；``storage_root`` 给出时覆盖配置里的存储目录。"""

    effective = config or default_config()
    if storage_root:
        effective = PlatformConfig(
            name=effective.name,
            project_id=effective.project_id,
            storage_root=os.path.abspath(storage_root),
            mold=effective.mold,
            press=effective.press,
            steam=effective.steam,
            temp=effective.temp,
            timer=effective.timer,
            cool=effective.cool,
            batch=effective.batch,
            pages=effective.pages,
        )
    return Platform(config=effective, clock=clock, unit=unit)
