"""模温采样整理。

传感器上报的是"传感器编号 → 温度"，而控制与判定需要"腔位 → 温度"。整理
这一步必须按模具最新代际的映射来做：换模或换探头之后若仍按旧代际解析，
腔位就会错配，某一腔的低温被另一腔的高温顶掉，欠硫直接漏判。

整理同时是一道拒收关口：某一腔的探头没上报、或上报了明显不可能的温度
（非数值、NaN、无穷、落在物理可信窗口之外），都不允许用别的腔的值顶替，
而要把本次采样发现的全部问题收集起来抛 :class:`SensorReadError`。调用方
必须据此拒收这一模的模温判定并提示复检，复检通过之后才能重新判定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..errors import (
    MoldError,
    SensorMissingError,
    SensorReadError,
    SensorValueError,
)
from ..mold.mapping import SensorMapping

DEFAULT_PLAUSIBLE_MIN_C = 0.0
DEFAULT_PLAUSIBLE_MAX_C = 300.0

REASON_MISSING = "missing"
REASON_NOT_A_NUMBER = "not_a_number"
REASON_NOT_FINITE = "not_finite"
REASON_OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class MoldTempReading:
    """一次按腔位整理的模温采样。"""

    mold_id: str
    generation: int
    sampled_at: float
    by_cavity: dict[int, float]

    def cavity_values(self) -> list[float]:
        return [self.by_cavity[index] for index in sorted(self.by_cavity)]

    def average_c(self) -> float:
        values = self.cavity_values()
        return sum(values) / len(values) if values else 0.0

    def minimum_c(self) -> float:
        values = self.cavity_values()
        return min(values) if values else 0.0

    def maximum_c(self) -> float:
        values = self.cavity_values()
        return max(values) if values else 0.0

    def spread_c(self) -> float:
        if not self.by_cavity:
            return 0.0
        return self.maximum_c() - self.minimum_c()

    def as_dict(self) -> dict:
        return {
            "mold_id": self.mold_id,
            "generation": self.generation,
            "sampled_at": round(self.sampled_at, 3),
            "by_cavity": {str(index): round(value, 3) for index, value in sorted(self.by_cavity.items())},
            "average_c": round(self.average_c(), 3),
            "minimum_c": round(self.minimum_c(), 3),
            "maximum_c": round(self.maximum_c(), 3),
            "spread_c": round(self.spread_c(), 3),
        }


def _displayable(raw: Any) -> Any:
    """把原始读数转成可安全落审计日志（JSON）的形式。"""

    if isinstance(raw, float) and not math.isfinite(raw):
        if math.isnan(raw):
            return "nan"
        return "inf" if raw > 0 else "-inf"
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return repr(raw)
    return raw


def sensor_problems(
    mapping: SensorMapping,
    raw_values: dict[str, Any],
    plausible_min_c: float = DEFAULT_PLAUSIBLE_MIN_C,
    plausible_max_c: float = DEFAULT_PLAUSIBLE_MAX_C,
) -> list[dict]:
    """逐腔核对探头读数，返回本次采样的全部问题；空列表表示整模采样可信。"""

    problems: list[dict] = []
    for cavity, sensor in mapping.sensors:
        if sensor not in raw_values:
            problems.append(
                {"cavity": cavity, "sensor": sensor, "reason": REASON_MISSING}
            )
            continue
        raw = raw_values[sensor]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            problems.append(
                {
                    "cavity": cavity,
                    "sensor": sensor,
                    "reason": REASON_NOT_A_NUMBER,
                    "raw": _displayable(raw),
                }
            )
            continue
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            problems.append(
                {
                    "cavity": cavity,
                    "sensor": sensor,
                    "reason": REASON_NOT_FINITE,
                    "raw": _displayable(raw),
                }
            )
        elif value < plausible_min_c or value > plausible_max_c:
            problems.append(
                {
                    "cavity": cavity,
                    "sensor": sensor,
                    "reason": REASON_OUT_OF_RANGE,
                    "raw": value,
                    "plausible_min_c": plausible_min_c,
                    "plausible_max_c": plausible_max_c,
                }
            )
    return problems


def read_error_for(mold_id: str, problems: list[dict]) -> SensorReadError:
    """按问题构成抛出最合适的拒收异常（缺报／越界／二者兼有）。"""

    missing = [item for item in problems if item["reason"] == REASON_MISSING]
    invalid = [item for item in problems if item["reason"] != REASON_MISSING]
    if missing and invalid:
        error_cls = SensorReadError
        detail = (
            f"有 {len(missing)} 个腔探头缺报、{len(invalid)} 个腔读数明显不可能"
        )
    elif missing:
        error_cls = SensorMissingError
        detail = f"有 {len(missing)} 个腔的探头没有上报温度"
    else:
        error_cls = SensorValueError
        detail = f"有 {len(invalid)} 个探头报出明显不可能的温度"
    return error_cls(
        f"模具 {mold_id} {detail}，本一模模温判定拒收，请复检探头后重判",
        problems=problems,
    )


def read_mold_temp(
    mold_id: str,
    mapping: SensorMapping,
    raw_values: dict[str, Any],
    sampled_at: float,
    plausible_min_c: float = DEFAULT_PLAUSIBLE_MIN_C,
    plausible_max_c: float = DEFAULT_PLAUSIBLE_MAX_C,
) -> MoldTempReading:
    """按映射把原始传感器读数整理成按腔位的模温。

    任一腔位缺报或读数越界都直接抛出 :class:`SensorReadError`，绝不拿别的腔
    或推算值顶替；异常的 ``problems`` 列出全部待复检腔位。
    """

    if mapping.mold_id != mold_id:
        raise MoldError(
            f"映射属于 {mapping.mold_id}，与读取对象 {mold_id} 不一致", component="temp"
        )
    problems = sensor_problems(
        mapping, raw_values, plausible_min_c, plausible_max_c
    )
    if problems:
        raise read_error_for(mold_id, problems)
    by_cavity: dict[int, float] = {}
    for cavity, sensor in mapping.sensors:
        by_cavity[cavity] = float(raw_values[sensor])
    return MoldTempReading(
        mold_id=mold_id,
        generation=mapping.generation,
        sampled_at=float(sampled_at),
        by_cavity=by_cavity,
    )
