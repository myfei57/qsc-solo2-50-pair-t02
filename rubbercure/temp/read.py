"""模温采样整理。

传感器上报的是"传感器编号 → 温度"，而控制与判定需要"腔位 → 温度"。整理
这一步必须按模具最新代际的映射来做：换模或换探头之后若仍按旧代际解析，
腔位就会错配，某一腔的低温被另一腔的高温顶掉，欠硫直接漏判。

探头没上报、或上报了明显不可能的温度时，这里直接拒收：不拿别的腔的值
顶替，也不让缺腔的平均值蒙混过关。判定方拿到的是异常，由上层提示复检；
探头复检通过后，用新的完整采样重新整理、重新判定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import MoldError
from ..mold.mapping import SensorMapping


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


def read_mold_temp(
    mold_id: str,
    mapping: SensorMapping,
    raw_values: dict[str, float],
    sampled_at: float,
    plausible_range: tuple[float, float] | None = None,
) -> MoldTempReading:
    """按映射把原始传感器读数整理成按腔位的模温。

    任一腔缺报、报非数值、或（给出 ``plausible_range`` 时）读数越出物理
    可信区间，都抛出 :class:`MoldError`：本模模温判定随之拒收，待探头
    复检，绝不用其他腔的值顶替。
    """

    if mapping.mold_id != mold_id:
        raise MoldError(
            f"映射属于 {mapping.mold_id}，与读取对象 {mold_id} 不一致", component="temp"
        )
    by_cavity: dict[int, float] = {}
    for cavity, sensor in mapping.sensors:
        if sensor not in raw_values:
            raise MoldError(
                f"模具 {mold_id} 第 {cavity} 腔的传感器 {sensor} 没有上报温度，"
                "本模模温判定拒收，待探头复检",
                component="temp",
            )
        value = float(raw_values[sensor])
        if not math.isfinite(value):
            raise MoldError(
                f"模具 {mold_id} 第 {cavity} 腔的传感器 {sensor} 上报了非数值温度，"
                "本模模温判定拒收，待探头复检",
                component="temp",
            )
        if plausible_range is not None:
            low_c, high_c = plausible_range
            if not low_c <= value <= high_c:
                raise MoldError(
                    f"模具 {mold_id} 第 {cavity} 腔的传感器 {sensor} 上报 {value:.2f}℃，"
                    f"越出可信区间 [{low_c:.1f}, {high_c:.1f}]℃，"
                    "本模模温判定拒收，待探头复检",
                    component="temp",
                )
        by_cavity[cavity] = value
    return MoldTempReading(
        mold_id=mold_id,
        generation=mapping.generation,
        sampled_at=float(sampled_at),
        by_cavity=by_cavity,
    )
