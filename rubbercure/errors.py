"""平台统一异常类型。

硫化现场必须区分几类失败：顺序被破坏属于工程接线或逻辑错误，必须在调试
阶段就暴露；联锁条件不满足是运行期可恢复的，设备状态一旦满足就能继续；
模具登记冲突与持久化失败则要单独归类，便于按类型复盘。审计日志按异常
类型分别落盘，复盘时才能一眼分清"顺序错"还是"条件没到"。
"""

from __future__ import annotations


class RubberCureError(Exception):
    """所有平台异常的基类，携带出错的组件名。"""

    def __init__(self, message: str, *, component: str = "rubbercure") -> None:
        super().__init__(message)
        self.message = message
        self.component = component

    def as_record(self) -> dict:
        """转成审计日志可以直接落盘的一条记录。"""

        return {
            "type": type(self).__name__,
            "component": self.component,
            "message": self.message,
        }


class SequenceError(RubberCureError):
    """硫化、开模、调度或停运顺序被破坏。"""


class InterlockError(RubberCureError):
    """联锁条件不满足，动作被拒绝。"""


class StorageError(RubberCureError):
    """持久化读写失败。"""


class ConfigError(RubberCureError):
    """配置缺失或取值非法。"""


class NamespaceError(RubberCureError):
    """命名空间重复注册或解析失败。"""


class MoldError(RubberCureError):
    """模具登记冲突、映射缺失或代际过期。"""


class SensorReadError(MoldError):
    """模温采样不可信（探头缺报或读数越界），本一模必须拒收并复检。

    ``problems`` 携带本次采样发现的全部问题（腔位、传感器、读数、原因），
    供复检台账与控制台直接展示，不允许用别的腔或推算值顶替后继续判定。
    """

    def __init__(self, message: str, *, component: str = "temp", problems: list[dict] | None = None) -> None:
        super().__init__(message, component=component)
        self.problems = list(problems or [])

    def as_record(self) -> dict:
        record = super().as_record()
        record["problems"] = self.problems
        return record


class SensorMissingError(SensorReadError):
    """腔位对应的探头没有上报温度。"""


class SensorValueError(SensorReadError):
    """探头上报了明显不可能的温度（非数值或落在物理可信窗口之外）。"""


class CureError(RubberCureError):
    """硫化计时状态非法。"""
