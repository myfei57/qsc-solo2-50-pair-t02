"""模温复检台账。

某一模有探头缺报或读数明显不可能时，这一模的模温判定直接拒收：既不用
别的腔的值顶替，也不让平均温度、腔间温差和硫化计时继续走。拒收之后必须
挂一条"待复检"记录，把出问题的腔位、传感器和原因摊给现场；复检值通过
可信窗口校验后才清账，并由调用方拿着复检值重新判定。复检仍不合格则更新
记录、继续拒收，绝不自动恢复判定。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TempRecheck:
    """一条待复检（或复检历史中的）模温拒收记录。"""

    mold_id: str
    press_id: str
    generation: int
    rejected_at: float
    problems: tuple[dict, ...]
    attempts: int = 0
    last_attempt_at: float | None = None
    last_result: str = "pending"

    @property
    def pending(self) -> bool:
        return self.last_result == "pending"

    def cavity_sensors(self) -> dict[int, str]:
        return {int(item["cavity"]): str(item["sensor"]) for item in self.problems}

    def as_dict(self) -> dict:
        return {
            "mold_id": self.mold_id,
            "press_id": self.press_id,
            "generation": self.generation,
            "rejected_at": round(self.rejected_at, 3),
            "problems": list(self.problems),
            "attempts": self.attempts,
            "last_attempt_at": (
                round(self.last_attempt_at, 3) if self.last_attempt_at is not None else None
            ),
            "last_result": self.last_result,
            "pending": self.pending,
            "cavities": sorted(self.cavity_sensors()),
        }


@dataclass
class RecheckBook:
    """按模具保存当前待复检记录；通过复检后自动销账。"""

    _pending: dict[str, TempRecheck] = field(default_factory=dict)
    _resolved: list[TempRecheck] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def reject(
        self,
        mold_id: str,
        press_id: str,
        generation: int,
        problems: list[dict],
        rejected_at: float,
    ) -> TempRecheck:
        """登记一条待复检记录；同一模具已挂账时保留首次拒收时间。"""

        with self._lock:
            existing = self._pending.get(mold_id)
            if existing is not None:
                record = TempRecheck(
                    mold_id=mold_id,
                    press_id=press_id,
                    generation=generation,
                    rejected_at=existing.rejected_at,
                    problems=tuple(problems),
                    attempts=existing.attempts,
                    last_attempt_at=existing.last_attempt_at,
                    last_result=existing.last_result,
                )
            else:
                record = TempRecheck(
                    mold_id=mold_id,
                    press_id=press_id,
                    generation=generation,
                    rejected_at=float(rejected_at),
                    problems=tuple(problems),
                )
            self._pending[mold_id] = record
            return record

    def record_attempt(
        self,
        mold_id: str,
        passed: bool,
        attempted_at: float,
        problems: list[dict] | None = None,
    ) -> TempRecheck:
        """记一次复检结果；通过则销账并转入已解决，不通过则继续挂账。"""

        with self._lock:
            current = self._pending.get(mold_id)
            if current is None:
                raise KeyError(f"模具 {mold_id} 没有待复检的模温记录")
            updated = TempRecheck(
                mold_id=mold_id,
                press_id=current.press_id,
                generation=current.generation,
                rejected_at=current.rejected_at,
                problems=tuple(problems) if problems is not None else current.problems,
                attempts=current.attempts + 1,
                last_attempt_at=float(attempted_at),
                last_result="passed" if passed else "failed",
            )
            if passed:
                self._pending.pop(mold_id, None)
                self._resolved.append(updated)
                self._trim_resolved()
            else:
                self._pending[mold_id] = updated
            return updated

    def release(self, mold_id: str) -> None:
        """换模或报废时撤掉挂账，避免旧账误锁下一模。"""

        with self._lock:
            self._pending.pop(mold_id, None)

    def pending_for(self, mold_id: str) -> TempRecheck | None:
        with self._lock:
            return self._pending.get(mold_id)

    def is_pending(self, mold_id: str) -> bool:
        return self.pending_for(mold_id) is not None

    def pending(self) -> list[TempRecheck]:
        with self._lock:
            return [self._pending[key] for key in sorted(self._pending)]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "pending": [record.as_dict() for record in self.pending()],
                "recent_resolved": [record.as_dict() for record in reversed(self._resolved[-20:])],
            }

    def _trim_resolved(self, limit: int = 200) -> None:
        if len(self._resolved) > limit:
            del self._resolved[: len(self._resolved) - limit]
