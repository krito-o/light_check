"""部署引擎：薄封装 SlotDetector，提供事件回调 + 告警状态查询。

用法：
    from engine import LightCheckEngine
    engine = LightCheckEngine("config.json")
    engine.process_frame(frame)  # 逐帧处理
    engine.alarm_slots()         # 查询故障槽
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

from slot_detector import SlotDetector


def _default_alarm_callback(roi_index, roi_name, old_status, new_status, frame_idx, details):
    """内置默认告警回调：状态变为 off/steady_on 时输出告警。"""
    if new_status in ("off", "steady_on"):
        print(f"[告警] f{frame_idx:5d} {roi_name} -> {new_status}  "
              f"median={details.get('median',0):.1f} osc={details.get('osc',0)} "
              f"area={details.get('area_med',0):.0f} gray={details.get('gray_med',0):.0f}",
              file=sys.stderr)


class LightCheckEngine:
    """部署引擎：封装 SlotDetector，提供便捷的告警查询和回调接口。"""

    def __init__(self, config_or_path, callbacks=None):
        """创建引擎。

        参数：
            config_or_path: config dict 或 json 文件路径。
            callbacks: 可选 dict，覆盖 SlotDetector 的回调。
                      内置默认告警回调（on_status 变为 off/steady_on 时输出）。
                      传入 None 保留默认；传入空 dict {} 禁掉所有回调。
        """
        if isinstance(config_or_path, str):
            with open(config_or_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = dict(config_or_path)

        # 合并回调：默认告警回调 + 用户自定义
        final_callbacks = {"on_status": _default_alarm_callback}
        if callbacks is not None:
            final_callbacks.update(callbacks)

        self.detector = SlotDetector(self.config, callbacks=final_callbacks)
        self._last_log_time = ""
        self._alarm_log_path = None

    def process_frame(self, frame) -> dict:
        """处理一帧，返回结构化结果。

        返回 dict:
            frame_idx: int
            slots: [{name, in_slot, light_status, ...}, ...]
            events: [{type, roi, name, ...}, ...]  (本帧发生的事件)
        """
        states, info = self.detector.process_frame(frame)
        return {
            "frame_idx": self.detector.frame_idx,
            "slots": self.detector.all_status(),
            "events": info.get("events", []),
        }

    def slot_status(self, i):
        """返回第 i 个槽的结构化状态。"""
        return self.detector.slot_status(i)

    def all_status(self):
        """返回所有槽的结构化状态列表。"""
        return self.detector.all_status()

    def is_any_alarm(self):
        """是否有槽在故障状态（off 或 steady_on）。"""
        return any(s["light_status"] in ("off", "steady_on")
                   for s in self.all_status())

    def alarm_slots(self):
        """返回当前故障槽列表。"""
        return [s for s in self.all_status()
                if s["light_status"] in ("off", "steady_on")]

    def alarm_log(self, log_dir="logs"):
        """写入 JSON 行告警日志到 logs/alarm_<日期>.jsonl。

        每行一条 JSON 记录，包含时间戳和所有槽状态。
        返回日志文件路径。
        """
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        if date_str != self._last_log_time:
            self._last_log_time = date_str
            import os
            os.makedirs(log_dir, exist_ok=True)
            self._alarm_log_path = f"{log_dir}/alarm_{date_str}.jsonl"

        record = {
            "time": now.isoformat(),
            "frame_idx": self.detector.frame_idx,
            "slots": self.all_status(),
        }
        with open(self._alarm_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self._alarm_log_path