"""核心检测器：每 ROI 独立状态机 + 信号提取 + 跨度级判定 + 多跨度投票。

流程（每个 ROI 独立运行）：
  WAITING（等待入槽）──入槽触发或 skip_enter──> DETECTING（持续灯状态检测）
  DETECTING 状态下持续采集该 ROI 的"亮像素 green_excess 均值"信号，用滑动跨度做
  最终判定，并通过多跨度投票抑制瞬态干扰。检测到设备被取走可回到 WAITING。

多 ROI 设计（每槽独立状态机）：
  每个 ROI 封装为一个 PerLightState 对象，独立维护 state/入槽计数/信号缓冲/确认状态。
  任一槽入槽不会连累其他槽（旧版 any(entered) 会让一个槽触发全部进 DETECTING，
  多槽位部署不可用）。单 ROI 退化为 lights[0]，行为不变。

与原设计文档的偏差（经实测验证，详见 README/TECH_REPORT）：
  - 信号用"亮像素 green_excess 均值"(ge>lit_thr 的亮像素均值)，而非全框均值。
    原因：框比灯大时全框均值把灯信号稀释成 0；亮像素均值只取灯体亮像素，不稀释。
  - 面积门区分"反光"(面积小)与"真灯"(面积大)；振荡门区分"闪烁"与"常亮"。
  - 参考区(ref)已移除：实测 ref 自适应阈值无效（环境无绿色时退化固定值，且面积门
    模式下 on_thr 不参与判定）。
  - 入槽检测每 ROI 独立（亮度法/背景差分法），不再 any()。
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from light_classifier import (
    classify_span,
    classify_window,
    green_excess,
)


class PerLightState:
    """单个槽位(ROI)的状态机 + 运行时缓冲。

    封装一个 ROI 从 WAITING→DETECTING→WAITING 的完整状态，以及入槽/取走计数、
    信号缓冲、投票确认。配置参数（阈值、跨度大小等）由 SlotDetector 持有并在
    方法调用时传入，本类不重复存储配置。
    """

    def __init__(self, roi, brightness_baseline_win, buf_len, confirm_spans, enter_win):
        self.roi = roi
        # 状态机
        self.state = "WAITING"
        self.enter_counter = 0
        self.leave_counter = 0
        self.detecting_frames = 0
        self.next_span_at = 0
        # 背景差分法（每 ROI 独立背景模型）
        self.bg_model = None
        self.bg_init_counter = 0
        # 入槽检测：绿色面积滑动窗口（面积法判入/出槽）
        # WAITING 阶段持续采集绿色面积，窗口内超阈占比达标判入槽
        self.area_buf = deque(maxlen=enter_win)
        self.stable_area = 0.0  # 入槽后稳定面积（DETECTING 阶段维护，供出槽判定参考）
        # 亮度法（保留兼容，旧 config 用）：灰度滑动窗口估计空槽基线 + 入槽时冻结快照
        self.brightness_buf = deque(maxlen=brightness_baseline_win)
        self.empty_baseline = None   # 入槽时冻结的空槽基线（供取走判定）
        self.last_baseline = None    # 最新算出的基线（供 enter_detecting 冻结）
        # 信号缓冲
        self.signals = deque(maxlen=buf_len)
        self.area_signals = deque(maxlen=buf_len)
        self.gray_signals = deque(maxlen=buf_len)
        # 投票确认
        self.recent_spans = deque(maxlen=max(confirm_spans, 1))
        self.confirmed = "pending"
        self.last_output = "pending"
        self.last_in_slot = "pending"  # 最近一次在槽的确认状态（出槽reset后保留，供汇总展示）

    def enter_detecting(self, span_size):
        """WAITING → DETECTING：重置检测计数，冻结空槽基线，清空信号缓冲。"""
        self.state = "DETECTING"
        self.enter_counter = 0
        self.leave_counter = 0
        self.detecting_frames = 0
        self.next_span_at = span_size  # 攒满一个跨度后开始判定
        # 冻结入槽时的空槽基线快照，供 DETECTING 阶段取走判定
        if self.last_baseline is not None:
            self.empty_baseline = self.last_baseline
        self.signals.clear()
        self.area_signals.clear()
        self.gray_signals.clear()
        self.recent_spans.clear()
        self.brightness_buf.clear()
        self.stable_area = 0.0
        self.confirmed = "pending"
        self.last_output = "pending"

    def reset_to_waiting(self):
        """DETECTING → WAITING：设备被取走，重置等待下一次入槽。

        不清 area_buf/brightness_buf（继续维护空槽基线，因为取走后该槽回到空槽态，
        历史有助于下次入槽判定）。
        """
        self.state = "WAITING"
        self.enter_counter = 0
        self.leave_counter = 0
        self.bg_model = None
        self.bg_init_counter = 0
        self.empty_baseline = None
        self.stable_area = 0.0
        self.signals.clear()
        self.area_signals.clear()
        self.gray_signals.clear()
        self.recent_spans.clear()
        self.confirmed = "pending"
        self.last_output = "pending"


class SlotDetector:
    """逐帧处理视频，输出每个灯 ROI 的工作状态。每 ROI 独立状态机。"""

    def __init__(self, config, callbacks=None):
        """创建检测器。

        callbacks: 可选 dict，事件回调：
            "on_enter":  callable(roi_index, roi_name, frame_idx)   — 入槽
            "on_leave":  callable(roi_index, roi_name, frame_idx, last_status)  — 出槽
            "on_status": callable(roi_index, roi_name, old_status, new_status, frame_idx, details)  — 灯状态变化
        """
        self._callbacks = callbacks or {}
        # --- ROI 配置 ---
        # roi_list: [{'name','norm':{'X','Y','W','H'}, 可选 'expected'}]，norm 为 0-100 归一化坐标。
        # 用户只需写 norm 的 XYWH，绝对像素坐标 x/y/w/h 由 norm + image_size 自动转换。
        # 若 roi 已含 x/y/w/h（旧 config）则直接用，向后兼容。
        img = config.get("image_size", {})
        img_w = int(img.get("w", 1920))
        img_h = int(img.get("h", 1080))
        self.roi_list = []
        for roi in config["roi_list"]:
            roi = dict(roi)  # 浅拷贝，避免改 config 原对象
            if "x" not in roi or "y" not in roi or "w" not in roi or "h" not in roi:
                norm = roi.get("norm")
                if norm:
                    roi["x"] = round(norm["X"] / 100 * img_w)
                    roi["y"] = round(norm["Y"] / 100 * img_h)
                    roi["w"] = round(norm["W"] / 100 * img_w)
                    roi["h"] = round(norm["H"] / 100 * img_h)
            self.roi_list.append(roi)
        self.n_lights = len(self.roi_list)

        # --- 入槽检测配置 ---
        # enter_method:
        #   "area"      = 绿色面积法（推荐，固定摄像头现场监控）：ROI 内 green_excess 亮像素
        #                 面积持续超阈判入槽、持续归零判出槽。设备入槽后灯体（无论点亮与否，
        #                 LED 点亮或仅绿色外壳）都会带来稳定绿色面积；机械臂移动时绿色仅短暂
        #                 掠过。用滑动窗口超阈占比过滤瞬态。不依赖灯是否点亮，能覆盖入槽不亮。
        #   "brightness"= 灰度法（旧，手持测试视频）：ROI 灰度偏离空槽基线判入槽。
        #                 注意：固定摄像头现场不可用——机械臂移动/设备空中停留都会改变灰度。
        #   "bg_diff"   = 背景差分法（原设计，需视频开头为空槽）。
        self.enter_method = config.get("enter_method", "area")
        self.skip_enter = bool(config.get("skip_enter", False))
        self.enter_threshold = float(config.get("enter_threshold", 30.0))
        self.enter_frames = int(config.get("enter_frames", 10))
        self.init_frames = int(config.get("init_frames", 100))
        self.alpha = float(config.get("alpha", 0.05))
        # 面积法参数
        ea = config.get("enter_area", {})
        self.area_enter = float(ea.get("area_thr", 100.0))   # 绿色面积入槽阈值(像素)
        self.enter_win = int(ea.get("win", 15))              # 滑动窗口(帧)
        self.enter_ratio = float(ea.get("ratio", 0.6))       # 窗口内超阈帧占比判入槽
        self.leave_ratio = float(ea.get("leave_ratio", 0.4)) # 窗口内超阈帧占比≤此值判出槽
        self.leave_drop_ratio = float(ea.get("leave_drop_ratio", 0.4))  # 当前面积<稳定值*此值判出槽(提前)
        # brightness 法参数（保留兼容）
        eb = config.get("enter_brightness", {})
        self.brightness_margin = float(eb.get("margin", 15.0))
        self.brightness_baseline_win = int(eb.get("baseline_win", 300))
        self.brightness_percentile = float(eb.get("percentile", 5.0))

        # --- 闪烁/状态检测配置 ---
        self.fps = float(config.get("fps", 50))
        self.span_size = int(config.get("span_size", 150))   # 最终判定跨度（帧）
        self.span_step = int(config.get("span_step", 25))    # 跨度滑动步长（帧）
        self.window_size = int(config.get("window_size", 40))  # 逐窗日志窗口（帧）
        self.confirm_spans = int(config.get("confirm_spans", 3))  # 连续一致跨度数才输出
        # 分类器阈值
        clf = config.get("classifier", {})
        self.on_threshold = float(clf.get("on_threshold", 5.0))
        self.osc_min = int(clf.get("osc_min", 6))
        self.min_sep = float(clf.get("min_sep", 8.0))
        self.smooth_kernel = int(clf.get("smooth_kernel", 5))
        # 信号提取：ROI 内"亮像素 green_excess 均值"(ge>lit_thr 的亮像素均值,无则0)。
        self.lit_thr = float(clf.get("lit_thr", 15.0))
        # 面积门：跨度内亮像素面积中位数 > area_min 才算"真灯"，否则 off。
        self.area_min = float(clf.get("area_min", 150.0))
        self.lit_ge = float(clf.get("lit_ge", 5.0))
        self.lit_ratio_min = float(clf.get("lit_ratio_min", 0.15))
        # 灰度门：区分"不亮"(外壳绿,LED未亮,灰度~51)与"常亮"(LED点亮,灰度~110)。
        # 仅"真灯体+不振荡"分支生效。取 80 可区分(实测不亮51、常亮110)。
        self.gray_thr = float(clf.get("gray_thr", 80.0))
        # 设备被取走判定：每 ROI 独立
        self.leave_check = bool(config.get("leave_check", False))

        # --- 每 ROI 独立状态机 ---
        buf_len = max(self.span_size, self.window_size) + 5
        self.lights = [PerLightState(roi, self.brightness_baseline_win, buf_len,
                                     self.confirm_spans, self.enter_win)
                       for roi in self.roi_list]

        # 全局帧计数
        self.frame_idx = -1

    @property
    def confirmed(self):
        """每 ROI 当前确认状态（兼容 main.py 的 detector.confirmed[i] 读取）。"""
        return [l.confirmed for l in self.lights]

    @property
    def in_slot_states(self):
        """每 ROI 最近一次在槽的确认状态（出槽reset后保留，供汇总展示）。

        当前在槽则等于 confirmed；已出槽则保留 reset 前的最后判定，避免汇总显示"待定"。
        """
        return [l.confirmed if l.state == "DETECTING" else l.last_in_slot
                for l in self.lights]

    def _fire(self, event, **kwargs):
        """触发回调（如果已注册）。"""
        cb = self._callbacks.get(event)
        if cb:
            cb(**kwargs)

    def slot_status(self, i):
        """返回第 i 个 ROI 的结构化状态，供外部查询/告警。"""
        light = self.lights[i]
        in_slot = light.state == "DETECTING"
        # 最近信号特征（取末尾 span_size 帧，供外部参考）
        signals = {}
        if light.signals:
            sig_list = list(light.signals)
            sig_arr = np.array(sig_list[-self.span_size:])
            signals["ge_median"] = float(np.median(sig_arr))
            if light.area_signals:
                area_arr = np.array(list(light.area_signals)[-self.span_size:])
                signals["area_median"] = float(np.median(area_arr))
            if light.gray_signals:
                gray_arr = np.array(list(light.gray_signals)[-self.span_size:])
                signals["gray_median"] = float(np.median(gray_arr))
        return {
            "name": light.roi.get("name", f"ROI{i}"),
            "in_slot": in_slot,
            "state": light.state,
            "light_status": light.confirmed if in_slot else light.last_in_slot,
            "last_in_slot_status": light.last_in_slot,
            "detecting_frames": light.detecting_frames,
            "stable_area": light.stable_area,
            "signals": signals,
        }

    def all_status(self):
        """返回所有 ROI 的结构化状态列表。"""
        return [self.slot_status(i) for i in range(self.n_lights)]

    # ------------------------------------------------------------------
    # ROI 取值
    # ------------------------------------------------------------------
    def _roi_lit_signal(self, frame, roi):
        """取 ROI 内"亮像素 green_excess 均值"、"亮像素面积"、"ROI 灰度均值"。

        green_excess = G - max(R, B)，逐像素计算，取 ge > lit_thr 的亮像素。
        返回 (亮像素 ge 均值, 亮像素面积, ROI 灰度均值)。无亮像素时 ge/面积为 0。
        灰度均值用于区分"不亮"(外壳绿,LED未亮,灰度~51)与"常亮"(LED点亮,灰度~110)。

        用亮像素均值而非全框均值：当灯体只占框的一小部分时（框比灯大），全框均值会把灯的
        高 ge 与大量背景的 0 平均掉（稀释成 ≈0，误判 off/unknown）；亮像素均值只取灯体亮像素，
        信号不被稀释。再配合面积门区分"反光"与"真灯"。green_excess 天然免疫环境光。
        """
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return 0.0, 0, 0.0
        gray = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        c = crop.astype(np.int16)
        ge = c[:, :, 1] - np.maximum(c[:, :, 0], c[:, :, 2])
        mask = ge > self.lit_thr
        n = int(mask.sum())
        if n <= 0:
            return 0.0, 0, gray
        return float(ge[mask].mean()), n, gray

    # ------------------------------------------------------------------
    # 入槽检测（每 ROI 独立）
    # ------------------------------------------------------------------
    def _check_enter_area_one(self, light, frame):
        """绿色面积法入槽检测（单 ROI，固定摄像头现场监控推荐）。

        原理（视频3实测验证）：
          - 设备入槽后，灯体（LED 点亮或仅绿色外壳）在 ROI 内带来稳定绿色面积（>100px），
            且真在槽时面积基本不归零（闪烁暗态面积仍>200，不亮外壳面积~450稳定）；
          - 机械臂移动经过空槽时，绿色面积会短暂出现又归零（进-出-进交替，每次持续<0.5s）；
          - 空槽时绿色面积恒为 0。
        故用"连续 enter_frames 帧 area>area_enter（不允许中间归零）"判入槽，可过滤
        机械臂经过的瞬态（持续<enter_frames 帧）。不依赖灯是否点亮，能覆盖"入槽不亮"。
        """
        roi = light.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            light.enter_counter = 0
            return False
        c = crop.astype(np.int16)
        ge = c[:, :, 1] - np.maximum(c[:, :, 0], c[:, :, 2])
        area = int((ge > self.lit_thr).sum())
        light.area_buf.append(area)
        # 连续超阈计数（归零即清零），连续 enter_frames 帧才确认入槽
        if area > self.area_enter:
            light.enter_counter += 1
        else:
            light.enter_counter = 0
        return light.enter_counter >= self.enter_frames
    def _check_enter_brightness_one(self, light, frame):
        """亮度法入槽检测（单 ROI）：ROI 灰度偏离"空槽基线"即认为设备进入。

        原理（实测验证，详见 README）：
          - 空槽时灯ROI是最暗的稳定状态（如灰度~74，std~0.7）；
          - 设备箱进入后该区域变亮（灰度~93-130），无论灯亮与否；
          - 灯亮/闪烁时灰度更高（~98-130）。
        故用滑动窗口的低百分位作为空槽基线（空槽=最暗态），当前灰度持续超过
        基线 + margin 即判入槽。该方法不依赖灯是否点亮，能覆盖"入槽出错灯不亮"
        的场景（只要设备箱进入了ROI就会变亮）。
        注意：基线用低百分位而非均值，避免被设备存在/灯亮的高灰度污染。
        """
        roi = light.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            light.last_baseline = None
            return False
        gray = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
        light.brightness_buf.append(gray)
        # 需要足够历史才估计基线
        if len(light.brightness_buf) < self.brightness_baseline_win // 2:
            light.last_baseline = None
            return False
        base = float(np.percentile(light.brightness_buf, self.brightness_percentile))
        light.last_baseline = base
        return gray > base + self.brightness_margin

    def _check_enter_bgdiff_one(self, light, gray_full):
        """背景差分法入槽检测（单 ROI，需视频开头为空槽作为背景）。

        每 ROI 独立背景模型。前 init_frames 帧累积初始背景，之后计算 ROI 区域
        当前帧与背景的绝对差均值，超过 enter_threshold 即判入槽。
        """
        if light.bg_model is None:
            light.bg_model = gray_full.astype(np.float32)
            light.bg_init_counter = 1
            return False
        if light.bg_init_counter < self.init_frames:
            cv2.accumulateWeighted(gray_full, light.bg_model, self.alpha)
            light.bg_init_counter += 1
            return False
        cv2.accumulateWeighted(gray_full, light.bg_model, self.alpha)
        bg = light.bg_model.astype(np.uint8)
        roi = light.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        cur = gray_full[y:y + h, x:x + w].astype(np.float32)
        bak = bg[y:y + h, x:x + w].astype(np.float32)
        if cur.size == 0:
            return False
        avg_diff = float(np.abs(cur - bak).mean())
        return avg_diff > self.enter_threshold

    # ------------------------------------------------------------------
    # 灯状态判定
    # ------------------------------------------------------------------
    def _classify(self, signal, area_seq=None, gray_seq=None):
        """对一段信号调用 classify_span。

        signal: 逐帧"亮像素 green_excess 均值"序列。
        area_seq: 逐帧亮像素面积序列（与 signal 同长）。传入则启用面积门区分反光/真灯。
        gray_seq: 逐帧 ROI 灰度序列（与 signal 同长）。传入则在"真灯+不振荡"分支区分不亮/常亮。
        on_thr 固定为 None（用固定 on_threshold）；面积门+灰度门模式下 on_threshold 不参与判定。
        """
        return classify_span(
            signal,
            on_threshold=self.on_threshold,
            osc_min=self.osc_min,
            min_sep=self.min_sep,
            smooth_kernel=self.smooth_kernel,
            fps=self.fps,
            lit_area_seq=area_seq,
            area_min=self.area_min,
            lit_ge=self.lit_ge,
            lit_ratio_min=self.lit_ratio_min,
            on_thr=None,
            gray_seq=gray_seq,
            gray_thr=self.gray_thr,
        )

    def _vote_and_confirm_one(self, light, roi_index=0, force_confirm=False):
        """单 ROI 跨度判定 + 连续 confirm_spans 一致才更新 confirmed。

        返回 (span_result, did_change)：
          span_result: 本轮跨度判定 dict（含 status/median/osc 等），无判定时为 None
          did_change: 本轮 confirmed 是否变化
        force_confirm: True 时跳过"连续 confirm_spans 一致"要求，单次跨度即确认
                      （用于出槽前紧急判定：在槽时间不足攒满 confirm_spans）。
        roi_index: 用于回调的槽索引。
        """
        sig = light.signals
        if len(sig) < self.span_size:
            return None, False
        seq = list(sig)[-self.span_size:]
        area_seq = None
        if len(light.area_signals) >= self.span_size:
            area_seq = list(light.area_signals)[-self.span_size:]
        gray_seq = None
        if len(light.gray_signals) >= self.span_size:
            gray_seq = list(light.gray_signals)[-self.span_size:]
        res = self._classify(seq, area_seq=area_seq, gray_seq=gray_seq)
        res["on_thr"] = self.on_threshold
        light.recent_spans.append(res["status"])
        # 连续 confirm_spans 个跨度结果一致才确认（force_confirm 时单次即确认）
        recent = list(light.recent_spans)
        did_change = False
        if force_confirm or (len(recent) >= self.confirm_spans
                             and len(set(recent[-self.confirm_spans:])) == 1):
            new_status = recent[-1]
            old_status = light.confirmed
            if new_status != old_status:
                light.confirmed = new_status
                light.last_in_slot = new_status  # 记录在槽确认状态
                # 回调：灯状态变化
                self._fire("on_status", roi_index=roi_index,
                           roi_name=light.roi.get("name", f"ROI{roi_index}"),
                           old_status=old_status, new_status=new_status,
                           frame_idx=self.frame_idx, details=res)
                did_change = True
        return res, did_change

    def _check_leave_one(self, light, frame):
        """单 ROI 设备被取走判定，回 WAITING。

        按 enter_method 选择判据：
          - "area"/默认：绿色面积法。设备取走时灯体离开 ROI，绿色面积从稳定高位下降。
            判据：窗口超阈占比≤leave_ratio（归零），或当前面积<stable_area*leave_drop_ratio。
            不依赖灯亮（外壳绿也有面积），不会把"在槽灯不亮"误判取走。
          - "brightness"：灰度法（旧，手持测试视频）。灰度回落到入槽时冻结的空槽基线附近。
            设备在槽但灯未亮时灰度仍高于基线（设备存在即变亮），不会误判取走。
        """
        if not self.leave_check:
            return False
        roi = light.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return False

        if self.enter_method == "brightness":
            # 灰度法出槽：回落到入槽时冻结的空槽基线
            base = light.empty_baseline
            if base is None:
                return False
            gray = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
            if gray > base + self.brightness_margin:
                light.leave_counter = 0
                return False
            light.leave_counter += 1
            return light.leave_counter >= self.enter_frames

        # 面积法出槽（默认）
        c = crop.astype(np.int16)
        ge = c[:, :, 1] - np.maximum(c[:, :, 0], c[:, :, 2])
        area = int((ge > self.lit_thr).sum())
        light.area_buf.append(area)
        if len(light.area_buf) < self.enter_win:
            return False
        ratio = sum(1 for a in light.area_buf if a > self.area_enter) / len(light.area_buf)
        # 面积基本归零 → 出槽
        if ratio <= self.leave_ratio:
            return True
        # 面积显著低于稳定值（设备正在离开，尚未完全归零）→ 出槽
        if light.stable_area > self.area_enter * 2 and area < light.stable_area * self.leave_drop_ratio:
            return True
        return False

    # ------------------------------------------------------------------
    # 主处理入口
    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """处理一帧，返回 (states, info)。

        states: 每个 ROI 的状态 list（"WAITING"/"DETECTING"），长度 = n_lights。
        info 为 dict，包含：
          states: 同上
          confirmed: 每个灯当前确认状态 list
          span_results: 本轮跨度判定（可能为 None 列表）
          changed: 本轮状态变化的灯索引 list
        """
        self.frame_idx += 1
        gray_full = None
        if self.enter_method == "bg_diff":
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        states = []
        span_results = [None] * self.n_lights
        changed = []
        info = {"states": states, "confirmed": self.confirmed,
                "span_results": span_results, "changed": changed}

        for i, light in enumerate(self.lights):
            states.append(self._step_one(i, light, frame, gray_full, span_results, changed))
        return states, info

    def _step_one(self, i, light, frame, gray_full, span_results, changed):
        """单 ROI 状态机推进，返回该 ROI 状态。"""
        if light.state == "WAITING":
            if self.skip_enter:
                # 跳过入槽检测：直接进入 DETECTING
                light.enter_detecting(self.span_size)
                self._fire("on_enter", roi_index=i, roi_name=light.roi.get("name", f"ROI{i}"),
                           frame_idx=self.frame_idx)
                self._do_detecting(i, light, frame, span_results, changed)
                return light.state

            if self.enter_method == "bg_diff":
                entered = self._check_enter_bgdiff_one(light, gray_full)
            elif self.enter_method == "brightness":
                entered = self._check_enter_brightness_one(light, frame)
            else:  # "area"：面积法内部自维护 enter_counter（连续超阈计数），外层不再干预
                if self._check_enter_area_one(light, frame):
                    light.enter_detecting(self.span_size)
                    self._fire("on_enter", roi_index=i, roi_name=light.roi.get("name", f"ROI{i}"),
                               frame_idx=self.frame_idx)
                    self._do_detecting(i, light, frame, span_results, changed)
                return light.state

            if entered:
                light.enter_counter += 1
                if light.enter_counter >= self.enter_frames:
                    light.enter_detecting(self.span_size)
                    self._fire("on_enter", roi_index=i, roi_name=light.roi.get("name", f"ROI{i}"),
                               frame_idx=self.frame_idx)
                    self._do_detecting(i, light, frame, span_results, changed)
            else:
                light.enter_counter = 0
            return light.state

        # DETECTING
        self._do_detecting(i, light, frame, span_results, changed)
        return light.state

    def _do_detecting(self, i, light, frame, span_results, changed):
        """DETECTING 状态：提取信号、滑动跨度判定、投票确认。"""
        # 1) 提取该 ROI 的信号（亮像素 green_excess 均值）+ 亮像素面积 + ROI 灰度
        sig, area, gray = self._roi_lit_signal(frame, light.roi)
        light.signals.append(sig)
        light.area_signals.append(area)
        light.gray_signals.append(gray)

        light.detecting_frames += 1

        # 维护入槽稳定面积参考（取近期面积中位数，供出槽判定比较）。
        # 用近 span_size 帧中位数代表"在槽稳定态"，出槽时面积会显著低于此值。
        if len(light.area_signals) >= 30:
            light.stable_area = float(np.median(list(light.area_signals)[-self.span_size:]))

        # 1) 设备取走检测（每帧检查，每 ROI 独立）—— 不绑定 span 判定，确保短时在槽也能 reset
        if self._check_leave_one(light, frame):
            # 出槽前紧急判定：在槽时间不足 span_size 攒满投票时，用已有帧(≥60)补一次判定。
            # 仅在信号明确时确认（osc≥osc_min 判闪烁，或面积门判 off），避免短跨度误判常亮。
            if light.last_in_slot == "pending" and light.detecting_frames >= 60:
                seq = list(light.signals)
                area_seq = list(light.area_signals) if light.area_signals else None
                gray_seq = list(light.gray_signals) if light.gray_signals else None
                res = self._classify(seq, area_seq=area_seq, gray_seq=gray_seq)
                res["on_thr"] = self.on_threshold
                st = res["status"]
                # 明确闪烁(osc达标) 或 明确不亮(面积/灰度门) 才确认；steady_on 不确认(短跨度易误判)
                if st == "flashing" or (st == "off" and res.get("area_med", 0) < self.area_min):
                    light.confirmed = st
                    light.last_in_slot = st
                    changed.append(i)
                span_results[i] = res
            self._fire("on_leave", roi_index=i, roi_name=light.roi.get("name", f"ROI{i}"),
                       frame_idx=self.frame_idx, last_status=light.last_in_slot)
            light.reset_to_waiting()
            return

        # 2) 到达判定时刻则做跨度判定
        if light.detecting_frames >= light.next_span_at:
            res, did_change = self._vote_and_confirm_one(light, i)
            span_results[i] = res
            if did_change:
                changed.append(i)
            light.next_span_at += self.span_step
