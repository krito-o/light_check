"""灯状态分类逻辑（基于"绿色超出量 + 振荡次数"的稳健时域判定）。

将绿灯的逐帧状态判定为三种之一：
  - 'flashing'  : 闪烁（绿灯亮且在高/低亮度间反复跳变；正常工作）
  - 'steady_on' : 常亮（绿灯持续亮、几乎不跳变；故障）
  - 'off'       : 不亮（绿灯未发光，仅背景；故障）

设计说明（基于实测数据，详见项目计划与 README）：
  1. 信号选择 —— 用"绿色超出量" green_excess = G - max(R, B)，而非 G 通道绝对亮度。
     原因：实测"不亮"故障灯处并非全黑，而是被环境光照亮的灰色背景（B≈G≈R≈120），
     其 G 通道绝对值(109)甚至高于某些闪烁灯的暗端。但绿灯发光时 G 明显大于 R、B
     (green_excess≈9-16)，而灰色背景 green_excess≈0。因此 green_excess 是判别
     "灯是否发光"的可靠特征，且天然对环境光（三通道同步变化）免疫——无需再做
     roi-ref 相减，规避了参考区噪声放大问题。
  2. 闪烁 ≠ 周期频率。实测正常绿灯为不规则的开/关跳变，FFT 主频能量极低，故放弃
     FFT 频率匹配，改用"振荡次数"判据。
  3. 离群点鲁棒。实测 off 灯在视频中段有反射造成的亮度突起，用 max/min 峰峰值会被
     污染，故一律用稳健统计量：中位数、25/75 百分位、MAD。
  4. 振荡判据。用 25/75 百分位差作为振幅门限（小于门限视为无波动/噪声，跳变数置 0），
     再以 (p25+p75)/2 为分界统计平滑后序列的穿越次数。steady/off/突起 → 0~1 次跳变；
     flashing → 多次跳变。
  5. 单短窗口可能恰好落在闪烁某段电平内，故最终判定应在较长跨度上做（classify_span），
    并由调用方做多跨度投票（见 SlotDetector）抑制瞬态。

核心判据（classify_span，信号 = 亮像素 green_excess 均值 + 亮像素面积 + ROI灰度）：
    not is_real_light(面积不够)        -> off       反光/空槽/无灯体
    is_real_light 且 振荡(osc>=osc_min) -> flashing  真灯闪烁
    is_real_light 且 不振荡 且 灰度高    -> steady_on LED点亮常亮
    is_real_light 且 不振荡 且 灰度低    -> off       灯体外壳绿但LED未点亮(不亮故障)

  "不亮"与"常亮"的区分（视频3实测发现）：
    两者 green_excess 亮像素均值几乎相同（不亮~31，常亮~34，因 max(R,B) 抹平了LED点亮
    带来的 G 通道提升），无法用 ge 区分。但 LED 点亮会显著照亮整个 ROI：不亮灰度~51，
    常亮灰度~110，差 2 倍。故不振荡时用 ROI 灰度均值区分（gray_med >= gray_thr 算LED点亮）。
    振荡分支不看灰度（闪烁亮暗态灰度跨度大，中位数无意义）。

本模块为纯函数，便于独立单元测试（见文件末尾 __main__）。
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import median_filter as _median_filter
    _HAS_SCIPY = True
except Exception:  # scipy 非必需，缺失时退化为无滤波
    _HAS_SCIPY = False


def _smooth(seq: np.ndarray, kernel: int) -> np.ndarray:
    """中值滤波平滑（抑制逐帧噪声）。scipy 不可用时原样返回。"""
    if not _HAS_SCIPY or kernel <= 1 or len(seq) < kernel:
        return seq
    return _median_filter(seq, size=kernel)


def green_excess(b_mean, g_mean, r_mean) -> np.ndarray:
    """由逐帧 B/G/R 通道均值计算"绿色超出量"信号 = G - max(R, B)。

    输入可为标量或数组（同形）。绿灯发光时该值显著为正；灰色背景/环境光时≈0。
    """
    b = np.asarray(b_mean, dtype=np.float64)
    g = np.asarray(g_mean, dtype=np.float64)
    r = np.asarray(r_mean, dtype=np.float64)
    return g - np.maximum(r, b)


def _oscillation_count(seq: np.ndarray, min_sep: float = 4.0) -> int:
    """统计序列在两个亮度级之间的"有效跳变"次数。

    方法：用 25/75 百分位刻画序列上下亮度级，其差值作为振幅；若振幅不足 min_sep
    （说明几乎无波动或仅噪声），直接返回 0。否则以 (p25+p75)/2 为分界，统计平滑后
    序列穿越分界的次数。这样：
      - 噪声底噪（steady/off）振幅小 → 0 跳变；
      - 单次瞬态突起（burst）穿越分界只有 1 次；
      - 真正的闪烁振幅大且反复穿越 → 多次跳变。
    """
    if len(seq) < 4:
        return 0
    p25, p75 = np.percentile(seq, [25, 75])
    if (p75 - p25) < min_sep:
        return 0
    mid = (p25 + p75) / 2.0
    above = seq > mid
    if len(above) < 2:
        return 0
    return int(np.sum(np.diff(above.astype(np.int8)) != 0))


def dominant_freq(seq: np.ndarray, fps: float) -> float:
    """计算窗口去直流后的 FFT 主频（Hz），仅用于日志/参考，不参与判定。"""
    seq = np.asarray(seq, dtype=np.float64)
    if len(seq) < 8:
        return 0.0
    centered = seq - seq.mean()
    if np.allclose(centered, 0):
        return 0.0
    mag = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(seq), d=1.0 / fps)
    if len(mag) <= 1:
        return 0.0
    idx = int(np.argmax(mag[1:])) + 1  # 忽略直流
    return float(freqs[idx])


def classify_window(
    seq,
    on_threshold: float = 5.0,
    osc_min: int = 3,
    min_sep: float = 4.0,
    smooth_kernel: int = 5,
    fps: float = 50.0,
    lit_area_seq=None,
    area_min: float = 150.0,
    lit_ge: float = 5.0,
    lit_ratio_min: float = 0.15,
    on_thr=None,
    gray_seq=None,
    gray_thr: float = 80.0,
) -> dict:
    """对单个短窗口的亮度信号做局部特征分类（用于逐窗日志/可视化）。

    参数与判定流程与 classify_span 一致（详见 classify_span 文档），
    仅 osc_min 默认值更小（短窗口用较小值）。lit_area_seq/gray_seq 等新参数可选，
    不传则退化为原逻辑，兼容旧调用。

    返回 dict: {status, median, mad, osc, pp, freq, lit_ratio, area_med, gray_med}
    status ∈ {'flashing', 'steady_on', 'off', 'unknown'}
    """
    seq = np.asarray(seq, dtype=np.float64)
    n = len(seq)
    if n == 0:
        return {"status": "unknown", "median": 0.0, "mad": 0.0,
                "osc": 0, "pp": 0.0, "freq": 0.0, "lit_ratio": 0.0,
                "area_med": 0.0, "gray_med": 0.0}

    med = float(np.median(seq))
    mad = float(np.median(np.abs(seq - med)) * 1.4826)  # 稳健 std 估计
    pp = float(seq.max() - seq.min())
    smooth = _smooth(seq, smooth_kernel)
    osc = _oscillation_count(smooth, min_sep)
    freq = dominant_freq(seq, fps)
    lit_ratio = float(np.mean(seq >= lit_ge)) if n > 0 else 0.0

    has_area = lit_area_seq is not None and len(lit_area_seq) > 0
    if has_area:
        area_arr = np.asarray(lit_area_seq, dtype=np.float64)
        area_seg = area_arr[-n:] if len(area_arr) >= n else area_arr
        area_med = float(np.median(area_seg)) if len(area_seg) > 0 else 0.0
        is_real_light = area_med > area_min
    else:
        area_med = 0.0
        is_real_light = True

    # ROI 灰度（区分"不亮"vs"常亮"：LED 点亮会显著照亮 ROI）
    has_gray = gray_seq is not None and len(gray_seq) > 0
    if has_gray:
        gray_arr = np.asarray(gray_seq, dtype=np.float64)
        gray_seg = gray_arr[-n:] if len(gray_arr) >= n else gray_arr
        gray_med = float(np.median(gray_seg)) if len(gray_seg) > 0 else 0.0
    else:
        gray_med = 0.0

    thr = on_threshold if on_thr is None else on_thr
    is_oscillating = osc >= osc_min

    if has_area:
        if not is_real_light:
            status = "off"
        elif is_oscillating:
            status = "flashing"
        elif has_gray:
            # 真灯体 + 不振荡：用灰度区分 LED 点亮(常亮) vs 仅外壳绿(不亮)
            status = "steady_on" if gray_med >= gray_thr else "off"
        else:
            status = "steady_on"
    else:
        if med < thr:
            status = "off" if osc < osc_min else "unknown"
        else:
            status = "flashing" if osc >= osc_min else "steady_on"

    return {
        "status": status,
        "median": med,
        "mad": mad,
        "osc": osc,
        "pp": pp,
        "freq": freq,
        "lit_ratio": lit_ratio,
        "area_med": area_med,
        "gray_med": gray_med,
    }


def classify_span(
    seq,
    on_threshold: float = 5.0,
    osc_min: int = 6,
    min_sep: float = 4.0,
    smooth_kernel: int = 5,
    fps: float = 50.0,
    lit_area_seq=None,
    area_min: float = 150.0,
    lit_ge: float = 5.0,
    lit_ratio_min: float = 0.15,
    on_thr=None,
    gray_seq=None,
    gray_thr: float = 80.0,
) -> dict:
    """在较长的观察跨度上做最终状态判定（推荐作为最终结果）。

    参数:
        seq: 观察跨度内逐帧"亮像素 green_excess 均值"信号序列（建议 >= 100 帧）。
            正式代码中 seq 为 ROI 内"亮像素 green_excess 均值"（ge>lit_thr 的亮像素均值，
            无亮像素则 0），可避免"框比灯大"时全框均值把灯信号稀释成 0。
        on_threshold: 信号中位数低于此值视为"灯不亮"的固定阈值（ref 缺失时用）。
        osc_min: 有效跳变次数阈值。实测闪烁跨度跳变≈5-29，稳态/不亮/突起≈0-1，取 6。
        min_sep: 振幅门限。green_excess 序列 25/75 百分位差小于此值视为无有效波动。
        smooth_kernel: 中值滤波核大小，抑制逐帧噪声造成的虚假跳变。
        fps: 仅用于计算日志中的 FFT 主频。
        lit_area_seq: 逐帧"亮像素面积"序列（与 seq 同长，可选）。若传入则启用面积门——
            跨度内面积中位数 <= area_min 视为"非真灯"（小面积反光/空槽）直接判 off。
            这能区分"框内局部绿色反光"(面积小,如60px)与"真灯亮起"(面积大,如400+px)。
            不传(默认 None)则退化为不使用面积门，兼容旧调用。
        area_min: 真灯面积门(像素)。反光通常<100，真灯通常>400，取 150 可区分。
        lit_ge: lit_ratio 计算用"亮帧"阈值，信号>=此值视为该帧灯亮。
        lit_ratio_min: 亮帧占比兜底阈值。稀释场景 median 偏低时，亮帧占比达标仍判灯亮。
        on_thr: 调用方算好的自适应阈值(由 ref 区绿色本底 + on_margin 得到)。None 则用 on_threshold。
        gray_seq: 逐帧 ROI 灰度均值序列（与 seq 同长，可选）。用于区分"不亮"与"常亮"——
            两者 ge 接近(不亮~31,常亮~34)但 LED 点亮使常亮灰度(~110)远高于不亮(~51)。
            仅在"真灯体+不振荡"分支生效：灰度>=gray_thr 判常亮，否则判不亮(off)。
        gray_thr: LED 点亮的灰度阈值。实测不亮灰度~51、常亮~110，取 80 可区分。

    判定流程（面积门 + 振荡门 + 灰度门）:
        not is_real_light(面积不够)        -> off       反光/空槽/无灯体
        is_real_light 且 振荡(osc>=osc_min) -> flashing  真灯闪烁
        is_real_light 且 不振荡 且 灰度高    -> steady_on LED点亮常亮
        is_real_light 且 不振荡 且 灰度低    -> off       外壳绿但LED未亮(不亮故障)
    其中 is_real_light 由面积门(若传入)决定；亮度门(median/lit_ratio)在退化模式(无面积门)下
    仍用于区分 off/unknown，保持旧行为。

    返回 dict: {status, median, mad, osc, pp, freq, lit_ratio, area_med, gray_med}
    status ∈ {'flashing', 'steady_on', 'off', 'unknown'}
    """
    seq = np.asarray(seq, dtype=np.float64)
    n = len(seq)
    if n == 0:
        return {"status": "unknown", "median": 0.0, "mad": 0.0,
                "osc": 0, "pp": 0.0, "freq": 0.0, "lit_ratio": 0.0,
                "area_med": 0.0, "gray_med": 0.0}

    med = float(np.median(seq))
    mad = float(np.median(np.abs(seq - med)) * 1.4826)
    pp = float(seq.max() - seq.min())
    smooth = _smooth(seq, smooth_kernel)
    osc = _oscillation_count(smooth, min_sep)
    freq = dominant_freq(seq, fps)
    lit_ratio = float(np.mean(seq >= lit_ge)) if n > 0 else 0.0

    # 面积门（可选）：传入面积序列才启用
    has_area = lit_area_seq is not None and len(lit_area_seq) > 0
    if has_area:
        area_arr = np.asarray(lit_area_seq, dtype=np.float64)
        area_seg = area_arr[-n:] if len(area_arr) >= n else area_arr
        area_med = float(np.median(area_seg)) if len(area_seg) > 0 else 0.0
        is_real_light = area_med > area_min
    else:
        area_med = 0.0
        is_real_light = True  # 不传面积 → 退化，不靠面积门

    # 灰度门（可选）：传入灰度序列才启用，区分"不亮"与"常亮"
    has_gray = gray_seq is not None and len(gray_seq) > 0
    if has_gray:
        gray_arr = np.asarray(gray_seq, dtype=np.float64)
        gray_seg = gray_arr[-n:] if len(gray_arr) >= n else gray_arr
        gray_med = float(np.median(gray_seg)) if len(gray_seg) > 0 else 0.0
    else:
        gray_med = 0.0

    thr = on_threshold if on_thr is None else on_thr
    is_lit = (med >= thr) or (lit_ratio >= lit_ratio_min)
    is_oscillating = osc >= osc_min

    if has_area:
        # 新模式：面积门 + 振荡门 + 灰度门
        if not is_real_light:
            status = "off"            # 反光/空槽：面积不够
        elif is_oscillating:
            status = "flashing"       # 真灯 + 振荡
        elif has_gray:
            # 真灯 + 不振荡：灰度区分 LED 点亮(常亮) vs 仅外壳绿(不亮)
            status = "steady_on" if gray_med >= gray_thr else "off"
        else:
            status = "steady_on"      # 真灯 + 不振荡 + 无灰度（旧调用兼容）
    else:
        # 退化模式（旧调用 / 旧 config 无面积）：保持原逻辑
        if med < thr:
            status = "off" if osc < osc_min else "unknown"
        else:
            status = "flashing" if osc >= osc_min else "steady_on"

    return {
        "status": status,
        "median": med,
        "mad": mad,
        "osc": osc,
        "pp": pp,
        "freq": freq,
        "lit_ratio": lit_ratio,
        "area_med": area_med,
        "gray_med": gray_med,
    }


# ---------------------------------------------------------------------------
# 单元测试（合成序列回归基线）：python light_classifier.py
# ---------------------------------------------------------------------------
def _self_test():
    np.random.seed(0)
    failures = []

    def check(name, cond):
        ok = bool(cond)
        if not ok:
            failures.append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    # --- classify_window ---
    # 常亮：green_excess 持续高、不振荡
    steady = np.full(40, 14.0) + np.random.normal(0, 1, 40)
    check("window steady_on", classify_window(steady)["status"] == "steady_on")

    # 不亮：green_excess≈0（灰色背景）、不振荡
    off = np.full(40, 0.0) + np.random.normal(0, 1, 40)
    check("window off", classify_window(off)["status"] == "off")

    # 闪烁：高/低两级跳变
    flash = np.concatenate([
        np.full(20, 2.0) + np.random.normal(0, 1, 20),
        np.full(20, 16.0) + np.random.normal(0, 1, 20),
    ])
    np.random.shuffle(flash)
    check("window flashing", classify_window(flash)["status"] == "flashing")

    # 渐变：不应判为闪烁
    ramp = np.linspace(0, 14, 40)
    check("window ramp not flashing", classify_window(ramp)["status"] != "flashing")

    # --- classify_span (最终判定) ---
    span_steady = np.full(150, 14.0) + np.random.normal(0, 1, 150)
    check("span steady_on", classify_span(span_steady)["status"] == "steady_on")

    span_off = np.full(150, 0.0) + np.random.normal(0, 1, 150)
    check("span off", classify_span(span_off)["status"] == "off")

    # 慢闪烁：大部分高电平(14)，偶尔跌到 0（4帧宽，能穿过中值滤波）
    span_slow = np.full(150, 14.0) + np.random.normal(0, 1, 150)
    for i in range(0, 150, 15):
        span_slow[i:i + 4] = 0.0
    check("span slow flashing", classify_span(span_slow)["status"] == "flashing")

    # 高电平跳变闪烁：2/16 两级
    span_high = np.concatenate([np.full(75, 2.0), np.full(75, 16.0)])
    np.random.shuffle(span_high)
    check("span high-level flashing", classify_span(span_high)["status"] == "flashing")

    # 渐变（环境光导致 green_excess 缓慢漂移）：不应判为闪烁
    span_ramp = np.linspace(0, 14, 150)
    check("span ramp not flashing", classify_span(span_ramp)["status"] != "flashing")

    # off 灯带短暂突起（反射）：整体 green_excess 偏低且不振荡 → 仍 off
    span_burst = np.full(150, 0.0) + np.random.normal(0, 1, 150)
    span_burst[70:80] = 14.0  # 短暂反射突起
    check("span off with burst stays off", classify_span(span_burst)["status"] == "off")

    # --- 面积门（lit_area_seq）---
    # 稀释闪烁：用"亮像素均值"信号(亮态ge≈35,暗态0,振荡明显)；面积大(真灯)→flashing
    # 模拟真实新视频: 固定框下全框均值被稀释, 但亮像素均值信号仍保留振荡
    dilute_flash = np.concatenate([np.full(30, 35.0), np.full(30, 0.0)] * 3)
    np.random.shuffle(dilute_flash)
    # 面积序列：亮帧400px(真灯)，暗帧0
    area_real = np.where(dilute_flash > 15, 400.0, 0.0)
    r_df = classify_span(dilute_flash, lit_area_seq=area_real, area_min=150.0)
    check("span dilute flashing (area gate)", r_df["status"] == "flashing")
    # 该信号即使不带面积门也应判 flashing(因 median 高且振荡)——面积门主要救"反光"场景
    check("span dilute flashing without area gate",
          classify_span(dilute_flash)["status"] == "flashing")

    # 反光：信号持续偏高(ge≈18)但不振荡；面积小(60px反光)→off（面积门挡住）
    reflect = np.full(150, 18.0) + np.random.normal(0, 1, 150)
    area_reflect = np.full(150, 60.0)  # 小面积反光
    r_rf = classify_span(reflect, lit_area_seq=area_reflect, area_min=150.0)
    check("span reflection -> off (area gate)", r_rf["status"] == "off")
    # 同样信号不带面积门(退化)会被判 steady_on(median高不振荡)——证明面积门是区分反光的关键
    check("span reflection no-area degrades to steady_on",
          classify_span(reflect)["status"] == "steady_on")
    # 同样信号但面积大(真灯)且不振荡 → steady_on
    area_big = np.full(150, 700.0)
    r_st = classify_span(reflect, lit_area_seq=area_big, area_min=150.0)
    check("span real steady_on (area gate)", r_st["status"] == "steady_on")

    # --- 灰度门（区分"不亮"vs"常亮"）---
    # 两者 ge 接近(亮像素均值~31)、面积都大(真灯体)、都不振荡：
    #   不亮: ROI灰度~51 (LED未点亮,仅外壳绿)
    #   常亮: ROI灰度~110 (LED点亮照亮ROI)
    shell_ge = np.full(150, 31.0) + np.random.normal(0, 1, 150)   # 外壳绿(不亮)
    led_ge = np.full(150, 34.0) + np.random.normal(0, 1, 150)     # LED点亮(常亮)
    area_real = np.full(150, 450.0)
    gray_off = np.full(150, 51.0)    # 不亮灰度
    gray_on = np.full(150, 110.0)    # 常亮灰度
    r_off = classify_span(shell_ge, lit_area_seq=area_real, area_min=150.0,
                          gray_seq=gray_off, gray_thr=80.0)
    check("span shell-green low-gray -> off (gray gate)", r_off["status"] == "off")
    r_on = classify_span(led_ge, lit_area_seq=area_real, area_min=150.0,
                         gray_seq=gray_on, gray_thr=80.0)
    check("span led-on high-gray -> steady_on (gray gate)", r_on["status"] == "steady_on")
    # 不传灰度 → 退化为 steady_on（旧行为，向后兼容）
    r_nogray = classify_span(shell_ge, lit_area_seq=area_real, area_min=150.0)
    check("span no-gray degrades to steady_on", r_nogray["status"] == "steady_on")

    # --- green_excess 辅助函数 ---
    # 第0点：G=130 明显偏绿 → 30；第1点：G=100 ≤ R=105 → -5（非绿）
    ge = green_excess(np.array([100, 100]), np.array([130, 100]), np.array([100, 105]))
    check("green_excess on/off", list(ge) == [30, -5])

    print(f"\n{'ALL PASSED' if not failures else 'FAILURES: ' + str(failures)}")
    return len(failures) == 0


if __name__ == "__main__":
    import sys
    ok = _self_test()
    sys.exit(0 if ok else 1)
