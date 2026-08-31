"""可视化叠加：在帧上绘制 ROI 框 + 状态文字（支持中文）。

独立于 SlotDetector 核心算法，部署时无需引入此模块。
"""

from __future__ import annotations

import os

import cv2
import numpy as np

# 中文绘制：cv2.putText 不支持中文（显示为 ?），用 PIL 渲染中文文本。
try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

# 加载中文字体（微软雅黑优先，回退到黑体/宋体），找不到则 _ZH_FONT=None 退化为英文
_ZH_FONT = None
if _PIL_OK:
    for _fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\simsun.ttc", "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"]:
        if os.path.exists(_fp):
            _ZH_FONT = _fp
            break


# 状态 -> 中文描述（用于可视化展示）
STATE_NAMES = {"WAITING": "等待入槽", "DETECTING": "检测中"}

# 灯状态 -> 中文（用于标注视频显示）
STATUS_CN = {
    "flashing": "闪烁", "off": "不亮", "steady_on": "常亮",
    "unknown": "未知", "pending": "待定",
}

# 灯状态 -> 颜色 (B,G,R) 用于可视化
# 规范：闪烁(正常)=绿，不亮(故障)=黄，常亮(故障)=红，检测中/待定=灰
STATUS_COLORS = {
    "flashing": (0, 200, 0),      # 绿 - 正常闪烁
    "off": (0, 230, 230),         # 黄 - 灯不亮(故障)
    "steady_on": (0, 0, 230),     # 红 - 灯常亮(故障)
    "unknown": (160, 160, 160),   # 灰 - 检测中/未定
    "pending": (160, 160, 160),   # 灰 - 尚未判定
}


def _put_text(img, text, org, color, scale=0.6, thickness=2):
    """在 OpenCV 图像上绘制文本，支持中文。

    有 PIL+中文字体时用 PIL 渲染（中文正常显示）；否则退化为 cv2.putText（仅 ASCII）。
    org 为文本左下角坐标 (x, y)，与 cv2.putText 一致。
    color 为 BGR 元组。
    """
    if not _PIL_OK or _ZH_FONT is None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
        return img
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(_ZH_FONT, max(int(scale * 32), 12))
    x, y = org
    try:
        ascent, _ = font.getmetrics()
        top = y - ascent
    except Exception:
        top = y - int(scale * 30)
    r, g, b = int(color[2]), int(color[1]), int(color[0])  # BGR -> RGB
    draw.text((x, top), text, font=font, fill=(r, g, b),
              stroke_width=max(thickness - 1, 1), stroke_fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _text_size(text, scale=0.6, thickness=2):
    """估算文本宽高 (w, h)，用于标签背景矩形。"""
    if _PIL_OK and _ZH_FONT is not None:
        font = ImageFont.truetype(_ZH_FONT, max(int(scale * 32), 12))
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            pass
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return tw, th


def draw_overlay(detector, frame):
    """在帧上绘制 ROI 框（按确认状态着色）+ 状态文字。返回新帧。

    参数：
        detector: SlotDetector 实例（通过 detector.lights 读取状态）。
        frame: 原始帧（不会被修改）。

    文本用 PIL 渲染以支持中文；无 PIL/中文字体时退化为英文 cv2.putText。
    所有文本在单次 PIL 会话中绘制，避免多次全图格式转换。
    """
    out = frame.copy()
    if not _PIL_OK or _ZH_FONT is None:
        return _draw_overlay_ascii(detector, out)

    # 先用 cv2 画矩形（快），再转 PIL 画中文文本（一次转换）
    rects = []  # 记录每个文本的绘制参数，统一在 PIL 画
    for i, light in enumerate(detector.lights):
        roi = light.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        # 展示语义：设备在槽=已入槽(显示灯状态)，不在槽=已出槽(保留最近灯状态)
        if light.state == "DETECTING":
            status = light.confirmed
            tag = "已入槽"
        else:
            status = light.last_in_slot
            tag = "已出槽"
        color = STATUS_COLORS.get(status, STATUS_COLORS["pending"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        name = roi.get("name", f"ROI{i}")
        status_cn = STATUS_CN.get(status, status)
        label = f"{name} {status_cn}({tag})"
        tw, th = _text_size(label, 0.6, 2)
        ty = max(y - 6, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 8, ty + 2), (0, 0, 0), -1)
        rects.append((label, (x + 2, ty), color, 0.6, 2))
        if "expected" in roi:
            exp_cn = STATUS_CN.get(roi["expected"], roi["expected"])
            rects.append((f"期望:{exp_cn}", (x, y + h + 18), (200, 200, 200), 0.5, 1))

    # 顶部状态条：每槽 已入槽/已出槽
    parts = []
    for i, light in enumerate(detector.lights):
        name = light.roi.get("name", f"ROI{i}")
        tag = "已入槽" if light.state == "DETECTING" else "已出槽"
        parts.append(f"{name}:{tag}")
    bar = f"帧:{detector.frame_idx}  " + "  ".join(parts)
    bw, _ = _text_size(bar, 0.6, 1)
    cv2.rectangle(out, (0, 0), (max(bw + 16, 420), 30), (0, 0, 0), -1)
    rects.append((bar, (8, 22), (255, 255, 255), 0.6, 1))

    # 一次 PIL 转换，画所有中文文本
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    fonts = {}
    for text, (x, y), color, scale, thickness in rects:
        key = scale
        if key not in fonts:
            fonts[key] = ImageFont.truetype(_ZH_FONT, max(int(scale * 32), 12))
        font = fonts[key]
        try:
            ascent, _ = font.getmetrics()
            top = y - ascent
        except Exception:
            top = y - int(scale * 30)
        r, g, b = int(color[2]), int(color[1]), int(color[0])  # BGR -> RGB
        draw.text((x, top), text, font=font, fill=(r, g, b),
                  stroke_width=max(thickness - 1, 1), stroke_fill=(r, g, b))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _draw_overlay_ascii(detector, out):
    """无 PIL 时的退化绘制：英文标签 + cv2.putText。"""
    for i, light in enumerate(detector.lights):
        roi = light.roi
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        if light.state == "DETECTING":
            status = light.confirmed
            tag = "in"
        else:
            status = light.last_in_slot
            tag = "out"
        color = STATUS_COLORS.get(status, STATUS_COLORS["pending"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        name = roi.get("name", f"ROI{i}")
        label = f"{name}:{status}({tag})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y - 6, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if "expected" in roi:
            cv2.putText(out, f"exp:{roi['expected']}", (x, y + h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    parts = [f"{light.roi.get('name', f'ROI{i}')}:{'in' if light.state=='DETECTING' else 'out'}"
             for i, light in enumerate(detector.lights)]
    bar = f"frame:{detector.frame_idx}  " + "  ".join(parts)
    (bw, _), _ = cv2.getTextSize(bar, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (0, 0), (max(bw + 16, 420), 30), (0, 0, 0), -1)
    cv2.putText(out, bar, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out