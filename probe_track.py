"""临时探针：用灯体跟踪为手持起落视频划定"建议检测区域"并做可视化展示。

用途：
  这批手持"起落"视频里灯（设备）在画面中移动，固定 ROI 对不准。
  本脚本逐帧用 green_excess>15 取最大绿色连通区跟踪灯体，
  - 实时画跟踪框（亮帧绿框、暗帧不画，避免噪声污染）；
  - 用亮↔暗跳变数校验闪烁节律（正常约 1秒内"亮-暗-亮"，~1-1.5Hz）；
  - 累计【亮帧】灯体范围，外扩成固定的"建议检测区域"（黄框），
    供固定摄像头部署时参考框的大小与位置。
  - 建议区域限定在画面内（归一化坐标 0-100，y 不超出 100）。

⚠️ 这是临时验证/展示脚本，不属于正式检测流程。
   正式代码（slot_detector.py + config 的固定 ROI 架构）不依赖本脚本。
   固定摄像头落地后灯位稳定，直接用固定 ROI 即可，无需跟踪。

用法：
    python probe_track.py --video "D:/.../正常起落.MP4"
    python probe_track.py --video "D:/.../正常起落.MP4" --no-show --pad 25
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


# green_excess>此阈值视为"灯亮像素"
GREEN_THRESH = 15
# 连通区面积下限（去噪点）
MIN_AREA = 50
# 亮帧判定：灯体 mean_ge 高于此值才算"亮帧"参与累计与绿框绘制
# （暗帧灯灭，green_excess 残余低，避免把噪声当灯体累计进建议区域）
LIT_GE_MIN = 25.0


def find_light_blob(frame):
    """取 green_excess>阈值的最大连通区，返回 (x,y,w,h,mean_ge) 或 None。"""
    bgr = frame.astype(np.int16)
    ge = bgr[:, :, 1] - np.maximum(bgr[:, :, 0], bgr[:, :, 2])
    mask = (ge > GREEN_THRESH).astype(np.uint8)
    ncc, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if ncc <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = 1 + int(np.argmax(areas))
    if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
        return None
    x = int(stats[k, cv2.CC_STAT_LEFT])
    y = int(stats[k, cv2.CC_STAT_TOP])
    w = int(stats[k, cv2.CC_STAT_WIDTH])
    h = int(stats[k, cv2.CC_STAT_HEIGHT])
    msk = (lab[y:y + h, x:x + w] == k)
    mean_ge = float(ge[y:y + h, x:x + w][msk].mean())
    return x, y, w, h, mean_ge


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="灯体跟踪探针（临时展示用）")
    ap.add_argument("--video", required=True, help="视频路径")
    ap.add_argument("--pad", type=int, default=20,
                    help="建议区域相对灯体常驻位的外扩像素（默认20）")
    ap.add_argument("--no-show", action="store_true", help="不弹窗显示")
    ap.add_argument("--no-save", action="store_true", help="不输出标注视频")
    ap.add_argument("--start", type=int, default=0, help="从指定帧开始")
    # 原 ROI 对比（可选，归一化 0-100，多个用 ; 分隔，格式 X,Y,W,H）
    ap.add_argument("--orig-rois", default=None,
                    help="对比用的原ROI(归一化X,Y,W,H)，多个用';'分隔，用于在视频上对比是否对准灯")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {args.video}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[视频] {args.video}\n       帧数={total} fps={fps:.1f} 尺寸={width}x{height}")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
        print(f"[跳过] 从第 {args.start} 帧开始")

    # 标注视频输出
    writer = None
    out_path = None
    if not args.no_save:
        src_dir = os.path.dirname(os.path.abspath(args.video)) or "."
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(src_dir, f"{base}_track.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        print(f"[输出] {out_path}")

    # 解析原 ROI（像素坐标）用于对比绘制
    orig_rois_px = []
    if args.orig_rois:
        for s in args.orig_rois.split(";"):
            s = s.strip()
            if not s:
                continue
            X, Y, W, H = [float(v) for v in s.split(",")]
            ox = round(X / 100 * width)
            oy = round(Y / 100 * height)
            ow = round(W / 100 * width)
            oh = round(H / 100 * height)
            orig_rois_px.append((ox, oy, ow, oh))

    # 累计【亮帧】灯体质心与尺寸（用于生成"常驻位"建议检测区域）
    lit_cx, lit_cy, lit_w, lit_h = [], [], [], []
    xs_min, xs_max = width, 0   # 全程轨迹范围（仅日志参考）
    ys_min, ys_max = height, 0
    lit_count = 0       # 亮帧数（mean_ge 达标）
    blob_count = 0      # 检出灯体(含暗帧)数
    bin_seq = []        # 闪烁节律：亮/暗二值序列

    show = not args.no_show
    frame_no = args.start
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            blob = find_light_blob(frame)
            vis = frame.copy()

            # 先画原 ROI（蓝框，置底），便于和跟踪框对比；用蓝以区别于状态色
            for oi, (ox, oy, ow, oh) in enumerate(orig_rois_px):
                cv2.rectangle(vis, (ox, oy), (ox + ow, oy + oh), (255, 0, 0), 2)
                cv2.putText(vis, f"orig{oi+1}", (ox, max(oy - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 框色规范：闪烁(正常)=绿，不亮(故障)=黄，常亮(故障)=红，检测中=灰
            # 探针逐帧跟踪：亮帧=绿(灯亮中)，暗帧/未检出=灰(检测中)
            COL_LIT = (0, 200, 0)       # 绿
            COL_DARK = (160, 160, 160)  # 灰
            is_lit = False
            if blob is not None:
                x, y, w, h, mge = blob
                blob_count += 1
                is_lit = mge >= LIT_GE_MIN
                if is_lit:
                    lit_count += 1
                    lit_cx.append(x + w / 2.0)
                    lit_cy.append(y + h / 2.0)
                    lit_w.append(w)
                    lit_h.append(h)
                    xs_min = min(xs_min, x); ys_min = min(ys_min, y)
                    xs_max = max(xs_max, x + w); ys_max = max(ys_max, y + h)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), COL_LIT, 2)
                    cv2.putText(vis, f"ge{mge:.0f}", (x, max(y - 6, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LIT, 1)
                    status_txt = "FLASH(亮)"
                    status_col = COL_LIT
                else:
                    # 暗帧：灯灭，灰框(检测中)，不累计范围
                    cv2.rectangle(vis, (x, y), (x + w, y + h), COL_DARK, 1)
                    status_txt = "dark(暗)"
                    status_col = COL_DARK
            else:
                status_txt = "dark(暗)"
                status_col = COL_DARK
            bin_seq.append(1 if is_lit else 0)

            # 顶部信息条
            bar = f"f:{frame_no}  {status_txt}"
            cv2.rectangle(vis, (0, 0), (300, 28), (0, 0, 0), -1)
            cv2.putText(vis, bar, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 1)

            if writer:
                writer.write(vis)
            if show:
                h_, w_ = vis.shape[:2]
                disp = vis
                if w_ > 1280:
                    s = 1280.0 / w_
                    disp = cv2.resize(vis, (int(w_ * s), int(h_ * s)))
                cv2.imshow("probe_track", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[中断] 用户退出")
                    break
            frame_no += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    # ---- 闪烁节律校验 ----
    print("-" * 60)
    bin_arr = np.array(bin_seq, dtype=np.int8)
    n_trans = int(np.sum(np.diff(bin_arr) != 0))  # 亮↔暗 跳变数
    dur = max(frame_no - args.start, 1) / fps
    trans_per_sec = n_trans / dur
    # "亮-暗-亮"周期数 ≈ 跳变数/2
    cycles_per_sec = trans_per_sec / 2
    print(f"[闪烁节律] 亮帧={lit_count} 检出帧={blob_count} 总帧={frame_no - args.start}")
    print(f"           亮↔暗跳变={n_trans} 次, {trans_per_sec:.2f} 次/秒, "
          f"约 {cycles_per_sec:.2f} Hz (亮-暗-亮周期/秒)")
    if cycles_per_sec >= 0.5:
        print(f"           -> 判定: 闪烁(flashing) 正常 ✓ (符合约1秒1个亮-暗-亮周期)")
    else:
        print(f"           -> 判定: 跳变过少，疑似常亮/不亮")

    # ---- 建议检测区域（灯体常驻位小框）----
    if lit_count == 0:
        print("[结果] 全程无亮帧（green_excess 阈值或 LIT_GE_MIN 是否过高？）")
        return 0

    # 常驻位：亮帧质心中位数 ± 灯体尺寸中位数，再外扩 pad
    cx = float(np.median(lit_cx))
    cy = float(np.median(lit_cy))
    bw = float(np.median(lit_w))
    bh = float(np.median(lit_h))
    rx0 = int(round(cx - bw / 2 - args.pad))
    ry0 = int(round(cy - bh / 2 - args.pad))
    rx1 = int(round(cx + bw / 2 + args.pad))
    ry1 = int(round(cy + bh / 2 + args.pad))
    # 夹到画面内（y 归一化保证 0-100）
    rx0 = max(rx0, 0); ry0 = max(ry0, 0)
    rx1 = min(rx1, width); ry1 = min(ry1, height)
    rw = rx1 - rx0
    rh = ry1 - ry0

    print(f"[灯体常驻位] 质心=({cx:.0f},{cy:.0f}) 灯体尺寸中位={bw:.0f}x{bh:.0f}")
    print(f"[全程轨迹范围(参考)] x[{xs_min}-{xs_max}] y[{ys_min}-{ys_max}] "
          f"(起落移动造成的范围，非固定ROI)")
    print(f"[建议检测区域] (常驻位+灯体尺寸, 外扩 {args.pad}px, 已夹到画面内, y归一化0-100):")
    print(f"       像素:   x={rx0} y={ry0} w={rw} h={rh}")
    nX = rx0 / width * 100
    nY = ry0 / height * 100
    nW = rw / width * 100
    nH = rh / height * 100
    print(f"       归一化: X={nX:.4f} Y={nY:.4f} W={nW:.4f} H={nH:.4f}")
    print(f"[说明] 固定摄像头部署后灯位稳定，可直接把上述区域作为 config 的 ROI。")
    print(f"       本脚本仅用于手持视频的展示与区域划定，正式检测仍用固定 ROI。")
    if out_path:
        print(f"[完成] 标注视频已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
