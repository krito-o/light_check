"""独立测试代码：视频输入 → 检测 → 标注视频 + CSV 日志 + 控制台输出。

用法（与 main.py 一致）：
    python test.py --config config_C0065.json                 # 默认输出标注视频到源视频同级目录
    python test.py --config config_off.json --no-show         # 无界面运行（批量/服务器）
    python test.py --config config_off.json --start 450        # 从指定帧开始
    python test.py --config config_off.json --no-save          # 不输出标注视频
    python test.py --config config_on.json --save-annotated /path/out.mp4  # 指定标注视频路径
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2

from slot_detector import SlotDetector
from visualizer import draw_overlay, STATUS_CN


STATUS_CN_MAIN = {
    "flashing": "闪烁(正常)",
    "steady_on": "常亮(故障)",
    "off": "不亮(故障)",
    "unknown": "未知",
    "pending": "待定",
}


def default_annotated_path(video_path, video_name):
    """标注视频默认输出路径：源视频同级目录，文件名 <video_name>_annotated.mp4。"""
    src_dir = os.path.dirname(os.path.abspath(video_path)) or "."
    return os.path.join(src_dir, f"{video_name}_annotated.mp4")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_logger(log_dir, video_name):
    """创建 CSV 日志文件并写入表头。返回 (filepath, writer, file_handle)。"""
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = video_name or "video"
    fp = os.path.join(log_dir, f"{safe}_{ts}.csv")
    fh = open(fp, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(fh)
    writer.writerow([
        "frame", "state", "roi_index", "roi_name", "status", "expected",
        "median", "mad", "osc", "pp", "freq",
        "on_thr", "lit_ratio", "area_med", "gray_med",
    ])
    return fp, writer, fh


def run_test(config_path, *, no_show=False, annotated_path=None, no_save=False, start_frame=0):
    """从视频文件读取，逐帧检测，输出控制台/CSV日志 + 标注视频。"""
    config = load_config(config_path)
    video_path = config["video_path"]
    video_name = config.get("video_name", os.path.splitext(os.path.basename(video_path))[0])

    detector = SlotDetector(config)
    show = config.get("show", True) and not no_show

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频: {video_path}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS) or config.get("fps", 50)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log_fp, log_writer, log_fh = open_logger(config.get("log_dir", "logs"), video_name)
    print(f"[日志] {log_fp}")
    print(f"[视频] {video_path}  帧数={total}  fps={fps:.1f}")
    print(f"[配置] ROI 数={detector.n_lights}  skip_enter={detector.skip_enter}  "
          f"span_size={detector.span_size}  span_step={detector.span_step}  "
          f"confirm_spans={detector.confirm_spans}")
    for i, roi in enumerate(detector.roi_list):
        print(f"       ROI[{i}] {roi.get('name','?')} "
              f"px=({roi['x']},{roi['y']},{roi['w']},{roi['h']}) "
              f"norm=({roi['norm']['X']:.2f},{roi['norm']['Y']:.2f},{roi['norm']['W']:.2f},{roi['norm']['H']:.2f}) "
              f"期望={roi.get('expected','-')}")

    # 跳过起始段（手持晃动）
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        detector.frame_idx = start_frame - 1
        print(f"[跳过] 从第 {start_frame} 帧开始")

    # 标注视频输出
    writer = None
    out_path = None
    if not no_save:
        out_path = annotated_path or default_annotated_path(video_path, video_name)
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[警告] 无法创建标注视频（编码不支持？）: {out_path}", file=sys.stderr)
            writer = None
            out_path = None
        else:
            print(f"[输出] 标注视频 → {out_path}")

    print("-" * 60)

    last_print = {i: "pending" for i in range(detector.n_lights)}
    frame_no = start_frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            states, info = detector.process_frame(frame)

            # 状态变化时打印 + 写日志
            changed = info.get("changed", [])
            span_results = info.get("span_results", [])

            for i in range(detector.n_lights):
                confirmed = detector.confirmed[i]
                if i in changed or (span_results[i] is not None):
                    if confirmed != last_print[i] or span_results[i] is not None:
                        roi = config["roi_list"][i]
                        sr = span_results[i] or {}
                        st_i = states[i] if i < len(states) else "WAITING"
                        slot_tag = "已入槽" if st_i == "DETECTING" else "已出槽"
                        line = (f"[f{frame_no:5d}] {slot_tag} "
                                f"{roi.get('name', f'ROI{i}'):14s} -> "
                                f"{STATUS_CN_MAIN.get(confirmed, confirmed):12s} "
                                f"(期望:{STATUS_CN_MAIN.get(roi.get('expected','-'),roi.get('expected','-'))})")
                        if sr:
                            line += (f"  median={sr.get('median',0):.1f} osc={sr.get('osc',0)} "
                                     f"pp={sr.get('pp',0):.1f} area={sr.get('area_med',0):.0f} "
                                     f"gray={sr.get('gray_med',0):.0f}")
                        print(line)
                        last_print[i] = confirmed
                        # CSV
                        log_writer.writerow([
                            frame_no, st_i, i, roi.get("name", f"ROI{i}"),
                            confirmed, roi.get("expected", ""),
                            round(sr.get("median", 0), 2), round(sr.get("mad", 0), 2),
                            sr.get("osc", 0), round(sr.get("pp", 0), 2),
                            round(sr.get("freq", 0), 2),
                            round(sr.get("on_thr", 0), 2), round(sr.get("lit_ratio", 0), 3),
                            round(sr.get("area_med", 0), 1), round(sr.get("gray_med", 0), 1),
                        ])

            # 可视化
            if show or writer:
                vis = draw_overlay(detector, frame)
                if writer:
                    writer.write(vis)
                if show:
                    disp = vis
                    h, w = vis.shape[:2]
                    if w > 1280:
                        scale = 1280.0 / w
                        disp = cv2.resize(vis, (int(w * scale), int(h * scale)))
                    cv2.imshow("light_check", disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print("[中断] 用户退出")
                        break
                    elif key == ord(" "):
                        while True:
                            k2 = cv2.waitKey(50) & 0xFF
                            if k2 == ord(" ") or k2 == ord("q"):
                                break
                        if k2 == ord("q"):
                            break

            frame_no += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        log_fh.close()
        if show:
            cv2.destroyAllWindows()

    # 汇总
    print("-" * 60)
    print("[汇总] 最终确认状态（出槽槽位显示最近一次在槽判定）：")
    in_slot = detector.in_slot_states
    for i, roi in enumerate(config["roi_list"]):
        conf = in_slot[i]
        exp = roi.get("expected", "-")
        ok = "[OK]" if conf == exp else "[X] "
        cur = detector.confirmed[i]
        cur_tag = "" if cur == conf else f"(当前:{STATUS_CN_MAIN.get(cur, cur)})"
        print(f"         {ok} ROI[{i}] {roi.get('name','?'):14s} "
              f"判定={STATUS_CN_MAIN.get(conf, conf):12s} 期望={STATUS_CN_MAIN.get(exp, exp)} {cur_tag}")
    print(f"[完成] 日志已写入 {log_fp}")
    if out_path:
        print(f"[完成] 标注视频已写入 {out_path}")
    return 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="工业指示灯检测系统（测试模式）")
    ap.add_argument("--config", required=True, help="配置文件路径 (json)")
    ap.add_argument("--no-show", action="store_true", help="不弹窗显示（无界面运行）")
    ap.add_argument("--save-annotated", default=None,
                    help="标注视频输出路径 (mp4)；不指定则默认输出到源视频同级目录")
    ap.add_argument("--no-save", action="store_true",
                    help="不输出标注视频（默认会输出）")
    ap.add_argument("--start", type=int, default=0, help="从指定帧开始（跳过起始晃动段）")
    args = ap.parse_args()

    sys.exit(run_test(args.config, no_show=args.no_show,
                      annotated_path=args.save_annotated, no_save=args.no_save,
                      start_frame=args.start))


if __name__ == "__main__":
    main()