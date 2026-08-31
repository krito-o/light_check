"""部署代码：摄像头输入 → 检测 → 告警日志 + 控制台输出。

用法：
    python deploy.py --config config_camera.json
    python deploy.py --config config_camera.json --camera 0
    python deploy.py --config config_camera.json --log-dir /var/log/light_check
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

import cv2

from engine import LightCheckEngine


def run_deploy(config_path, *, camera_id=0, log_dir="logs"):
    """从摄像头读取，逐帧检测，告警输出到 JSON 日志 + 控制台。

    不弹窗（部署无 GUI），不输出标注视频。
    Ctrl+C 优雅退出，打印汇总。
    """
    engine = LightCheckEngine(config_path)
    config = engine.config
    video_name = config.get("video_name", f"camera_{camera_id}")
    fps = config.get("fps", 50)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[错误] 无法打开摄像头 {camera_id}", file=sys.stderr)
        return 2

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    print(f"[摄像头] {camera_id}  {width}x{height}  fps={cap_fps:.1f}")
    print(f"[配置] ROI 数={engine.detector.n_lights}  "
          f"enter_method={engine.detector.enter_method}  "
          f"span_size={engine.detector.span_size}")
    for i, roi in enumerate(engine.detector.roi_list):
        print(f"       ROI[{i}] {roi.get('name','?')} "
              f"px=({roi['x']},{roi['y']},{roi['w']},{roi['h']})")

    print("-" * 60)
    print("[运行中] 按 Ctrl+C 停止")

    # 每 N 秒打印状态摘要
    last_summary = 0
    summary_interval = 30  # 秒

    # 优雅退出
    running = True

    def on_stop(sig, frame):
        nonlocal running
        running = False
        print("\n[停止] 收到停止信号...")

    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("[警告] 摄像头读帧失败，重试...", file=sys.stderr)
                time.sleep(0.1)
                continue

            result = engine.process_frame(frame)
            fi = result["frame_idx"]

            # 告警日志（每帧写入 JSON 行）
            engine.alarm_log(log_dir)

            # 定期打印状态摘要
            now = time.time()
            if now - last_summary > summary_interval:
                last_summary = now
                alarm_slots = engine.alarm_slots()
                if alarm_slots:
                    parts = [f"{s['name']}={s['light_status']}" for s in alarm_slots]
                    print(f"[f{fi:6d}] 告警槽: " + ", ".join(parts))
                else:
                    in_slot = sum(1 for s in engine.all_status() if s["in_slot"])
                    print(f"[f{fi:6d}] 正常 在槽数={in_slot}/{engine.detector.n_lights}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    # 汇总
    print("\n" + "-" * 60)
    print("[汇总] 最终状态：")
    for s in engine.all_status():
        tag = "已入槽" if s["in_slot"] else "已出槽"
        print(f"         {s['name']:14s} {s['light_status']:12s} ({tag})")
    print(f"[完成] 共处理 {engine.detector.frame_idx + 1} 帧")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="工业指示灯检测系统（部署模式）")
    ap.add_argument("--config", required=True, help="配置文件路径 (json)")
    ap.add_argument("--camera", type=int, default=0, help="摄像头 ID（默认 0）")
    ap.add_argument("--log-dir", default="logs", help="告警日志目录（默认 logs）")
    args = ap.parse_args()

    sys.exit(run_deploy(args.config, camera_id=args.camera, log_dir=args.log_dir))


if __name__ == "__main__":
    main()