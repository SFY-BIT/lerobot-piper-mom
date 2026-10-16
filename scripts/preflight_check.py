"""启动前连接预检 — 检查所有设备是否就绪

检查项:
  1. libusb 后端 (CAN 通信必需)
  2. CAN 设备 (candleLight USB-CAN)
  3. NI-DAQmx (力传感器)
  4. UArm 串口 (遥操作)
  5. 摄像头
  6. ffmpeg

用法: python preflight_check.py
"""

import sys
import os
import subprocess
from pathlib import Path


def check(label, ok_msg, fail_msg):
    """打印检查结果"""
    pass  # 占位，实际在 try/except 中调用


def main():
    print("=" * 55)
    print("PRE-FLIGHT CHECK — lerobot + Piper + Force Sensor")
    print("=" * 55)
    all_ok = True

    # ── 1. libusb 后端 ──
    print("\n[1] libusb backend (CAN)...", end=" ")
    try:
        import usb.core
        devs = list(usb.core.find(find_all=True))
    except usb.core.NoBackendError:
        print("FAIL: NoBackendError — libusb-1.0.dll not found")
        print("      FIX: $env:Path = 'D:\\miniconda\\envs\\lerobot\\Library\\bin;$env:Path'")
        all_ok = False
        devs = []
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False
        devs = []

    if devs:
        print(f"OK ({len(devs)} USB device(s))")
        for d in devs:
            try:
                name = usb.util.get_string(d, d.iProduct) if d.iProduct else "?"
            except Exception:
                name = "?"
            print(f"      VID={d.idVendor:04X} PID={d.idProduct:04X}  {name}")
    elif all_ok:
        print("OK (no devices, but backend works)")

    # ── 2. CAN 设备 candleLight ──
    print("\n[2] CAN device (candleLight VID=1D50 PID=606F)...", end=" ")
    try:
        import usb.core
        dev = usb.core.find(idVendor=0x1D50, idProduct=0x606F)
        if dev:
            print(f"OK (bus={dev.bus}, addr={dev.address})")
        else:
            print("NOT FOUND — check USB cable")
            all_ok = False
    except Exception:
        print("SKIP (libusb not available)")
        all_ok = False

    # ── 3. NI-DAQmx ──
    print("\n[3] NI-DAQmx (force sensor)...", end=" ")
    try:
        import nidaqmx
        system = nidaqmx.system.System.local()
        devs = system.devices
        if devs:
            print(f"OK ({len(devs)} device(s))")
            for d in devs:
                print(f"      {d.name}: {d.product_type}")
        else:
            print("NO DEVICES")
            all_ok = False
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False

    # ── 4. UArm 串口 ──
    uarm_port = os.getenv("LEROBOT_UARM_PORT", "COM5")
    print(f"\n[4] UArm serial ({uarm_port})...", end=" ")
    try:
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if uarm_port in ports:
            print(f"OK ({uarm_port} found)")
        else:
            print(f"NOT FOUND — available: {ports}")
            all_ok = False
    except Exception as e:
        print(f"SKIP: {e}")

    # ── 5. 摄像头 ──
    cam_idx = os.getenv("LEROBOT_PIPER_CAM_ONE_INDEX", "3")
    print(f"\n[5] Camera (index={cam_idx})...", end=" ")
    try:
        import cv2
        idx = int(cam_idx)
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"OK ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print("OPENED but can't read frame")
            cap.release()
        else:
            print(f"CAN'T OPEN index {idx}")
            all_ok = False
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False

    # ── 6. ffmpeg ──
    print("\n[6] ffmpeg...", end=" ")
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        ver = result.stdout.split("\n")[0].split(" ")[2][:5]
        print(f"OK (v{ver})")
    except FileNotFoundError:
        print("NOT IN PATH")
        all_ok = False
    except Exception as e:
        print(f"FAIL: {e}")
        all_ok = False

    # ── 7. 力传感器标定 ──
    cal_path = os.getenv("LEROBOT_FORCE_CAL_PATH", "")
    print(f"\n[7] Force sensor calibration...", end=" ")
    if cal_path and Path(cal_path).exists():
        print(f"OK ({Path(cal_path).name})")
    else:
        print(f"MISSING: {cal_path}" if cal_path else "NOT SET")
        all_ok = False

    # ── 8. Python 路径 ──
    print(f"\n[8] Python: {sys.executable}")
    print(f"    Version: {sys.version.split()[0]}")

    # ── 总结 ──
    print("\n" + "=" * 55)
    if all_ok:
        print("ALL CHECKS PASSED — ready to record!")
    else:
        print("SOME CHECKS FAILED — review above and fix before starting")
    print("=" * 55)


if __name__ == "__main__":
    main()
