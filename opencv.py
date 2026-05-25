from pathlib import Path

import cv2


PROP_NAMES = {
    "FRAME_WIDTH": "CAP_PROP_FRAME_WIDTH",
    "FRAME_HEIGHT": "CAP_PROP_FRAME_HEIGHT",
    "FPS": "CAP_PROP_FPS",
    "BRIGHTNESS": "CAP_PROP_BRIGHTNESS",
    "CONTRAST": "CAP_PROP_CONTRAST",
    "SATURATION": "CAP_PROP_SATURATION",
    "HUE": "CAP_PROP_HUE",
    "GAIN": "CAP_PROP_GAIN",
    "EXPOSURE": "CAP_PROP_EXPOSURE",
    "AUTO_EXPOSURE": "CAP_PROP_AUTO_EXPOSURE",
    "SHARPNESS": "CAP_PROP_SHARPNESS",
    "GAMMA": "CAP_PROP_GAMMA",
    "TEMPERATURE": "CAP_PROP_TEMPERATURE",
    "FOCUS": "CAP_PROP_FOCUS",
    "AUTOFOCUS": "CAP_PROP_AUTOFOCUS",
    "ZOOM": "CAP_PROP_ZOOM",
    "WB_BLUE_U": "CAP_PROP_WB_BLUE_U",
    "WB_RED_V": "CAP_PROP_WB_RED_V",
    "BACKLIGHT": "CAP_PROP_BACKLIGHT",
    "BUFFERSIZE": "CAP_PROP_BUFFERSIZE",
    "FOURCC": "CAP_PROP_FOURCC",
    "FORMAT": "CAP_PROP_FORMAT",
    "MODE": "CAP_PROP_MODE",
}


def list_camera_devices():
    by_id_dir = Path("/dev/v4l/by-id")
    devices = []
    seen = set()

    if by_id_dir.exists():
        for path in sorted(by_id_dir.iterdir()):
            try:
                resolved = path.resolve()
            except FileNotFoundError:
                continue
            resolved_str = str(resolved)
            if resolved_str in seen:
                continue
            seen.add(resolved_str)
            devices.append((str(path), resolved_str))

    for path in sorted(Path("/dev").glob("video*")):
        resolved_str = str(path.resolve())
        if resolved_str in seen:
            continue
        seen.add(resolved_str)
        devices.append((str(path), resolved_str))

    return devices


def extract_index(device_path):
    name = Path(device_path).name
    if name.startswith("video"):
        suffix = name.removeprefix("video")
        if suffix.isdigit():
            return int(suffix)
    return None


def dump_properties(cap):
    for name, attr_name in PROP_NAMES.items():
        pid = getattr(cv2, attr_name, None)
        if pid is None:
            continue
        value = cap.get(pid)
        print(f"  {name}: {value}")


def try_open_camera(display_path, device_path):
    index = extract_index(device_path)
    print(f"设备: {display_path}")
    print(f"真实路径: {device_path}")
    print(f"OpenCV index: {index if index is not None else '未知'}")

    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("  打开失败")
        print()
        return False

    ret, frame = cap.read()
    if not ret or frame is None:
        print("  已打开，但无法读取首帧")
        cap.release()
        print()
        return False

    print(f"  首帧读取成功: shape={frame.shape}")
    dump_properties(cap)
    cap.release()
    print()
    return True


def main():
    devices = list_camera_devices()
    if not devices:
        print("没有发现任何 /dev/video* 或 /dev/v4l/by-id 设备")
        raise SystemExit(1)

    print("发现以下视频设备：")
    for index, (display_path, device_path) in enumerate(devices):
        video_index = extract_index(device_path)
        print(
            f"[{index}] {display_path} -> {device_path} "
            f"(OpenCV index: {video_index if video_index is not None else '未知'})"
        )
    print()

    success_count = 0
    readable_devices = []
    for display_path, device_path in devices:
        if try_open_camera(display_path, device_path):
            success_count += 1
            readable_devices.append((display_path, device_path, extract_index(device_path)))

    print(f"可正常读取首帧的设备数量: {success_count}/{len(devices)}")
    if readable_devices:
        print("可读设备汇总：")
        for display_path, device_path, index in readable_devices:
            print(
                f"  - {display_path} -> {device_path} "
                f"(OpenCV index: {index if index is not None else '未知'})"
            )


if __name__ == "__main__":
    main()
