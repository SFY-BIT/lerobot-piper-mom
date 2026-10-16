# Windows 完整运行指南 (30Hz + 力传感器 + 正运动学)

> **环境**：Windows + conda 环境 `lerobot`（Python 3.10，`D:\miniconda\envs\lerobot\python.exe`）+ 本仓库
> **约定**：所有命令都在**仓库根目录** `C:\Users\User\Desktop\lerobot\lerobot-piper-mom` 下执行
> **每次新开终端**都要先跑 §2 的环境变量，否则设备/相机/力传感器都连不上
> **热键**：先看 §0 总表，细节在 §7

---

## 0. 速查总表：做什么 → 用什么指令 → 会用到哪些热键

| # | 工作 | 指令（仓库根目录） | 键盘热键 | 手柄 / 领航臂 |
|---|------|-------------------|---------|--------------|
| 1 | 设备预检 | `scripts/preflight_check.py` | — | — |
| 2 | 纯遥控（调视角/看相机） | `lerobot/scripts/control_robot.py --control.type=teleoperate` | 无（`ESC`/`→` 不生效） | 手柄 `RB` 提前结束；领航臂 `q` |
| 3 | 普通录制（无模型） | `control_robot.py --control.type=record` | `→` `←` `ESC` | 手柄 `RB` `LB` `Y` `Start`；领航臂 `q` |
| 4 | **模型引导录制**（16-token + 力） | `scripts/record_ws_dataset.py` | `s` `f` `→` `←` `h` `v` `ESC` | —（不接管手柄） |
| 5 | 纯手控录制（不连 server） | `record_ws_dataset.py --no-server --start-teleop` | `→` `←` `v` `ESC`（`h` 被拒绝） | — |
| 6 | 推理验证（旧 ZMQ client） | `lerobot/scripts/piper_force_client.py` | 无（跑满 `--duration-s` 或 Ctrl-C） | 手柄 `RB` 提前结束 |
| 7 | 推理验证（新 websocket client） | `scripts/piper_ws_force_client.py` | `f` `→` `v` `ESC` | 手柄 `RB` 提前结束 |
| 8 | 提取视频（parquet→mp4） | `scripts/extract_parquet_videos.py` | — | — |
| 9 | 转 flexiv 兼容格式 | `scripts/finalize_dataset.py` | — | — |
| 10 | 删坏轨迹 + 补录 | `scripts/remove_episode.py` → `record_ws_dataset.py --resume` | `s` `f` `→` | — |
| 11 | 数据集统计 / 力曲线 | `scripts/summarize_test_datasets.py` | — | — |
| 12 | 力传感器归零 | `scripts/test/test_force_viz.py` | `b` `t` `z` `r` | — |

> ⚠️ **两套键盘机制，效果不一样**：
> - `control_robot.py` 用 **pynput**（全局键盘监听）——终端在后台也生效
> - `record_ws_dataset.py` / `piper_ws_force_client.py` 用 **msvcrt**（控制台原始输入）——**必须让运行脚本的那个终端窗口处于焦点**，否则按键无效
> - 手柄/领航臂的事件只有 `control_robot.py` 路径会消费（它调用 `robot.consume_control_events()`）

---

## 1. 环境要求

- conda 环境 `lerobot`（Python 3.10，路径 `D:\miniconda\envs\lerobot\python.exe`）
- **注意：`conda activate` 不可用，必须用完整路径调用 python**
- ffmpeg（winget 安装）
- NI-DAQmx（NI USB-6341 驱动，ATI Mini40 力传感器用）
- candleLight USB-CAN（libusb / Zadig 配 WinUSB）

---

## 2. 环境变量（每个新终端都要设）

```powershell
# ===== 机械臂 =====
$env:LEROBOT_PIPER_INTERP_SLEEP_S = "0"           # ★ 必须为0
$env:LEROBOT_CAN_BUSTYPE = "gs_usb"
$env:LEROBOT_CAN_CHANNEL = "0"
$env:LEROBOT_UARM_PORT = "COM5"
$env:LEROBOT_PIPER_TELEOP_SOURCE = "uarm"
$env:LEROBOT_PIPER_STARTUP_ZERO_HOLD_S = "0.4"

# ===== 摄像头 =====
$env:LEROBOT_PIPER_CAM_ONE_INDEX = "3"
$env:LEROBOT_OPENCV_BACKEND = "MSMF"

# ===== 力传感器 (可选) =====
# 设置 = 启用力传感器; 【不设置/注释掉这行】= 不使用力传感器 (拆掉传感器时用, 见 §10)
$env:LEROBOT_FORCE_CAL_PATH = "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\FORCE\FORCE\FT32663\FT32663.cal"

# ===== 系统 PATH =====
$env:Path = "C:\Users\User\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin;$env:Path"
$env:Path = "D:\miniconda\envs\lerobot\Library\bin;$env:Path"
```

全部变量含义见 §9。

---

## 3. 预检（每次开机动设备前先跑）

```powershell
D:\miniconda\envs\lerobot\python.exe scripts/preflight_check.py
```

检查项：CAN 适配器 / 摄像头 / DAQ / UArm 串口 / ffmpeg。

---

## 4. 工作流 A：纯遥控（调视角 / 检查相机 / 手把手验证）

```powershell
D:\miniconda\envs\lerobot\python.exe lerobot/scripts/control_robot.py `
  --robot.type=piper --robot.inference_time=false `
  --control.type=teleoperate --control.display_cameras=true --control.fps=30
```

- **这个模式没有 `→`/`←`/`ESC` 热键**（`control_robot.py` 只在 `record` 分支装键盘监听）
- 结束方式：`Ctrl-C`，或加 `--control.teleop_time_s=N` 等它超时（默认 `None` = 无限循环）
- 例外：手柄 `RB` / 领航臂 `q` 会触发 `exit_early`，能提前结束循环（`control_loop` 消费 `consume_control_events()`）

---

## 5. 工作流 B：数据集录制（3 种模式）

### 5.1 先选模式

| | 模式 A 普通录制 | 模式 B 模型引导录制 | 模式 C 纯手控录制 |
|---|---|---|---|
| 脚本 | `control_robot.py record` | `record_ws_dataset.py` | `record_ws_dataset.py --no-server --start-teleop` |
| 模型 | 无（纯人工） | 有（websocket 推理） | 无（全程人工遥控） |
| 力传感器 | 装了就有 `observation.force` | **必需**（脚本直接读该键） | **必需** |
| 轨迹成功/失败标注 | `results.jsonl`（靠 `→`/`←` 提前结束…） | `results.jsonl` + `s`/`f`/`→` | `→`（无标注） |
| `force_pred` / `value` 列 | 无 | `--record-force-pred` / `--record-value` | 可加 |
| 归零（每段前回起点） | 无（手动） | **有**（内置低速归零） | **有** |
| 适用 | 早期任务、无模型对照 | 论文主流程 | 补录、模型不可用时的对照数据 |

---

### 5.2 模式 A：普通录制（`control_robot.py record`）

```powershell
# 旧任务 stamp seal
D:\miniconda\envs\lerobot\python.exe lerobot/scripts/control_robot.py `
  --robot.type=piper --robot.inference_time=false `
  --control.type=record --control.fps=30 `
  --control.display_cameras=true --control.warmup_time_s=2 `
  --control.episode_time_s=60 --control.reset_time_s=10 `
  --control.repo_id=my_robot/stamp_seal `
  --control.num_episodes=3 --control.single_task="stamp seal" `
  --control.push_to_hub=false --control.video=true

# 新任务 Erase Board (40 条轨迹)
D:\miniconda\envs\lerobot\python.exe lerobot/scripts/control_robot.py `
  --robot.type=piper --robot.inference_time=false `
  --control.type=record --control.fps=30 `
  --control.display_cameras=true --control.warmup_time_s=2 `
  --control.episode_time_s=60 --control.reset_time_s=10 `
  --control.repo_id=my_robot/erase_board `
  --control.num_episodes=40 --control.single_task="Erase Board" `
  --control.push_to_hub=false --control.video=true
```

流程：`warmup(2s)` → 每段 `录制(episode_time_s)` → `reset(重置环境 reset_time_s)` → 直到录满 `num_episodes`。

**`--control.*` 参数（默认值取自 [control_configs.py](c:/Users/User/Desktop/lerobot/lerobot-piper-mom/lerobot/common/robot_devices/control_configs.py)）：**

| 参数 | 默认 | 含义 |
|------|:--:|------|
| `--control.repo_id` | 必填 | 数据集名（约定 `{hf用户名}/{数据集名}`） |
| `--control.single_task` | 必填 | 任务描述（写进数据集） |
| `--control.num_episodes` | 50 | 录制段数 |
| `--control.episode_time_s` | 60 | 单段录制时长（秒） |
| `--control.reset_time_s` | 60 | 段间重置时长（秒） |
| `--control.warmup_time_s` | 10 | 开录前预热（设备同步用） |
| `--control.fps` | 策略 fps（无策略时不限） | 帧率 |
| `--control.video` | True | 把帧编码成视频 |
| `--control.push_to_hub` | **True** | ⚠️ 默认会上传 HuggingFace Hub，本地录制务必显式 `false` |
| `--control.private` | False | HF 仓库设为私有 |
| `--control.resume` | False | 续录已有数据集 |
| `--control.display_cameras` | True | 实时显示相机画面 |
| `--control.play_sounds` | True | 语音播报事件 |
| `--control.root` | 空 | 数据集存储根目录 |
| `--control.num_image_writer_processes` | 0 | 写图子进程数（fps 不稳时调） |
| `--control.num_image_writer_threads_per_camera` | 4 | 每相机写图线程数 |
| `--control.policy` | 空 | 指定策略后可边推理边录（一般用模式 B 代替） |

`--control.type=teleoperate` 的参数更少：`fps`（默认不限）、`teleop_time_s`（默认 `None` = 无限）、`display_cameras`（默认 True）。

热键：见 §7.1（`→` 提前结束当前阶段、`←` 重录上一条、`ESC` 结束整个流程）。

---

### 5.3 模式 B：模型引导录制（`record_ws_dataset.py`）★ 主力

**每段（episode）自动流程**：

1. **归零**：低速插值回启动零位（`RETURN_SPEED=40`，不吃推理值，避免瞬间全速发热）
2. **取起点**：把归零后的实际位姿记为本段起点（结束段返回目标 / `←` 重置目标）
3. **warmup**：从起点开始推理，让模型进入状态并攒够力历史
4. **录制**：按 `fps` 录帧，可随时 `h` 切人工遥控（遥控段也照录）
5. **结束**：`s`（成功）/ `f`（失败）/ `→`（无标注）→ 结果写入 `<数据集>/meta/results.jsonl`
6. **返回**：低速插值回本段起点，进入下一段

```powershell
$env:LEROBOT_POLICY_JOINT_DELTA = "0.1"   # 提速 (默认 0.025 很慢; 0.1≈3rad/s, 0.2≈6rad/s)

# —— 基础：模型引导录制 ——
D:\miniconda\envs\lerobot\python.exe scripts/record_ws_dataset.py `
  --server ws://10.96.45.13:8000 --task "Pump bottle" `
  --repo-id my_robot/Pump_bottle `
  --num-episodes 5 --episode-time-s 120 --reset-time-s 10 `
  --fps 30 --num-action-steps 30 --show-camera `
  --joint-limits-file joint_limits_pump_bottle.json

# —— 全功能：启动即遥控 + 记录力预测 + value 标注 + 力消融 ——
D:\miniconda\envs\lerobot\python.exe scripts/record_ws_dataset.py `
  --server ws://10.96.45.13:8000 --task "Pump bottle" `
  --repo-id my_robot/Pump_bottle_eef `
  --num-episodes 5 --episode-time-s 120 --reset-time-s 10 `
  --fps 30 --num-action-steps 30 --show-camera `
  --start-teleop `
  --record-force-pred `
  --record-value `
  --zero-force-to-server `
  --joint-limits-file joint_limits_pump_bottle.json

# —— 续录（补条数 / 删轨迹后补齐）——
D:\miniconda\envs\lerobot\python.exe scripts/record_ws_dataset.py `
  --server ws://10.96.45.13:8000 --task "Erase Board" `
  --repo-id test_board/Erase_Board_FORCE_EEF --num-episodes 1 --resume `
  --episode-time-s 120 --reset-time-s 10 --warmup-time-s 5 `
  --fps 30 --num-action-steps 30 --show-camera `
  --start-teleop --record-force-pred --record-value
```

**参数表**：

| 参数 | 默认 | 含义 |
|------|:--:|------|
| `--repo-id` | 必填 | 数据集名，如 `my_robot/erase_board_ws` |
| `--server` | `ws://127.0.0.1:8000` | 推理 server（`serve_policy.py`） |
| `--no-server` | — | 纯本地手控，不连 server、不推理（**必须配 `--start-teleop`**） |
| `--task` | `Erase Board` | 任务 prompt |
| `--num-episodes` | 10 | 段数 |
| `--episode-time-s` | 60 | 单段最长时长（到点自动结束，`none` 标注） |
| `--reset-time-s` | 10 | 段间重置时长 |
| `--warmup-time-s` | 5 | 段前推理预热时长 |
| `--fps` | 30 | 录制/执行帧率 |
| `--num-action-steps` | 30 | 每次推理后执行的 chunk 步数（1~30；越大推理越省、响应越钝） |
| `--max-joint-delta` | 0.2 | 单步关节最大变化（rad） |
| `--max-gripper-delta` | 0.02 | 单步夹爪最大变化 |
| `--joint-limits-file` | 空 | 数据集绝对关节限幅 JSON（`extract_joint_limits.py` 生成，**按任务分开**） |
| `--start-teleop` | — | 启动即人工遥控（默认推理）；按 `h` 随时切换 |
| `--record-force-pred` | — | 记录模型预测力 `observation.force_pred [6]`（推理帧=预测，遥控/返回=0） |
| `--record-value` | — | 记录 `v` 键切换的 `observation.value [1] int64`（每帧 0/1） |
| `--zero-force-to-server` | — | 发给 server 的力全 0（输入消融）；数据集仍记录本地真实力 |
| `--no-force` | — | 同 `--zero-force-to-server`（等价写法） |
| `--video` | — | 直接录 mp4（需 ffmpeg） |
| `--show-camera` | — | 实时显示摄像头画面 |
| `--resume` | — | 续录已有数据集 |

热键：见 §7.2。

**关节限幅 JSON 按任务生成**（`total_2task`：ep0-49 = Erase Board，ep50-99 = Pump bottle）：

```powershell
D:\miniconda\envs\lerobot\python.exe scripts/utils/extract_joint_limits.py my_robot/total_2task --out joint_limits_erase_board.json --ep-range 0-49
D:\miniconda\envs\lerobot\python.exe scripts/utils/extract_joint_limits.py my_robot/total_2task --out joint_limits_pump_bottle.json --ep-range 50-99
```

---

### 5.4 模式 C：纯手控录制（不连 server）

用于**模型不可用 / 想要纯人工对照数据**，但走的仍是 `record_ws_dataset.py` 的数据管线（自动归零、`results.jsonl`、力记录、`--record-value`）：

```powershell
D:\miniconda\envs\lerobot\python.exe scripts/record_ws_dataset.py `
  --no-server --start-teleop --task "Erase Board" `
  --repo-id my_robot/erase_board_manual `
  --num-episodes 10 --episode-time-s 120 --reset-time-s 10 `
  --fps 30 --show-camera --record-value
```

⚠️ `--no-server` 时按 `h` 切推理会被**拒绝**（没有可用的 server），只能全程遥控。

---

### 5.5 录制后处理

```powershell
# 1) 提取视频: 未加 --video 录的数据集 (图像 PNG 内嵌在 parquet) → 合成 mp4
#    输出 <repo>_videos/camera_one_ep000000.mp4 等; --out 指定目录, --camera one/two 单选
D:\miniconda\envs\lerobot\python.exe scripts/extract_parquet_videos.py fvt/Pump_bottle_Force_Joint

# 2) 一键转换 flexiv 兼容格式 (force -> state 13维 + 图像嵌入 + float32)
D:\miniconda\envs\lerobot\python.exe scripts/finalize_dataset.py my_robot/stamp_seal
D:\miniconda\envs\lerobot\python.exe scripts/finalize_dataset.py my_robot/erase_board
D:\miniconda\envs\lerobot\python.exe scripts/finalize_dataset.py my_robot/erase_board_ws
D:\miniconda\envs\lerobot\python.exe scripts/finalize_dataset.py my_robot/stamp_seal --float64   # float64 精度
D:\miniconda\envs\lerobot\python.exe scripts/finalize_dataset.py my_robot/stamp_seal --no-embed  # 不嵌入图像 (省空间)

# 3) 删坏轨迹 + 补录 (详见 §13)
D:\miniconda\envs\lerobot\python.exe scripts/remove_episode.py `
  test_board/Erase_Board_FORCE_EEF --index 2 --backup "H:\a-test\_backup\erase_board_force_eef"

# 4) 统计 + 力曲线 (详见 §12)
D:\miniconda\envs\lerobot\python.exe scripts/summarize_test_datasets.py --root "H:\a-test\test" --exclude Pump_bottle_eval_dataset
```

输出：`{repo}_flexiv`（如 `stamp_seal_flexiv`），原始数据不动。

---

## 6. 工作流 C：推理验证（远端 server + 本地 client）

### 6A. 旧 client（ZMQ 5555，当前帧力，1-token 模型）

**远端 server（SSH 过去）：**
```bash
ssh sfy@10.96.45.13
cd /mnt/hdd/sfy/openpi-force
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src python zmq_force_server.py \
    --checkpoint checkpoints/26000 --port 5555
# 杀旧 server: kill -9 $(pgrep -f zmq_force_server)
```

**本地 client：**
```powershell
# 旧任务 stamp seal
D:\miniconda\envs\lerobot\python.exe lerobot/scripts/piper_force_client.py `
  --task "stamp seal" --server tcp://10.96.45.13:5555 `
  --exec-steps 30 --no-clamp --show-camera `
  --log-dir "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\client_logs" `
  --log-name token1_stamp

# 新任务 Erase Board (白板擦, 2w ckpt)
D:\miniconda\envs\lerobot\python.exe lerobot/scripts/piper_force_client.py `
  --task "Erase Board" --server tcp://10.96.45.13:5555 `
  --fps 30 --num-action-steps 30 --no-clamp --show-camera `
  --log-dir "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\client_logs" `
  --log-name eraseBoard_2w

# 消融: 去掉力输入 (force 全 0)
D:\miniconda\envs\lerobot\python.exe lerobot/scripts/piper_force_client.py `
  --task "Erase Board" --server tcp://10.96.45.13:5555 `
  --exec-steps 30 --no-clamp --show-camera --no-force `
  --log-dir "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\client_logs" `
  --log-name eraseBoard_2w_noForce
```

**参数说明：**

| 参数 | 默认 | 含义 |
|------|:--:|------|
| `--exec-steps` | 25 | 每次从 30-chunk 取用的动作数 |
| `--num-action-steps` | 0(=exec-steps) | 每次推理后执行的 chunk 步数 |
| `--fps` | 30 | 执行帧率（固定节奏） |
| `--no-clamp` | — | 跳过大增量限幅 |
| `--no-force` | — | 消融：不发送力（server 自动填 0） |
| `--show-camera` | — | 弹出两个摄像头实时画面 |
| `--log-dir` | 空 | 数据日志目录（JSONL 含 state/action/force/force_pred） |
| `--log-name` | client_log | 日志文件名前缀，便于区分模型 |
| `--duration-s` | 120 | 最长运行时长，到点自动退出（无键盘热键，Ctrl-C 可中断） |

### 6B. 新 client（websocket 8000，60 帧力历史，lora/16k）

**远端 server（SSH 过去）：**
```bash
ssh sfy@10.96.45.13
cd /mnt/hdd/sfy/FA-OPENPI
kill -9 $(pgrep -f serve_policy) 2>/dev/null
python serve_policy.py --checkpoint <ckpt路径> --port 8000
```

**本地 client：**
```powershell
# 完整 30 步执行 (推理 ~1Hz)
D:\miniconda\envs\lerobot\python.exe scripts/piper_ws_force_client.py `
  --server ws://10.96.45.13:8000 --task "Erase Board" `
  --fps 30 --num-action-steps 30 `
  --show-camera --log-name lora_Erase_Board `
  --log-dir "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\client_logs"

# 高频推理 + 15 步执行 (推理 2Hz)
D:\miniconda\envs\lerobot\python.exe scripts/piper_ws_force_client.py `
  --server ws://10.96.45.13:8000 --task "Erase Board" `
  --fps 30 --num-action-steps 15 `
  --show-camera --log-name lora_Erase_Board_15 `
  --log-dir "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\client_logs"

# 夹爪执行放大 (gripper-scale 1.2: 反馈÷1.2, 执行×1.2)
D:\miniconda\envs\lerobot\python.exe scripts/piper_ws_force_client.py `
  --server ws://10.96.45.13:8000 --task "Erase Board" `
  --fps 30 --num-action-steps 30 `
  --show-camera --gripper-scale 1.2 `
  --log-name eef_2k5_scale `
  --log-dir "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\client_logs"

# 遥控开始 → F键切推理 (热切换)
D:\miniconda\envs\lerobot\python.exe scripts/piper_ws_force_client.py `
  --server ws://10.96.45.13:8000 --task "Erase Board" `
  --fps 30 --num-action-steps 30 `
  --show-camera --gripper-scale 1.2 --start-teleop `
  --log-name eef_2k5_toggle `
  --log-dir "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\client_logs"
```

**参数说明：**

| 参数 | 默认 | 含义 |
|------|:--:|------|
| `--fps` | 30 | 执行频率（send_action 频率, Hz） |
| `--num-action-steps` | 1 | 每次推理后执行的 chunk 步数（推理率 = fps/该值） |
| `--infer-fps` | 0 | 推理频率上限（0=不限制） |
| `--max-joint-delta` | 0.2 | 单步关节最大变化（rad） |
| `--max-gripper-delta` | 0.02 | 单步夹爪最大变化 |
| `--gripper-scale` | 1.0 | 夹爪执行增益（1.2 = 反馈÷1.2, 执行×1.2） |
| `--q4-max-pos` | 0 | 关节4（q4）正向绝对位置上限（度），0=不限制 |
| `--start-teleop` | — | 启动即遥控，按 `f` 切推理 |
| `--duration-s-teleop` | 0 | 遥控段最长时长（0=不限，按 `f` 回推理） |
| `--no-force-history` | — | 不攒 60 帧历史，只发当前帧力（测试用） |
| `--no-mark` | — | 禁用 `→` 键 mark 标注 |
| `--save-images` | — | 每帧把画面 JPEG base64 嵌入日志（日志会很大） |
| `--joint-limits-file` | 空 | 关节绝对限幅 JSON |
| `--show-camera` | — | 弹出两个摄像头实时画面 |
| `--log-dir` / `--log-name` | 空 / client_log | 日志目录 / 文件名前缀 |

**两个 client 核心区别：**

| | 旧 client (ZMQ 5555) | 新 client (websocket 8000) |
|------|---------------------|---------------------------|
| 力输入 | 当前帧 6 维力 | 60 帧力历史 `[60,6]` |
| server | `zmq_force_server.py` | `serve_policy.py` |
| 模型 | openpi-force (1-token) | FA-OPENPI (lora/full 16k) |
| 任务 | stamp seal / Erase Board | stamp seal / Erase Board |
| 键盘热键 | 无 | `f` `→` `v` `ESC` |

---

## 7. 全局热键总表

### 7.1 `lerobot/scripts/control_robot.py`（`teleoperate` / `record`）

键盘（**pynput 全局监听**，终端不在前台也生效）：

| 键 | 功能 | 来源 | 生效范围 |
|----|------|------|---------|
| `→` | `exit_early`：提前结束当前阶段（录制段 / 重置段） | `control_utils.init_keyboard_listener` | 仅 `record` |
| `←` | `rerecord_episode`：丢弃并重录上一条轨迹 | 同上 | 仅 `record` |
| `ESC` | `stop_recording`：结束整个录制流程并保存 | 同上 | 仅 `record` |
| `q` | `exit_early`（UArm / SO101 领航臂控制器自带） | `gamepad._UArmLeaderController._on_key_press` | `teleoperate` + `record` |

手柄（PS5，仅当遥控源是手柄时；`control_loop` 每帧调用 `robot.consume_control_events()`）：

| 按钮 | 功能 |
|------|------|
| `RB` | `exit_early`：提前结束当前阶段 |
| `LB` | `rerecord_episode`：重录上一条（同时触发 `exit_early`；`teleoperate` 下无实际效果） |
| `Start` | 切换 关节模式 ↔ 末端位姿（EE）模式（需 IK 可用） |
| `Y` | 归零：6 关节置 0 + 夹爪置 0 |
| 左摇杆 | 关节模式：关节移动 / EE 模式：平移 |
| 右摇杆 | 关节模式：关节移动 / EE 模式：旋转 |
| 十字键 `↑↓←→` | 步进微调 |
| `LT` / `RT` | 夹爪 −/+（每帧 ±0.001，范围 [0, 0.1]） |
| `Ctrl-C` | 强制退出（任何模式） |

### 7.2 `scripts/record_ws_dataset.py`（模型引导录制）

键盘（**msvcrt 控制台输入 → 该终端窗口必须在焦点**）：

| 键 | 功能 |
|----|------|
| `s` | 标注 **success** 并结束本段（写 `meta/results.jsonl`） |
| `f` | 标注 **fail** 并结束本段 |
| `→` | 无标注结束（`none`） |
| `←` | 重置：丢弃本段重录（回本段起点） |
| `h` | 切换 **人工遥控 ↔ 模型推理**（`--no-server` 时切推理会被拒绝） |
| `v` / `V` | 切换 `value` 标注 0/1（需 `--record-value`，大小写都认） |
| `ESC` | 完成并退出（保存数据集） |

> ⚠️ `s`、`f`、`h` **必须小写**（别按 Shift）：msvcrt 读到 `S`/`F`/`H` 不会匹配。`v` 兼容大小写；`ESC` 无此问题。
> 该脚本**不消费手柄事件**：手柄 `RB`/`LB`/`q` 在这里无效，结束/标注只能用上面的键。

### 7.3 `scripts/piper_ws_force_client.py`（新推理 client）

| 键 | 功能 |
|----|------|
| `f` | 切换 **遥控 ↔ 推理** |
| `→` | 切换 mark（倾斜标注）0/1，写日志 `mark` 字段 |
| `v` / `V` | 切换 `value` 标注 0/1，写日志 `value` 字段 |
| `ESC` | 退出 |
| `↑` `↓` `Space` | **预留未绑定**（脚本读取到事件但无处理分支） |

> 同样用 msvcrt（窗口需聚焦）；`f` 需小写。

### 7.4 `scripts/test/test_force_viz.py`（力传感器归零）

matplotlib 窗口热键（**窗口需聚焦**）：

| 键 | 功能 |
|----|------|
| `b` | 基础归零（无工具，100 帧平均） |
| `t` | 工具归零（装好工具后，100 帧平均） |
| `z` | 清除工具偏置 |
| `r` | 清空历史曲线 |

### 7.5 手柄按键映射（PS5，供参考）

```python
SONY_BUTTON_MAP = {"a":0, "b":1, "x":3, "y":2, "lb":4, "rb":5,
                   "lt":6, "rt":7, "back":8, "start":9, "home":10, "l3":11, "r3":12}
```

---

## 8. 力传感器归零

```powershell
D:\miniconda\envs\lerobot\python.exe scripts/test/test_force_viz.py
# 图窗热键: [b] 基础归零  [t] 工具归零  [z] 清除工具偏置  [r] 清空历史
```

---

## 9. 关键环境变量

| 变量 | 值 | 作用 |
|------|-----|------|
| `LEROBOT_PIPER_INTERP_SLEEP_S` | `0` | ★ 消除 24ms 插值延迟（必须为 0） |
| `LEROBOT_PIPER_INTERP_STEPS` | `5` | 插值步数 |
| `LEROBOT_PIPER_MOVE_SPEED` | `100` | 执行速度 1-100（归位脚本内部自动降到 40） |
| `LEROBOT_CAN_BUSTYPE` | `gs_usb` | Windows CAN 驱动（Linux 用 `socketcan`） |
| `LEROBOT_CAN_CHANNEL` | `0` | CAN 设备索引（Linux 为 `can0`） |
| `LEROBOT_CAN_BITRATE` | `1000000` | CAN 波特率 |
| `LEROBOT_UARM_PORT` | `COM5` | UArm 串口（需按实际确认） |
| `LEROBOT_PIPER_TELEOP_SOURCE` | `uarm` | 遥控源：`auto` / `uarm` / `so101`(`leader`,`so101_leader`) / `gamepad` |
| `LEROBOT_PIPER_STARTUP_ZERO_HOLD_S` | `0.4` | 上电归零保持时长（秒） |
| `LEROBOT_PIPER_STARTUP_GRIPPER_OPEN` | `0.1` | 上电归零后夹爪开度（归一化上限 0.1） |
| `LEROBOT_PIPER_TELEOP_GRIPPER_GAIN` | `1.0` | 遥控夹爪增益（仅遥控，不影响推理） |
| `LEROBOT_PIPER_CAM_ONE_INDEX` | `3` | 主相机 OpenCV 索引 |
| `LEROBOT_PIPER_CAM_TWO_SN` | `339222071125` | 腕相机 RealSense 序列号 |
| `LEROBOT_OPENCV_BACKEND` | `MSMF` | 摄像头后端：`MSMF` / `DSHOW` / `ANY` |
| `LEROBOT_FORCE_CAL_PATH` | `.cal` | ATI Mini40 标定文件；**不设置 = 不用力传感器** |
| `LEROBOT_FORCE_DEVICE` | `Dev1` | NI-DAQmx 设备名 |
| `LEROBOT_FORCE_CHANNELS` | `ai0:5` | 6 路模拟输入通道 |
| `LEROBOT_POLICY_JOINT_DELTA` | `0.025` | 推理每步关节硬限幅（rad，调大加速） |
| `LEROBOT_POLICY_GRIPPER_DELTA` | `0.005` | 推理每步夹爪硬限幅 |

---

## 10. 无传感器模式（拆掉 ATI / 不想用 DAQ 时）

**原理**：`robots/piper.py` 里力传感器可用性由 `LEROBOT_FORCE_CAL_PATH` 决定——**不设置它 → `force_sensor=None` → 自动跳过连接/读取**，遥控与普通录制完全不受影响，**无需改代码**。

```powershell
# 新终端里【不要设置】$env:LEROBOT_FORCE_CAL_PATH (其余环境变量照常)，然后跑 §4 / §5.2 的命令
```

⚠️ 注意：

- 此模式录的数据集**没有 `observation.force` 特征**（`force_features` 返回空），力曲线/力统计不适用
- **`record_ws_dataset.py`（模式 B/C）依赖力传感器**（录帧要读 `observation.force`，无传感器会 `KeyError`）——无力传感器请改用 §5.2 的 `control_robot.py record`
- `LEROBOT_FORCE_DEVICE` / `LEROBOT_FORCE_CHANNELS` 一并忽略

---

## 11. 数据集路径与日志目录

- 数据集：`C:\Users\User\.cache\huggingface\lerobot\{repo_id}\`（部分已迁到 `H:\a-test\test`）
- 客户端日志 `client_logs/` 按 日期/前缀 组织：

```
client_logs/
├── 20260803/
│   ├── token1_openpiforce42000_zmq30steps/
│   └── ...
├── 20260806/
│   └── eraseBoard_2w/
└── ...
```

---

## 12. 数据集统计与可视化 (`summarize_test_datasets.py`)

对所有数据集统一统计：**成功率**（全部）+ **力预测质量**（带 force 的数据集）+ **每条轨迹曲线图**。
后续新增数据集后**重跑一次脚本即可自动纳入**，无需改代码。

### 12.1 运行

```powershell
# 数据集当前位于 H:\a-test\test (原 C:\Users\User\.cache\huggingface\lerobot\test 已迁移; 共8个可统计数据集 + eval)
D:\miniconda\envs\lerobot\python.exe scripts/summarize_test_datasets.py --root "H:\a-test\test" --exclude Pump_bottle_eval_dataset

# 扫其他目录 (如 cache 下新录的数据)
D:\miniconda\envs\lerobot\python.exe scripts/summarize_test_datasets.py --root my_robot
```

**参数：**

| 参数 | 默认 | 含义 |
|------|:--:|------|
| `--root` | `test` | 数据集根目录（扫描其下所有数据集） |
| `--eps` | 0.1 | 百分比分母真值阈值（\|true\|>eps 才计入，防除零） |
| `--clip-pct` | 99.0 | 内部按 P99 分位过滤单点爆炸后取均值（不打印） |
| `--no-plot` | — | 不画曲线图 |
| `--plot-dir` | 空 | 曲线输出目录（默认每个数据集自身 `force_curves/`） |
| `--smooth` | 15 | 曲线滑动平均窗口（帧数；0=不平滑） |
| `--exclude` | 空 | 跳过指定数据集（逗号分隔），如 `Pump_bottle_eval_dataset` |
| `--out` | 空 | 汇总 json 路径（默认 `<root>/SUMMARY.json`） |

### 12.2 输出

| 输出 | 位置 |
|------|------|
| 控制台表格 + 汇总 | `test/SUMMARY.json` |
| 每个数据集结果 | `<数据集>/meta/result_summary.json` |
| 曲线图（每轨迹一张） | `<数据集>/force_curves/ep000000.png` |

### 12.3 指标含义

| 指标 | 含义 |
|------|------|
| 成功率% | success / (success + fail)（来自 `meta/results.jsonl`） |
| maxΔ(N) | 每帧 max(\|pred − true\|)，预测与真实的最大差值（N/Nm） |
| delta%均值 | mean( \|Δpred − Δtrue\| / \|Δtrue\| × 100 )，Δx[t] = x[t] − x[t−1] 相邻帧力变化量 |
| abs%均值 | mean( \|\|pred\| − \|true\|\| / \|true\| × 100 )，先各自取绝对值再比 |
| 曲线图 | 蓝=真实力，橙虚线=预测力（仅带力数据集；PI05 只画真实力），绿底=推理段（pred 非零）；滑动平均平滑（窗口 15）；标题带单位：Fx/Fy/Fz 为 **N**，Tx/Ty/Tz 为 **N·m** |

### 12.4 重要约定

- **force_pred 语义**：模型预测的**绝对力**（与 `observation.force` 同量级，实测 -12.7~+14N vs 真实 -11~+13N，**不是 delta 变化量**）
- **只统计推理帧**（pred 非零）；遥控/返回段 pred=0 是补零，不算预测
- **PI05 数据集**（模型不带力）`force_pred` 全 0 → 跳过力指标（表格显示 `-`），但仍出**只画真实力**的曲线图
- **force 曲线图**需要 `--record-force-pred` 录制的数据集（pred+真实）；不带 force_pred 的旧数据集没有曲线
- 改成功率标注：直接编辑 `<数据集>/meta/results.jsonl`（`{"episode_index": N, "result": "success"/"fail"/"none"}`），再重跑脚本

---

## 13. 删除坏轨迹并重新录制 (remove_episode.py)

```powershell
# 1) 删除坏轨迹 (例: ep2; 后续 3~9 自动前移为 2~8, 数据集变 9 条)
#    v2 原子流程 + 必带 --backup 备份到独立目录
D:\miniconda\envs\lerobot\python.exe scripts/remove_episode.py `
  test_board/Erase_Board_FORCE_EEF --index 2 --backup "H:\a-test\_backup\erase_board_force_eef"

# 2) 重新录制 1 条补齐 (resume 从 total_episodes 继续, 新轨迹成为最后一条)
D:\miniconda\envs\lerobot\python.exe scripts/record_ws_dataset.py `
  --server ws://10.96.45.13:8000 --task "Erase Board" `
  --repo-id test_board/Erase_Board_FORCE_EEF --num-episodes 1 --resume `
  --episode-time-s 120 --reset-time-s 10 --warmup-time-s 5 `
  --fps 30 --num-action-steps 30 --show-camera `
  --start-teleop --record-force-pred --record-value
```

---

## 14. 项目文件清单

### 核心修改（lerobot 源码）

| 文件 | 修改 |
|------|------|
| `lerobot/common/robot_devices/utils.py` | `busy_wait` Windows 分支 (`time.sleep(0)` 自旋, 保 30Hz) |
| `lerobot/common/robot_devices/motors/piper.py` | gs_usb CAN 适配 + `interp_sleep_s` + `move_speed`/`set_move_speed` |
| `lerobot/common/robot_devices/cameras/opencv.py` | `LEROBOT_OPENCV_BACKEND` (MSMF/DSHOW/ANY) |
| `lerobot/common/robot_devices/robots/configs.py` | CAN 通道 / 相机索引 / RealSense 序列号 走环境变量 |
| `lerobot/common/robot_devices/sensors/force_sensor.py` | ATI Mini40 驱动 + 连续采样 + drain |
| `lerobot/common/robot_devices/robots/piper.py` | 力传感器集成 + `force_features` + 策略限幅/夹爪参数化 |
| `lerobot/common/robot_devices/teleop/gamepad.py` | Windows COMx 串口修复 + 夹爪行程 0.08→0.1 |
| `lerobot/common/datasets/utils.py` | `get_features_from_robot` 合并 `force_features` |

### 工具脚本

| 文件 | 功能 |
|------|------|
| `lerobot/scripts/piper_force_client.py` | **力感知推理客户端 (旧, ZMQ)** 发送 image+state+force |
| `scripts/piper_ws_force_client.py` | **力感知推理客户端 (新, websocket 60 帧力历史)** |
| `scripts/record_ws_dataset.py` | 16-token 模型引导数据集录制（§5.3，含 `--no-server` 纯手控模式） |
| `scripts/remove_episode.py` | **删除坏轨迹并前移补位**（v2 原子版，必带 `--backup`） |
| `scripts/extract_parquet_videos.py` | 从 parquet 内嵌 PNG 提取合成 mp4 |
| `scripts/utils/extract_joint_limits.py` | 按数据集/轨迹段生成关节限幅 JSON |
| `scripts/compute_start_pose.py` | 计算标准初始位（每 ep 前 N 帧 state 平均） |
| `scripts/preflight_check.py` | 启动前设备预检（CAN/Camera/DAQ/UArm/ffmpeg） |
| `scripts/finalize_dataset.py` | 一键转换 flexiv 兼容格式 |
| `scripts/summarize_test_datasets.py` | 数据集成功率 + 力预测质量 + 曲线图（§12） |
| `scripts/utils/piper_fk.py` | Piper 正运动学（URDF 提取） |
| `scripts/utils/browse_ep2_impact.py` | 交互式浏览器：视频 + EEF + 力数据对齐 |
| `scripts/force/play_force.py` | 力+视频播放器：左视频 右6轴柱状图 |
| `scripts/force/visualize_force_video.py` | 6轴柱状图 + 视频同步播放器（旧版） |
| `scripts/checks/check_alignment.py` | 批量数据对齐检查 |
| `scripts/test/test_force_viz.py` | 力传感器实时可视化 + 归零 |

---

## 15. ZMQ 推理协议（Server 适配参考）

**推理客户端：** `lerobot/scripts/piper_force_client.py`

### Client → Server 请求

```
消息类型: zmq.REQ/REP, 序列化: pyobj (pickle)

{type: "reset"}                    # 重置策略状态

{type: "infer",
 task: "stamp seal",               # 任务描述
 state: [j1..j7],                  # 7维 Piper 关节+夹爪 (float list)
 force: [Fx,Fy,Fz,Tx,Ty,Tz],       # 6轴力 (float list, 可选)
 images: {
   "observation.images.camera0": ndarray,  # 640×480×3 RGB uint8
   "observation.images.camera1": ndarray,  # 640×480×3 RGB uint8
 }}
```

### Server → Client 响应

```
{status: "ok"}                     # reset 响应

{status: "ok",
 action: [[...], [...], ...]}      # 30×7 完整 action chunk (绝对关节位置)
                                    # 30步预测 @ 30fps
```

### 适配要点

| 项目 | 说明 |
|------|------|
| state 维度 | 7（Piper 关节 1-6 + 夹爪） |
| force 维度 | 6（Fx,Fy,Fz,Tx,Ty,Tz, N/Nm） |
| 图像格式 | numpy ndarray, HWC, RGB, uint8 |
| 图像 key | `observation.images.camera0`, `observation.images.camera1` |
| action 格式 | 30×7, 绝对关节位置, 首帧=当前帧的预测 |
| 端口 | 默认 5555 |
| 客户端文件 | `lerobot/scripts/piper_force_client.py` |

---

## 16. 数据安全铁律

⚠️（2026-08-20 教训）删除/覆盖任何用户数据前**必须先备份 + 用户确认**；批量改名必须用临时名中间态（新旧文件名重叠的"先写新再删旧"曾误删 6 条轨迹）。

- 删轨迹一律用 `scripts/remove_episode.py <repo_id> --index N --backup <目录>`（v2 原子版，`--backup` 必填）
- 数据集 parquet / `meta/*.jsonl` / 录制数据 **不许直接删/覆盖/移动**；要清理先移动到 `H:\a-test\_backup\`
