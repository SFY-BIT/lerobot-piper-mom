# FORCE/ — 力传感器（ATI Mini40）本机文件占位

> 本目录**不入库**（见根目录 `.gitignore`）：内容为第三方厂商软件与本机标定文件，
> 体积大且含厂商版权，仓库里只保留这个说明文件。

## 本机应放什么

```
FORCE/
└── FORCE/
    ├── 1001 DAQ/          # ATI DAQ 软件/驱动与 C 库（ATIDAQ）
    ├── FT29363/           # 各传感器序列号对应的标定文件
    ├── FT32663/           #   └─ 其中 *.cal 为 XML 格式，含 6x6 UserAxis 标定矩阵
    └── FT36232/
```

## 为什么需要它

`lerobot/common/robot_devices/sensors/force_sensor.py` 通过 **NI-DAQmx + `.cal` 标定矩阵**
把 NI USB-6341 采集到的 6 路电压换算成 `Fx, Fy, Fz, Tx, Ty, Tz`（N / N·m）。
没有 `.cal` 文件就无法标定，力数据不可用。

## 怎么用

启动前把标定文件路径指给环境变量（不设置 = 完全不使用力传感器，遥控/普通录制不受影响）：

```powershell
$env:LEROBOT_FORCE_CAL_PATH = "C:\Users\User\Desktop\lerobot\lerobot-piper-mom\FORCE\FORCE\FT32663\FT32663.cal"
$env:LEROBOT_FORCE_DEVICE   = "Dev1"      # NI-DAQmx 设备名 (nilsdev 可查)
$env:LEROBOT_FORCE_CHANNELS = "ai0:5"
```

详细流程见 `win_migration/RUN.md`（§2 环境变量、§8 力传感器归零、§10 无传感器模式）。

## 获取方式

- ATI DAQ / 驱动与 C 库：ATI 官网（需按传感器型号下载对应版本）
- 传感器标定文件（`FT*.cal`）：随机附带的 U 盘/光盘，或在 ATI 官网按序列号下载

> ⚠️ 这些文件（尤其安装包与厂商库）请勿提交到仓库。
