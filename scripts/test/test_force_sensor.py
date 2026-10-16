"""测试：读取 FT32663 标定矩阵并采集一次力传感器数据"""
import xml.etree.ElementTree as ET
import numpy as np
import nidaqmx

CAL_PATH = r"C:\Users\User\Desktop\lerobot\lerobot-piper-mom\FORCE\FORCE\FT32663\FT32663.cal"

# 1. 解析标定矩阵
tree = ET.parse(CAL_PATH)
root = tree.getroot()
print(f"Sensor: {root.get('Serial')} ({root.get('BodyStyle')})")
cal = root.find("Calibration")
print(f"Part: {cal.get('PartNumber')}, CalDate: {cal.get('CalDate')}")
print(f"Units: {cal.get('ForceUnits')} / {cal.get('TorqueUnits')}")
print(f"OutputRange: ±{cal.get('OutputRange')}V, Bipolar: {cal.get('OutputBipolar')}")
print()

# 用 UserAxis 构建 6x6 标定矩阵（注释中明确说 "To parse this file, use the UserAxis elements"）
axes = root.findall(".//UserAxis")
cal_matrix = np.zeros((6, 6))
for i, axis in enumerate(axes):
    values = [float(v) for v in axis.get("values").split()]
    cal_matrix[i, :] = values
    print(f"  {axis.get('Name'):4s}  [{', '.join(f'{v:10.5f}' for v in values)}]  max={axis.get('max')}")

print(f"\n标定矩阵 (6x6):\n{np.round(cal_matrix, 4)}")
print(f"cond: {np.linalg.cond(cal_matrix):.1f}")

# 2. 尝试采集一次
print("\n--- 采集测试 ---")
try:
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(
        "Dev1/ai0:5", min_val=-10.0, max_val=10.0
    )
    voltages = np.array(task.read(number_of_samples_per_channel=1))
    print(f"Raw voltages: {np.round(voltages, 6)} V")
    ft = cal_matrix @ voltages
    print(f"Force/Torque:  Fx={ft[0]:.4f}N  Fy={ft[1]:.4f}N  Fz={ft[2]:.4f}N  "
          f"Tx={ft[3]:.4f}Nm  Ty={ft[4]:.4f}Nm  Tz={ft[5]:.4f}Nm")
    task.close()
    print("采集正常！")
except Exception as e:
    print(f"采集失败: {e}")
    print("（如果传感器没接线，这是正常的）")
