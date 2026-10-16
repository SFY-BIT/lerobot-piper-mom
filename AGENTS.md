# Project Guidelines

## 数据安全（最高优先级）

1. **禁止直接删除/覆盖/移动任何非本会话 agent 创建的用户文件**（数据集 parquet、`meta/*.jsonl`、录制数据等）。需要删除时：先移动到备份目录（如 `H:\a-test\_backup\`）或让用户自己操作。
2. **不可逆操作前先备份 + 用户确认**：删除、覆盖、重建数据集、改 meta 一律先备份。
3. **批量文件改名必须原子化**：先唯一临时名 → 处理 → 全部成功后才删临时；新/旧文件名不得重叠。
4. 用户真实数据优先"移动备份"，其次"复制"，最后才删除。

## 数据集操作约定

- 删除坏轨迹：`scripts/remove_episode.py <repo_id> --index N --backup <目录>`（v2 原子版，必须带 `--backup`）
- 统计可视化：`scripts/summarize_test_datasets.py --root "H:\a-test\test" --exclude Pump_bottle_eval_dataset`
- 删除脚本 v1 曾因"新旧文件名重叠 + 先写新再删旧"误删 6 条轨迹（2026-08-20），严禁复现该模式
