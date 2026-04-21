# Project Archive Summary (2026-04-21)

## 1) 归档结论

- 项目主线（`SubGeo-Adam`/`Geo-Adam` 在当前设置下）进入 **No-Go 归档状态**。
- 判定依据：多轮多种子实验下，核心对照（`AdamW` 基线）未被稳定超越，`holdout` 指标方向不一致，无法支撑“稳定增益”主张。
- 决策：停止继续在该主线上投入算力与调参；保留代码和实验资产用于复盘与后续方法迁移参考。

## 2) 判定范围（本次归档覆盖）

- 数据/任务：`ag_news` 双任务协议（含 `task1_only` 压力测试）。
- 主要脚本：`experiments/phase1/run_joint_geometry_cl.py`。
- 关键设置：`B_grad=64, r_sub=16, anchor_steps=200, post_steps=400`，并测试 `tau=40/80/120/200`、`asym/sym`、`holdout eval`。
- 最后一轮止损实验：`adamw + geo_reg` vs `adamw`（同协议同种子）。

## 3) 关键观察（归档原因）

- 结果对随机种子敏感，`de0/de1`（holdout 差值）跨种子符号反复翻转。
- `asym` 与 `sym` 均未形成稳定优势带，`tau` 扫描未改变结论。
- `adamw + geo_reg` 作为“降级转向”验证，仍未达到稳定优于基线的门槛。

## 4) 代码与文档资产（保留）

- 几何与优化器实现：
  - `src/subgeo/geometry.py`
  - `src/subgeo/joint_geometry.py`
  - `src/subgeo/optimizer.py`
- Phase1 实验入口与可视化：
  - `experiments/phase1/run_joint_geometry_cl.py`
  - `experiments/phase1/run_hf_real_two_task_cl.py`
  - `experiments/phase1/plot_hf_real_cl_log.py`
- 汇总与执行脚本：
  - `scripts/summarize_joint_geom_jsonl.py`
  - `scripts/dev_verify.sh`
  - `scripts/dev_verify.ps1`
- 主文档：
  - `docs/CORE_SubGeo_Research_Narrative.md`
  - `docs/SubGeo_Adam_Technical_Report.md`
  - `docs/SERVER_RUN.md`

## 5) 复盘建议（不属于当前项目继续）

- 如果未来重启该方向，建议以新项目分支进行，避免污染本归档结论：
  1. 先完成 `Vanilla / L2 / EWC` 灵魂基线，确认遗忘痛点规模；
  2. 仅保留一个最小新方法（Loss 正则或梯度混合），设置固定止损门槛；
  3. 若仍无稳定优势，停止该方向并转题。

## 6) 归档执行清单

- [x] 结论确认：No-Go（主线停止）
- [x] 文档收口：本文件 + 主文档状态标记
- [ ] 结果文件打包（服务器 logs/runs 到长期存储）
- [ ] 仓库打标签（如 `archive-2026-04-subgeo-adam`，可选）
- [ ] 新方向立项（新仓库或新目录，避免混淆）

---

维护说明：本文件用于“为什么归档、归档了什么、未来怎么复盘”的单点入口，不再记录过程性实验日志。
