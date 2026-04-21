# 下一步推进计划（简版）

## 文档真源（各司其职）

| 文档 | 职责 |
|------|------|
| `docs/CORE_SubGeo_Research_Narrative.md` | 问题—方法—判据（定稿叙事） |
| `docs/SubGeo_Adam_Technical_Report.md` | 符号、公式、实现与消融 ID |
| `docs/GeoAdam_Experiment_Plan.md` | 阶段、基线矩阵、里程碑、算力 |
| `docs/SERVER_RUN.md` | 服务器环境、命令、产物目录 |

## 近期交付状态

| 顺序 | 交付物 | 状态 |
|------|--------|------|
| 1 | `run_toy_cl.py` | 已完成 |
| 2 | `geometry.py` + `verify_geometry.py` | 已完成 |
| 3 | `run_hf_real_two_task_cl.py`（真数据、JSONL、`--tau`/`--run-both`） | 已完成 |
| 3b | holdout eval + `summarize_hf_real_cl_log.py` + `plot_hf_real_cl_log.py` | 已完成 |
| **4** | **`joint_geometry.py` + `run_joint_geometry_cl.py`**（锚点处真实 \(V_k\) + 双任务） | **已完成（判别主线可跑）** |
| 5 | 服务器：按 `SERVER_RUN.md` 跑 **(5) vs (4)/(1)** 多种子、保存 JSONL/`*.pt` | **进行中（上机）** |
| 6 | 正式消融表与论文级图表 | 待 Phase1 数据回填 |

## 端到端烟雾（层级 A～D）

| 层级 | 命令 / 条件 | 说明 |
|------|----------------|------|
| A | `py -3 experiments\phase1\run_phase1_smoke_all.py` | 无网络：几何 + Concat + toy + Phase0 |
| B | `EVIDENCE_HF_SMOKE=1` 同上 | 含 HF 最小步 |
| C | `run_hf_real_two_task_cl.py` | 真数据管线（可用随机 \(V\) 调试） |
| D | `requirements_plot.txt` + `plot_hf_real_cl_log.py` | 绑图 |

**判别「是否值得推进」**：以 **`run_joint_geometry_cl.py`**（联合 \(V_k\)）在服务器上相对 **`--adamw-lora`** 与消融 **(4)** 的 JSONL / Pareto 为准；见 `CORE_SubGeo_Research_Narrative.md` §4。

## 何时上 7B

在 **Joint 判别脚本** 与日志格式在 Phase1 稳定后，再开 QLoRA 7B（单卡 48GB 预算）。

## 环境与命令

```powershell
conda run -n base --cwd "d:\cursor_try\Evidence" python experiments\phase1\run_toy_cl.py
```
