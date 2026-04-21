# Phase1：几何模块 + 玩具 CL +（可选）HF 最小步

## 0. 端到端烟雾（推荐先做）

**无网络、无 HF**：一键跑几何 + Concat + toy CL + Phase0：

```powershell
Set-Location d:\cursor_try\Evidence
py -3 experiments\phase1\run_phase1_smoke_all.py
```

含 HF 最小步（需依赖与缓存/网络；脚本内仍可 `SKIP`）：

```powershell
$env:EVIDENCE_HF_SMOKE="1"
py -3 experiments\phase1\run_phase1_smoke_all.py
```

真数据 + 绑图见下文；**锚点联合几何 + 判别训练** 见 `run_joint_geometry_cl.py`。服务器清单见 **`docs/SERVER_RUN.md`**。

### 锚点联合 \(V_k\) + 双任务（判别主线）

在任务 0 上训练若干步作为 **锚点**，用 **B 个 batch 梯度矩阵 + r 次 Hvp** 提取 `extract_joint_vk_gamma`，再在后段训练并写 JSONL（`vtm_lora` 等，支持交替任务或仅 task1）。与 `run_hf_real_two_task_cl.py` 中 **随机 \(V\)** 的调试管线区分：本脚本用于 **「是否值得推进」** 的初步实验。

```powershell
py -3 experiments\phase1\run_joint_geometry_cl.py --dataset ag_news --anchor-steps 200 --post-steps 400 --B-grad 64 --r-sub 16 --tau 80 --device cuda
py -3 experiments\phase1\run_joint_geometry_cl.py --adamw-lora  # 同上超参加此开关作对照
# 推荐加 holdout + 周期性 eval（JSONL 内 kind:eval，与 CORE 遗忘 proxy 对齐）：
# py -3 experiments\phase1\run_joint_geometry_cl.py ... --max-per-class 200 --holdout-per-class 40 --eval-every 20
# 若要更贴近顺序CL遗忘压力：再加 --post-task-mode task1_only（默认 alternate）
# 机制消融：SubGeo 可切 --subgeo-mode sym（默认 asym；--adamw-lora 时忽略该参数）
```

## 1. 几何验证（仅 torch）

验证 `src/subgeo/geometry.py`：Hvp 与二次型解析 Hessian 一致、`V^T H V` 与 autograd 投影一致、`G` 子空间 QR 正交、`build_gamma`。

```powershell
Set-Location d:\cursor_try\Evidence
py -3 experiments\phase1\verify_geometry.py
```

或使用 conda（若控制台 GBK 报错，优先用上面的 `py -3`）：

```powershell
conda run -n base --cwd "d:\cursor_try\Evidence" python experiments\phase1\verify_geometry.py
```

## 2. 玩具两任务 CL

见 `run_toy_cl.py`：锚点、`V` 与 B 梯度正交、vanilla vs asym。

```powershell
py -3 experiments\phase1\run_toy_cl.py
```

## 3. DistilGPT2 + LoRA 最小步（可选）

需安装：

```powershell
py -3 -m pip install -r experiments\phase1\requirements_hf.txt
py -3 experiments\phase1\train_distilgpt2_minimal.py
```

### Hugging Face 镜像

脚本在导入 `transformers` 前会配置 `huggingface_hub` 使用的 **`HF_ENDPOINT`**：

| 情况 | 行为 |
|------|------|
| 已设置 `HF_ENDPOINT` 或 `HUGGINGFACE_HUB_ENDPOINT` | 不修改，直连你指定的 Hub |
| 未设置且未关闭镜像 | 默认 `https://hf-mirror.com`（可按需改为其他兼容 Hub API 的镜像） |
| `EVIDENCE_HF_MIRROR=0` | 不设置 `HF_ENDPOINT`，使用官方 `huggingface.co` |

自定义镜像基址（仍须未手动设置 `HF_ENDPOINT` 时才会写入）：

```powershell
$env:EVIDENCE_HF_ENDPOINT="https://hf-mirror.com"
py -3 experiments\phase1\train_distilgpt2_minimal.py
```

强制走官方（不经过本脚本的默认镜像）：

```powershell
$env:EVIDENCE_HF_MIRROR="0"
py -3 experiments\phase1\train_distilgpt2_minimal.py
```

无网络或 SSL 失败时会打印 `SKIP` 并以 **exit 0** 结束（不阻断 CI）。有本地 HF 缓存或修好证书后会拉权重并跑 3 个优化步。

镜像逻辑复用模块：`experiments/phase1/hf_hub_endpoint.py`（`train_distilgpt2_minimal.py` 与 `run_hf_two_task_cl.py` 均通过其 `configure_hf_hub_endpoint()` 在导入 transformers 前生效）。

## 4. HF 双任务交替 + JSONL 日志（可选）

`run_hf_two_task_cl.py`：与最小步相同的 DistilGPT2+LoRA，按步 **task 0 / 1 交替**（不同随机 `input_ids`、标签全 0 / 全 1）；默认 **ConcatSubGeoAdam** 训 LoRA，**AdamW** 训非 LoRA；每步写入 **JSONL**（首行为 meta，随后每行含 `step`、`task`、`loss`、`vtm_lora`，其中 `vtm_lora` 为固定随机子空间 `V` 上的 `||V^T m||`）。`--adamw-lora` 时 LoRA 也走 AdamW，仍用同一 `V` 记录 `vtm_lora` 便于对照。

```powershell
py -3 experiments\phase1\run_hf_two_task_cl.py --steps 40
py -3 experiments\phase1\run_hf_two_task_cl.py --adamw-lora --steps 40
```

默认日志目录：`experiments/phase1/logs/`（已加入该目录 `.gitignore`）。`--log path.jsonl` 可指定路径。

### 真数据双任务（AGNews / DBpedia-14，推荐）

统一入口 **`run_hf_real_two_task_cl.py`**（`real_two_task_data.py`）：任务 0 = 原始类 0 vs 1，任务 1 = 原始类 2、3 映射为二分类 0/1；文本列在 AGNews 为 `text`，DBpedia 为 `content`。需 **`datasets`**。

```powershell
py -3 -m pip install -r experiments\phase1\requirements_hf.txt
py -3 experiments\phase1\run_hf_real_two_task_cl.py --dataset ag_news --run-both --steps 120 --max-per-class 250 --batch-size 8 --max-length 128 --tau 100
py -3 experiments\phase1\run_hf_real_two_task_cl.py --dataset dbpedia_14 --steps 80 --max-per-class 200
```

- **`--tau`**：`make_weak_pd_operators` 的分母（默认 `1e9` 接近恒等）；**调小**（如 `50`–`200`）可加强子空间阻尼，便于与 AdamW 区分。  
- **`--holdout-per-class` + `--eval-every`**：每类划出 holdout，每隔 N 步写一行 **`{"kind":"eval","eval_loss_task0":...,"eval_loss_task1":...}`**（遗忘 proxy：看任务切换后 holdout loss 是否漂移）。  
- **`--run-both`**：SubGeo 与 AdamW 各写 `logs/<dataset>_real_subgeo_*.jsonl` / `_real_adamw_*.jsonl`。  
- **`run_agnews_two_task_cl.py`** 仍为兼容入口（等价默认 `--dataset ag_news`）。

日志摘要（仅标准库）：

```powershell
py -3 experiments\phase1\summarize_hf_real_cl_log.py experiments\phase1\logs\某文件.jsonl
```

绑图（需 `requirements_plot.txt`）：

```powershell
py -3 -m pip install -r experiments\phase1\requirements_plot.txt
py -3 experiments\phase1\plot_hf_real_cl_log.py experiments\phase1\logs\subgeo.jsonl experiments\phase1\logs\adamw.jsonl --out experiments\phase1\logs\compare.png --labels SubGeo AdamW
```

## 5. ConcatSubGeoAdam（多 LoRA 张量拼接空间）

`src/subgeo/optimizer.py`：`lora_trainable_parameters(model)` 按 **参数名排序** 收集 `lora_*` 可训张量；`ConcatSubGeoAdam` 将各张量梯度按同一顺序拼接为长度 `D` 的向量，再施加与 Phase0 相同的 `P_d`（`mode=asym` 时 `g_m = P_d g`、`g_v = g`）。`make_weak_pd_operators(D, k, ...)` 用于烟雾：随机正交列 `V` + 弱 `gamma`（大 `tau` 时接近恒等）。

自检（不依赖 HF）：

```powershell
Set-Location d:\cursor_try\Evidence
py -3 experiments\phase1\test_concat_optimizer.py
```

## 6. 代码入口

| 模块 | 路径 |
|------|------|
| 展平梯度、SVD 子空间、投影 Hessian、`build_gamma`、`apply_pd` | `src/subgeo/geometry.py` |
| ConcatSubGeoAdam、`concat_subgeo_m_flat`、`momentum_energy_in_subspace`、LoRA 参数收集 | `src/subgeo/optimizer.py` |
| HF Hub 镜像（`HF_ENDPOINT`） | `experiments/phase1/hf_hub_endpoint.py` |
| DistilGPT2+LoRA 分类加载 | `experiments/phase1/hf_lora_model.py` |
| AdamW 动量拼接（日志） | `experiments/phase1/hf_metrics.py` |
| 真数据双任务池（AGNews/DBpedia）+ tokenize | `experiments/phase1/real_two_task_data.py` |
| 锚点联合提取 \(V_k,\gamma\) | `src/subgeo/joint_geometry.py` |
| 锚点 + 联合几何 + 双任务训练 | `experiments/phase1/run_joint_geometry_cl.py` |
| 真数据双任务训练（含 tau / eval；随机 V 可调管线） | `experiments/phase1/run_hf_real_two_task_cl.py` |
| JSONL 摘要 | `experiments/phase1/summarize_hf_real_cl_log.py` |
| 绑图（train loss + eval） | `experiments/phase1/plot_hf_real_cl_log.py` + `requirements_plot.txt` |
| 一键 Phase1 烟雾 | `experiments/phase1/run_phase1_smoke_all.py` |
| AGNews 兼容封装 | `experiments/phase1/agnews_data.py`（→ `real_two_task_data`） |
| SubGeoAdam 优化器（Phase0 单张量） | `smoke/subgeo_optimizer.py` |

**注意**：若阻尼方向 `V` 与 **当前任务梯度** 共线且 `γ→1`，则 `P_d g≈0` 会停滞；真实管线用 Joint 提取的 `V_k`，勿退化为「每步沿 g 强阻尼」。
