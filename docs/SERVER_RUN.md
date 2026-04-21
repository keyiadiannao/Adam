# 服务器实验准备与运行清单

**目标**：在 GPU 节点上复现「**锚点联合几何 + 双任务**」与后续 Phase1/2，支撑 **是否值得推进** 的判据（见 `CORE_SubGeo_Research_Narrative.md` §4）。

## 1. 环境与依赖

- Python 3.10+，CUDA 与 PyTorch 版本写入 `runs/<id>/env.txt`（`python -V`、`pip freeze` 头 80 行）。
- 安装：`pip install -r experiments/phase1/requirements_hf.txt`；绑图另装 `requirements_plot.txt`。
- Hugging Face：默认镜像见 `experiments/phase1/hf_hub_endpoint.py`；官方 Hub 则 `EVIDENCE_HF_MIRROR=0`。
- 建议设置：`HF_HOME` 或 `TRANSFORMERS_CACHE` 指向节点本地高速盘；`TOKENIZERS_PARALLELISM=false`。

## 2. Git：本地仓库已与服务器对齐方式

**本地（`d:\cursor_try\Evidence`）**：已执行 `git init`、根目录 `.gitignore`、首提交与远程 **`https://github.com/keyiadiannao/Adam`**（默认分支 `main`）。之后改代码：

```powershell
Set-Location d:\cursor_try\Evidence
git add -A
git status
git commit -m "描述本次改动"
```

**第一次推到远程（任选 GitHub / Gitee / 实验室 Git）**：

1. 在网页上新建 **空仓库**（不要勾选自动添加 README，避免冲突）。
2. 本地添加远程并推送（HTTPS 示例，把 URL 换成你的）：

```powershell
git remote add origin https://github.com/keyiadiannao/Adam.git
git branch -M main
git push -u origin main
```

若远程默认仍想用 `master`：可省略 `git branch -M main`，改为 `git push -u origin master`。若本机已存在 `origin`：`git remote set-url origin https://github.com/keyiadiannao/Adam.git`。

**AutoDL 上拉代码**：

```bash
mkdir -p /root/autodl-tmp/work && cd /root/autodl-tmp/work
git clone https://github.com/keyiadiannao/Adam.git
cd Adam
mkdir -p runs && git rev-parse HEAD > runs/manual_git_sha.txt
```

私有仓库：在 AutoDL 上配置 **SSH 公钥** 或 **HTTPS Token**（勿把密码写在脚本里）。

**GitHub `git pull` / `clone` 报 TLS / GnuTLS（如 -110）**：多为到 `github.com` 的链路不稳。可依次尝试：

1. 重试几次；或 `git config --global http.postBuffer 524288000` 后再 `git pull`。
2. **改用 SSH 远程**（控制台已绑公钥）：`git remote set-url origin git@github.com:keyiadiannao/Adam.git && git pull`。
3. **单文件 zip（可行，仅换代码时）**：用 `wget`/`curl` 拉 **`main` 分支源码包**（仍是 GitHub，若仍 TLS 失败需换能访问的镜像或本机下载后 scp 上传）：
   ```bash
   cd /root/autodl-tmp/work
   wget -O Adam-main.zip https://github.com/keyiadiannao/Adam/archive/refs/heads/main.zip
   unzip -o Adam-main.zip && rm -rf Adam && mv Adam-main Adam && cd Adam
   ```
   说明：zip **不含 `.git`**，之后无法用 `git pull` 更新，只适合救急；长期仍建议修好 Git 或镜像后再 `clone`。
4. 将仓库 **同步到 Gitee / 实验室 Git** 再在节点上 `clone` 该地址（国内常更稳）。

固定随机种子：`--seed` 及 PyTorch `manual_seed_all`（若脚本未全设，在提交作业前补一行）。

**依赖装好后先跑一键校验**（与 §3-A 等价，便于复制）：

```bash
cd /path/to/Adam   # 你的 clone 目录
source .venv/bin/activate
bash scripts/dev_verify.sh
```

## 3. 门闸命令（由轻到重）

**A. 无 GPU 核心烟雾（验收机器与代码）**

```bash
python experiments/phase1/run_phase1_smoke_all.py
```

**B. HF 最小（需权重缓存或网络）**

```bash
export EVIDENCE_HF_SMOKE=1   # 可选
python experiments/phase1/run_phase1_smoke_all.py
```

**C. 真数据 + 随机 V（管线与 JSONL）**

```bash
python experiments/phase1/run_hf_real_two_task_cl.py --dataset ag_news --run-both \
  --steps 200 --max-per-class 200 --holdout-per-class 40 --eval-every 20 --tau 100
```

**D. 锚点联合几何 + 双任务（判别主线的最低实现）**

```bash
python experiments/phase1/run_joint_geometry_cl.py --dataset ag_news \
  --anchor-steps 200 --post-steps 400 --B-grad 64 --r-sub 16 --tau 80 \
  --batch-size 8 --max-length 128 --device cuda --save-vk runs/ag_news_vk.pt
```

对照 AdamW LoRA：在 **同一组超参** 下再加 `--adamw-lora` 重跑一遍，便于对比 JSONL。

**E. 绑图（下载 JSONL 到本机后）**

```bash
pip install -r experiments/phase1/requirements_plot.txt
python experiments/phase1/plot_hf_real_cl_log.py subgeo.jsonl adamw.jsonl --out pareto_loss.png --labels SubGeo AdamW
```

## 4. 建议目录布局

```
runs/
  <YYYYMMDD_expname>/
    config.json          # 超参、数据集、git sha
    env.txt
    *.jsonl
    *.pt                 # V_k 等检查点
    figures/
```

## 5. Slurm 示例（按需改分区/GPU）

```bash
#!/bin/bash
#SBATCH --job-name=subgeo_p1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o runs/%x-%j.out

cd /path/to/Adam
python experiments/phase1/run_joint_geometry_cl.py --device cuda \
  --dataset ag_news --anchor-steps 200 --post-steps 400 --B-grad 64 --r-sub 16
```

## 6. 过线自检（与 CORE §4 对齐）

- [ ] 同数据、同种子下保存 **(5) vs (4) 或 (1)** 的 JSONL；
- [ ] holdout eval（若启用）与 `eval_loss_task0` 趋势可解释；
- [ ] `runs/` 中可复现 `git` 与 `pip` 信息。

---

**维护**：与 `GeoAdam_Experiment_Plan.md` 里程碑同步更新；新增脚本时在本文 §3 追加一行示例命令。
