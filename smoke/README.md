# Phase0 烟雾测试（SubGeo-Adam）

## 环境

本机已验证：`py -3`（Python 3.11）且已安装 `torch` 即可，无需单独 conda（若你用 conda，在已装 torch 的环境里同样运行）。

```powershell
cd d:\cursor_try\Evidence
py -3 smoke\run_smoke.py
```

若缺少 torch：

```powershell
py -3 -m pip install -r smoke\requirements.txt
```

## 测什么

| 门控 | 含义 |
|------|------|
| `apply_pd` | 与手写 `g - V(γ⊙V^T g)` 数值一致 |
| 第一步 | `gamma` 非零时 **sym ≠ asym** 的第一步更新 |
| 弱阻尼 | `tau` 极大使 `γ≈0`，**asym / sym / grad_damp** 终 loss 与 **vanilla** 相对误差 < 5% |
| reverse | 中等阻尼下 **reverse** 与 **asym** 的 **‖V^T m‖** 终值不同（实现可分路） |

**不测什么**：不在此脚本里要求「asym 优于 sym」——在纯凸二次型上 **强阻尼** 会妨碍沿主曲率方向下降，劣于 vanilla 是正常现象；强弱对比留给后续真实 CL 实验。

## 与文档对应

见 `docs/SubGeo_Adam_Technical_Report.md` §10 Phase0。
