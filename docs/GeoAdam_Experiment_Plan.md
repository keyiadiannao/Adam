# SubGeo-Adam 实验计划（落地版 v2.0）

**方法定义**见 [SubGeo_Adam_Technical_Report.md](./SubGeo_Adam_Technical_Report.md)。**叙事与判据**见 [CORE_SubGeo_Research_Narrative.md](./CORE_SubGeo_Research_Narrative.md)。**服务器命令**见 [SERVER_RUN.md](./SERVER_RUN.md)。

本文：实验阶段、基线、算力、里程碑。

**算力**：单卡 48GB；冻结底座 + LoRA；必要时 QLoRA。

---

## 1. 研究问题与过线

| RQ | 内容 | 最低判据 |
|----|------|----------|
| RQ1 | 非对称 g_m=P_d g, g_v=g | Asym 明显优于 Sym；Reverse 更差 |
| RQ2 | Joint 相对单信号 | Joint 不低于 Hessian-only/Grad-only；报告 V_grad 与 Hessian-top 夹角（锚点+动态段） |
| RQ3 | 预算淘汰 | 序列长度>=5 时 K_max 优于无预算堆叠 |
| RQ4 | 对强基线 | Pareto 上对 2～3 个关键基线有稳定优势区 |

## 2. 方法落地（与代码对齐）

- 联合几何：B>=max(2r,64) 收集 G；r 次 Hvp 得 H_proj；V_k=V_grad Q；gamma 见技术报告。
- 成本：统一 T_full；阶段一约 B+r 次。
- 非对称：仅 LoRA；其余 AdamW。
- 效用：U_i=R_i+w C_i；在线每 T 步更新；淘汰 argmin U_i。

## 3. 阶段

- **Phase0**：smoke/run_smoke.py。
- **Phase1**：distilgpt2 + LoRA；run_joint_geometry_cl.py（主判别）；run_hf_real_two_task_cl.py（管线）。
- **Phase2**：7B 或 QLoRA；多任务主表。

## 4. 数据协议

固定种子与任务顺序；每任务固定步数；锚点提取后落盘；新任务 LoRA 重初始化或热启动（正文固定一种）。

## 5. P0 基线

B0 Vanilla；B1 强 WD；B2 EWC；B3 O-LoRA；B4 SOAP；B6 CODE-CL；B7 SGP；B5 Ours。

## 6. 消融映射

机制：Asym/Sym/Reverse/Grad-Damp。信号：Hessian-only、Grad-only、Joint。预算：无预算、K_max+淘汰、K_max+效用。

## 7. 复现

固定版本与 yaml、git、命令行；存 V_k、gamma、tau、rho、w。

## 8. 里程碑（12 周可压缩）

W0-1 Phase0+1；W2-4 消融+B0-B2+B6/B7 其一；W5-8 Phase2 主表；W9-10 敏感性与夹角图；W11-12 初稿。

## 9. 风险

7B OOM 转 QLoRA；SOAP 难复现则简化；Joint 不优则调 B、r、w。

## 10. 分工

A 优化器；B 几何；C 基线；D 编排；E 写作。

## 11. 默认超参

r=16, alpha=32；beta2 0.95 或 0.999；tau 从 lambda 分位扫；B>=max(2r,64)。

## 12. 文档维护

本文件 UTF-8 无 BOM 为真源。若从 Python 模板生成 Markdown，须用 raw 字符串避免转义破坏公式。

---

**版本**：v2.0（2026-04-21）
