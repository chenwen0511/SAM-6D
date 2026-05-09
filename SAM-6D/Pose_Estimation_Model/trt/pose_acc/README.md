# PEM 位姿估计优化记录（pose_acc）

本目录用于归档 **Pose Estimation Model（PEM）** 在推理链路优化过程中的阶段性结论：分项耗时、精度对照思路、以及与上层 `trt/` 基线工作的衔接。

更完整的 TensorRT 与基线脚本说明见上一级：[`../README.md`](../README.md)。

---

## 1. 分项耗时快照（`run_pose_inference` 内 `_log`，单位 ms）

说明：**creating model …** 在 PEM 权重缓存命中时开销很小，未单独打点；**saving results …**（写 JSON）未单独打点。以下为同一接口下的阶段耗时。

### 1.1 优化前（每次推理冷读模板目录）

| 阶段 | 日志关键字 | 耗时 (ms) | 备注 |
|------|------------|-----------:|------|
| 模板读取与预处理 | `get_templates elapsed_ms` | 234.625 | 多视角 png/npy 读盘 + OpenCV/NumPy + 逐视角 `.cuda()` |
| 模板 ViT 特征 | `template feature_extraction.get_obj_feats elapsed_ms` | 120.421 | `feature_extraction.get_obj_feats` |
| 观测数据与 CAD | `get_test_data elapsed_ms` | 89.699 | RGB/深度/相机/分割/CAD 采样与实例构造 |
| 网络前向 | `model.forward elapsed_ms` | 30.845 | `model(input_data)` |
| 可视化（若开启） | `visualize elapsed_ms` | 126.892 | `save_visualization=True` 时 |

与可视化相关的 **五段相加** 约 **602.5 ms**（仅用于理解瓶颈分布）；端到端 **`pose_s`** 仍以 HTTP 包裹的整段 `run_pose_inference` 为准。

### 1.2 优化后（服务启动预加载模板 GPU 张量，`/infer` 命中缓存）

实现要点：`warmup_http_service.py` 在 **startup** 中对当前 `SAM6D_CAD_PATH` 对应的模板目录调用 `preload_pem_templates_gpu_cache`；推理时出现 **`get_templates (gpu cache hit, skipped disk reload)`**，不再重复读盘与组装模板张量。

典型日志（本次 **`POST /infer`** 返回 200，未包含可视化计时行，推断服务端关闭了 `save_visualization` 或未进入该分支）：

| 阶段 | 日志关键字 | 耗时 (ms) | 备注 |
|------|------------|-----------:|------|
| 模板（缓存命中） | `get_templates elapsed_ms` | **0.002** | 日志含 `get_templates (gpu cache hit, skipped disk reload)` |
| 模板 ViT 特征 | `template feature_extraction.get_obj_feats elapsed_ms` | 120.808 | 与优化前同量级 |
| 观测数据与 CAD | `get_test_data elapsed_ms` | 90.054 | 与优化前同量级 |
| 网络前向 | `model.forward elapsed_ms` | 30.346 | 与优化前同量级 |

**相对 §1.1**：`get_templates` 约 **234.6 ms → ~0 ms（缓存命中）**，其余阶段基本一致。环境变量：`SAM6D_PEM_PRELOAD_TEMPLATES`（默认开启）、可选 `SAM6D_PEM_CONFIG_PATH` / `SAM6D_RD_SEED` 须与推理配置一致方可命中缓存。

---

## 2. 优化优先级（更新）

1. **模板侧**：磁盘侧 **`get_templates`** 已通过启动预加载 + 进程内 GPU 缓存削峰；进一步可关注 **`get_obj_feats`（ViT）** TRT、或减少 **`n_template_view`**（需做精度回归）。
2. **观测侧**：`get_test_data` 仍为 CPU/IO 大头之一，可关注 CAD/深度/RLE 与重复计算缓存。
3. **核心 `forward`**：绝对数值已较小，TRT 需与导出成本权衡。
4. **可视化**：服务端仅需 JSON 时保持 **`save_visualization=False`**，避免百毫秒级绘图写盘。

---

## 3. 精度与基线对照

端到端时延与 pose 数值基线：`trt/baseline.py`（`baseline.csv` / `*_results.json`）。随机种子与 **`score` / `xyzrxryrz`** 稳定性见 `trt/README.md`。模板缓存不改变数值路径（与同一 `rd_seed` + 同一模板目录 + 同一 `test_dataset` 关键字段下的冷读结果应对齐）。

---

## 4. 变更日志（手动维护）

| 日期 | 说明 |
|------|------|
| （待填） | §1.1 冷读模板分项快照 |
| （待填） | §1.2 启动预加载模板 GPU 缓存后 `/infer` 分项快照 |
