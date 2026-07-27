# SAM-6D 推理加速过程记录

本文记录从最初冷启动到当前热路径的性能优化历程，以及后续可继续挖的点。  
入口服务以 `warmup_http_service.py` 为准；端到端耗时字段为 HTTP 返回中的 `total_s`。

---

## 总览

| 阶段 | 大致耗时 | 相对基线 |
|------|----------|----------|
| 初始（冷启动 + 命令行调用） | ~**10 s** | 1× |
| API 直调 + PEM 权重预加载 | 位姿约 **0.6 s** | ~16× |
| + 41 模板 GPU 预加载 | 位姿约 **0.4 s** | ~25× |
| + PEM ViT（rgb_net）TensorRT | ViT 前向 **0.26 s → 0.16 s** | — |
| **当前端到端** | **`total_s` ≈ 0.774 s** | ~13×（含分割等） |

说明：前几步数字主要对应**位姿解算（PEM）**路径；`total_s` 还包含分割后端、IO 等，因此会高于纯 `pose_s`。

---

## 阶段 0：基线（~10 s）

最初每次推理大致是：

1. 服务侧通过 **cmd / 子进程**拉起推理脚本；
2. 每次重新加载 **`sam-6d-pem-base.pth`（约 1.6G）**；
3. 现场读模板、提特征、再跑 PEM。

冷启动成本（进程创建 + 大权重加载）占主导，单次约 **10 s**。

---

## 阶段 1：调用方式 + 权重预加载 → 位姿 ~0.6 s

做了两点，位姿解算时间压到约 **0.6 s**：

### 1. 摒弃 cmd，改为进程内函数 API

- 不再每次 `subprocess` / 命令行拉起 PEM；
- 改为直接调用 `run_warmup_inference_custom.run_pose_inference(...)`。

收益：去掉进程启动、Python 解释器冷启动、重复 import 等固定开销。

### 2. 预加载 `sam-6d-pem-base.pth`

- 启动时把约 **1.6G** 的 PEM 权重载入进程并缓存（`_CACHED_DEFAULT_MODEL`）；
- 环境变量：`SAM6D_PEM_PRELOAD_DEFAULT=1`（默认开启）；
- 实现：`preload_default_pem_model()`（`run_warmup_inference_custom.py`）。

收益：请求路径不再重复 `load_checkpoint`，位姿阶段从「秒级加载」变成「百毫秒级前向」。

此时端到端仍偏高（例如 `total_s` ≈ **0.774 s**），位姿已不是唯一瓶颈，但 PEM 前向仍是大头之一。

---

## 阶段 2：41 模板预加载 → 位姿 ~0.4 s

工程优化：把 CAD 渲染得到的 **41 个视角模板**在服务启动时预加载到 **GPU 缓存**。

- 环境变量：`SAM6D_PEM_PRELOAD_TEMPLATES=1`（默认开启）；
- 实现：`preload_pem_templates_gpu_cache()` + `warmup_http_service` startup hook；
- 命中缓存后日志可见 `get_templates (gpu cache hit)`。

效果：位姿耗时由约 **0.6 s → 0.4 s**（去掉每次请求的模板磁盘读取与上送 GPU）。

---

## 阶段 3：PEM ViT（rgb_net）TensorRT → 0.26 s → 0.16 s

对位姿估计中的 **ViT / `rgb_net`（ViT_AE）dense 特征**做 TRT FP16 引擎加速：

- 导出与说明：`Pose_Estimation_Model/trt/pose_acc/vit_acc.md`、`Pose_Estimation_Model/trt/README.md`；
- 运行时开关：`SAM6D_PEM_RGB_TRT_ENGINE=<path/to/*.engine>`；
- 接入点：`feature_extraction.py` → `PemRgbNetTrt.forward_dense()`。

效果：该子模块推理由约 **0.26 s → 0.16 s**。  
注意：当前是 **分阶段 TRT**（只加速可稳定导出的图像 ViT 子图），点云 / 几何匹配等仍走 PyTorch，整网并未一键 TRT。

---

## 当前状态（小结）

已落地：

| 优化项 | 状态 |
|--------|------|
| PEM 进程内 API（非 cmd） | ✅ |
| PEM 1.6G 权重预加载 | ✅ |
| 41 模板 GPU 预加载 | ✅ |
| PEM ViT TensorRT FP16 | ✅（子图） |
| YOLO 进程内 + 可选 `.engine` | ✅（分割后端之一） |

仍偏慢 / 未做完：

| 方向 | 说明 |
|------|------|
| 服务里残留的 cmd / 子进程 | 如 `seg_backend=sam6d_ism`、`seg_backend=sam3` 仍走 `subprocess`；SAM3 每次冷启加载，模型内 `_MODEL_CACHE` 跨请求用不上 |
| PEM 更完整的 TensorRT | 位姿大头仍在 PEM；ViT 已加速，其余分支（几何匹配、pointnet2 等）还可继续拆子图或优化 |
| SAM3 常驻 / 进程内调用 | 外部已有 `run_server.py` 与 BF16，本仓库尚未接入 |

当前观测的端到端量级：**`total_s` ≈ 0.774 s**（相对最初 ~10 s 已有数量级提升），仍有压缩空间。

---

## 后续优先项（建议）

1. **清掉服务路径上剩余的 cmd 调用**  
   - ISM / SAM3：改为进程内 API，或常驻 HTTP（SAM3 可复用外部 `run_server.py`），避免每次冷加载。
2. **继续做位姿 TensorRT**  
   - 在现有 ViT engine 基础上，按 `trt/README.md` 的「分阶段」策略扩展可导出子图；用 `trt/baseline.py` 做时延与 `xyzrxryrz` 对齐。
3. **压测与拆分计时**  
   - 区分 `sam3_s` / `yolo_s` / `pose_s` / `total_s`，确认下一刀砍在分割还是 PEM。

---

## 相关代码与文档

| 路径 | 作用 |
|------|------|
| `warmup_http_service.py` | HTTP 入口、startup 预加载、分段计时 |
| `Pose_Estimation_Model/run_warmup_inference_custom.py` | PEM API、权重/模板预加载 |
| `Pose_Estimation_Model/trt/` | PEM TRT 指南与工具 |
| `Pose_Estimation_Model/trt/pose_acc/vit_acc.md` | ViT rgb_net TRT 导出备忘 |
| `yolo_seg_backend.py` | YOLO 进程内分割（可挂 TRT engine） |
| `sam3_seg_backend.py` | SAM3 子进程适配（待优化） |
| `docs/rest_api.md` | 接口说明 |
