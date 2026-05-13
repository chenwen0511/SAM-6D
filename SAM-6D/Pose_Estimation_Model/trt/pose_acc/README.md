# PEM 位姿估计优化记录（pose_acc）

本目录用于归档 **Pose Estimation Model（PEM）** 在推理链路优化过程中的阶段性结论：分项耗时、精度对照思路、以及与上层 `trt/` 基线工作的衔接。

更完整的 TensorRT 与基线脚本说明见上一级：[`../README.md`](../README.md)。**YOLO `best.engine` 在升级 TensorRT 后需重导** 见 **§5**；系统级 TRT / `trtexec` 安装备忘见 [`trtexec_install.md`](trtexec_install.md)。

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

## 4. `warmup_http_service` 启动与 `/infer` 实测（2026-05-13）

### 4.1 服务启动（shell）

```bash
export SAM6D_CAD_PATH=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/models/tray_180mm_centered_mesh_v2.ply
export SAM6D_OUTPUT_ROOT=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs
export CUDA_VISIBLE_DEVICES=0
cd /home/mui/projects/smt/SAM-6D/SAM-6D
python warmup_http_service.py --host 0.0.0.0 --port 8001
```

说明：若需 **YOLO 启动预加载**、**PEM 模板 GPU 预加载**，可另行设置 `SAM6D_YOLO_WEIGHTS`、`SAM6D_PEM_PRELOAD_TEMPLATES` 等（见 `warmup_http_service.py` 与 §1.2）。本次记录以用户实际环境为准。

### 4.2 请求（`yolo_seg` + TensorRT engine）

```bash
curl -X POST "http://127.0.0.1:8001/infer" \
  -F "rgb=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png" \
  -F "depth=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png" \
  -F "camera=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json" \
  -F "seg_backend=yolo_seg" \
  -F "yolo_weights=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine" \
  -F "yolo_conf=0.25" \
  -F "yolo_imgsz=640" \
  -F "yolo_class_id=0" \
  -F "det_score_thresh=0.00"
```

**注意**：`det_score_thresh=0.00` 会让 **所有** 超过几何过滤的 YOLO 实例进入 PEM，显存与耗时随实例数上升；生产环境建议 **≥0.25** 或与 `yolo_conf` 配合使用。

### 4.3 响应摘要（HTTP 200）

| 字段 | 值 |
|------|-----|
| `score` | `0.12119197100400925` |
| `xyz_mm` | `[-61.437, -131.235, 606.296]`（约 mm，浮点略截断） |
| `rotation_euler_zyx_rad` | `[2.977, 0.816, 1.273]`（弧度，ZYX） |
| `result_dir` | `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260513_100347_7af98c24` |

分割结果路径字段名仍为历史命名 **`detection_ism_path`**，在 `yolo_seg` 模式下指向本次生成的 **分割 JSON**（与 PEM 的 `detection_pem.json` 不同文件）。

### 4.4 `timing`（秒）

| 键 | 值 (s) | 说明 |
|----|--------:|------|
| `upload_s` | 0.000964 | 上传写盘 |
| `templates_s` | 0.000147 | 模板目录 symlink |
| `ism_s` | `null` | 未走 SAM ISM 子进程 |
| `yolo_s` | 0.017548 | YOLO 分割（含 TRT 路径） |
| `pose_s` | 0.244655 | 整段 `run_pose_inference`（含 `get_templates` / ViT / `forward` 等） |
| `pipeline_s` | 0.262351 | `templates_s + yolo_s + pose_s` |
| `total_s` | 0.263315 | `upload_s + pipeline_s` |

### 4.5 服务端控制台分项（与 HTTP `timing` 对照）

同一路径下，**`yolo_seg_backend` 打印**与 **`run_pose_inference` 内 `_log`**（需 `verbose=True`，例如环境变量 `SAM6D_PEM_VERBOSE=true`，且 HTTP 服务里已对 PEM 打开 verbose）典型一行请求如下。

**YOLO（TensorRT engine，缓存命中）**

| 日志 | 耗时 (ms) |
|------|-----------:|
| `_load_model elapsed_ms` | 0.054 |
| `model.predict elapsed_ms` | 16.104 |
| `load+predict elapsed_ms` | 16.164 |

**PEM（模板 GPU 缓存命中、无 `visualize` 行）**

| 日志 | 耗时 (ms) | 备注 |
|------|-----------:|------|
| `get_templates elapsed_ms` | 0.003 | 含 `get_templates (gpu cache hit, skipped disk reload)` |
| `template feature_extraction.get_obj_feats elapsed_ms` | 119.058 | ViT 模板特征 |
| `get_test_data elapsed_ms` | 89.603 | 观测 + CAD 等 |
| `model.forward elapsed_ms` | 34.591 | coarse / fine 等 |
| **上述 PEM 四段相加** | **≈243.3** | 与 **`pose_s`≈0.245 s** 同量级（另含建模型分支日志、`saving results` 等未打点部分） |

原始日志片段（便于检索）：

```
[yolo_seg_backend] load request: /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine (resolved: /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine)
[yolo_seg_backend] model cache hit: /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine
[yolo_seg_backend] _load_model elapsed_ms=0.054
[yolo_seg_backend] model.predict elapsed_ms=16.104
[yolo_seg_backend] load+predict elapsed_ms=16.164
set CUDA_VISIBLE_DEVICES as 0
=> creating model ...
=> extracting templates ...
=> get_templates (gpu cache hit, skipped disk reload)
=> get_templates elapsed_ms=0.003
=> template feature_extraction.get_obj_feats elapsed_ms=119.058
=> loading input data ...
=> get_test_data elapsed_ms=89.603
=> running model ...
=> model.forward elapsed_ms=34.591
=> saving results ...
INFO:     127.0.0.1:38326 - "POST /infer HTTP/1.1" 200 OK
```

**小结**：**§4.4** 的 `timing.yolo_s` / `timing.pose_s` 可与 **§4.5** 表中 **~16 ms**、**PEM 四段 ~243 ms** 对照；与 §1.1 旧表不宜逐行等同（无可视化、缓存与随机种子等差异）。PEM 详细日志依赖 **`SAM6D_PEM_VERBOSE`**（及 `warmup_http_service` 内传入 `run_pose_inference` 的 `verbose`）。

### 4.6 `/infer` 再测：开启 PEM `rgb_net` TensorRT 后（2026-05-13）

以下日志来自同一 **`warmup_http_service` + `yolo_seg` + `best.engine`** 路径；**`template feature_extraction.get_obj_feats elapsed_ms` 由约 119 ms 降至约 34 ms**，与在进程环境中设置 **`SAM6D_PEM_RGB_TRT_ENGINE=checkpoints/pem_rgb_net_b1_224_fp16.engine`**（**`ViTEncoder`** 走 **`PemRgbNetTrt`**，见 **`vit_acc.md` §9**）一致。**YOLO `model.predict` 本行约 75 ms**，与 **§4.5 的 ~16 ms** 可差数倍，多为 **Ultralytics 首次/偶发慢路径、GPU 竞争、输入尺寸与实例数** 等导致，**不宜单独与 PEM 子模块加速混为一谈**；对比 PEM 时请固定 **`get_obj_feats` / `forward`** 前后文。

**分项表（本组日志）**

| 模块 | 日志 | 耗时 (ms) |
|------|------|-----------:|
| YOLO | `_load_model elapsed_ms` | 0.062 |
| YOLO | `model.predict elapsed_ms` | 74.931 |
| YOLO | `load+predict elapsed_ms` | 74.999 |
| PEM | `get_templates elapsed_ms` | 0.004 |
| PEM | `template feature_extraction.get_obj_feats elapsed_ms` | **34.192** |
| PEM | `get_test_data elapsed_ms` | 89.565 |
| PEM | `model.forward elapsed_ms` | 35.867 |
| PEM 四段相加 | — | **≈159.6** |

**与 §4.5 同口径对照（量级）**

| 项 | §4.5（PyTorch rgb） | §4.6（本组，含 PEM rgb TRT） |
|----|---------------------|------------------------------|
| `get_obj_feats` | **119.058** | **34.192** |
| `get_test_data` | 89.603 | 89.565 |
| `model.forward` | 34.591 | 35.867 |
| YOLO `predict` | 16.104 | 74.931 |

原始日志片段：

```
[yolo_seg_backend] load request: /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine (resolved: /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine)
[yolo_seg_backend] model cache hit: /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine
[yolo_seg_backend] _load_model elapsed_ms=0.062
[yolo_seg_backend] model.predict elapsed_ms=74.931
[yolo_seg_backend] load+predict elapsed_ms=74.999
set CUDA_VISIBLE_DEVICES as 0
=> creating model ...
=> extracting templates ...
=> get_templates (gpu cache hit, skipped disk reload)
=> get_templates elapsed_ms=0.004
=> template feature_extraction.get_obj_feats elapsed_ms=34.192
=> loading input data ...
=> get_test_data elapsed_ms=89.565
=> running model ...
=> model.forward elapsed_ms=35.867
=> saving results ...
INFO:     127.0.0.1:50434 - "POST /infer HTTP/1.1" 200 OK
```

---

## 5. YOLO 分割 `best.engine` 因 TensorRT 环境对齐重新导出（2026-05-13）

在完成本目录 [`trtexec_install.md`](trtexec_install.md) 所述的 **系统级 TensorRT / Python `tensorrt` 版本对齐**（日志中出现 **`TensorRT 10.16.1.11`**）后，**旧版 `best.engine` 与当前运行时 TensorRT 可能不兼容**，需用 **`best.pt` 重新 `yolo export`** 生成新 engine，服务与 CLI 再指向新文件。

### 5.1 备份与导出命令（已在本机验证）

工作目录：`user_data/yolo_runs/tray_seg/weights`

```bash
mv best.engine best.engine.bak
# 将路径换为你本机 best.pt 所在目录
yolo export model=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.pt \
  format=engine half=True workspace=8
```

说明：日志中 **`TensorRT requires GPU export, automatically assigning device=0`** 为 Ultralytics 提示，无需再手写 `device=0`（可加 `device=0` 显式指定）。

### 5.2 本次导出关键信息摘要

| 项 | 值 |
|----|-----|
| Ultralytics | `8.1.46` |
| PyTorch | `2.0.0+cu117` |
| GPU | `NVIDIA GeForce RTX 4090` |
| TensorRT（导出时） | **`10.16.1.11`** |
| 输入 | `images`，`(1, 3, 640, 640)` FP32（engine 为 **FP16**） |
| 输出 | `output0` `(1, 37, 8400)`，`output1` `(1, 32, 160, 160)` |
| ONNX | `best.onnx`，约 `12.6 MB`，`onnxsim` 简化成功 |
| Engine | `best.engine`，约 **`8.2 MB`** |
| Builder 耗时 | **约 172 s**（`Engine generation completed in 171.72 seconds`） |
| 总导出耗时 | 日志约 **173.7 s** |

### 5.3 导出后自检（分割任务）

```bash
yolo predict task=segment \
  model=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine \
  imgsz=640 half
```

HTTP 服务请将 **`SAM6D_YOLO_WEIGHTS`** 或 curl 的 **`yolo_weights`** 指向新生成的 **`best.engine`**；若进程内已缓存旧模型，**需重启服务** 后再压测。

### 5.4 与 PEM 记录的关系

§4 的 **`yolo_s` / `yolo_seg_backend` 耗时** 基于 **Ultralytics + TensorRT engine**；**升级 / 重装 TensorRT 后务必按本节重导 YOLO engine**，否则可能出现加载失败或静默数值漂移。PEM 侧 **`rgb_net` TensorRT** 接入与数值验证见 [`vit_acc.md`](vit_acc.md)；**`/infer` 上 PEM ViT 段加速前后对照**见 **§4.5 / §4.6**。
