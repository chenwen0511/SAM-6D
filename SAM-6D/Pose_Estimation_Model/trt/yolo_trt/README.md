# YOLO 分割优化记录（当前已完成）

当前只记录已经落地并验证有效的优化，不展开未实施方案。

## 已完成优化点

前置过程（YOLO 分割）已从约 **164ms** 优化到约 **10ms**（本次实测 `yolo_s=0.008888s`，约 `8.9ms`）。

主要优化点：

1. 注释掉两个 `_draw_overlay` 画图保存过程（主要耗时点）
   - `vis_yolo_seg.png`
   - `vis_ism.png`

2. 权重文件由 `.pt` 转为 `.engine` 并用于推理
   - 说明：当前分割模型较小，单看模型前向速度优势不明显；
   - 主要收益来自服务端链路开销下降（配合第 1 点效果更明显）。

## 实测调用命令

```bash
curl -X POST "http://127.0.0.1:8001/infer" \
  -F "rgb=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png" \
  -F "depth=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png" \
  -F "camera=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json" \
  -F "seg_backend=yolo_seg" \
  -F "yolo_conf=0.25" \
  -F "yolo_imgsz=640" \
  -F "yolo_class_id=0" \
  -F "det_score_thresh=0.0"
```

## 返回关键时延（本次）

- `yolo_s`: `0.008888s`（约 `8.9ms`）
- `pose_s`: `0.606515s`
- `pipeline_s`: `0.615689s`
- `upload_s`: `0.001106s`
- `total_s`: `0.616795s`

对比说明：相较此前 warm 记录中 `yolo_s ≈ 0.1618s`，YOLO 分割阶段耗时显著下降。

## 备注

- 当前记录聚焦时延优化，不展开其它未定方案。
- 本次返回 `score` 偏低（约 `0.0051`），上线前仍需做精度回归（`score`、`xyzrxryrz`、业务容差）。
# YOLO 分割 · TensorRT 推理（RTX 4090）

权重示例：

`/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.pt`

采集相机：**640×480**（宽 × 高）。YOLO 推理时会自动按 `imgsz` 做缩放 / letterbox，无需把相机改成正方形。

以下为 **Linux + 单卡 RTX 4090** 的可操作步骤。下文 **§1 方案 A** 步骤最少；**PyBind** 见 §3。

---

## 0. 环境假设

- 驱动与运行时支持当前环境对应的 CUDA 主版本（你当前实测为 `torch 2.0.0+cu117`，即 CUDA 11.7；4090 也常见 CUDA 12.x 组合）。
- Python 环境已安装 **`ultralytics`**（与训练该 `best.pt` 的版本尽量一致）。
- 一键导出 **`.engine`** 时，同一环境需能 **`import tensorrt`**（与 CUDA 匹配的 TensorRT Python 包或 wheel）。

可选：`onnx`、`onnxruntime`（方式 B）；`trtexec`（方式 B）。

自检（任选）：

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import tensorrt as trt; print(trt.__version__)"
nvidia-smi -L
```

### 0.0 当前机器已验证环境（本次实测）

以下组合来自本次实际导出日志，可作为首选参考：

- OS：Ubuntu（x86_64）
- GPU：NVIDIA GeForce RTX 4090（单卡）
- Python：3.9.6（micromamba 环境 `sam6d`）
- PyTorch：`2.0.0+cu117`
- Ultralytics：`8.1.46`（从 `8.0.135` 升级后）
- TensorRT（Python）：`10.13.0.35`
- ONNX：`1.19.1`
- onnxsim：`0.4.36`
- onnxruntime-gpu：`1.19.2`



### 0.1 报错 `ModuleNotFoundError: No module named 'tensorrt'`

方案 A 的 **`yolo export ... format=engine`** 会在 **当前 Python 环境** 里调用 TensorRT；必须先装好 **与 CUDA 主版本一致** 的包。

**1）先看 PyTorch 报告的 CUDA 版本（决定装哪一个 pip 包）：**

```bash
python -c "import torch; print('cuda', torch.version.cuda)"
```

**2）用 pip 安装（在已激活的 conda/micromamba 环境中执行）：**

```bash
python -m pip install -U pip wheel

# PyTorch 为 CUDA 12.x（输出类似 12.1 / 12.4）时常见：
python -m pip install tensorrt-cu12

# PyTorch 为 CUDA 11.x（输出类似 11.8）时常见：
# python -m pip install tensorrt-cu11
```

官方说明可参考：[TensorRT pip 安装](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)。若默认源较慢，可换国内镜像或 NVIDIA 文档中的索引。

**3）再自检：**

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```




### 0.2 `max_workspace_size` 报错的最小修复版本（实测）

若日志出现：

`IBuilderConfig object has no attribute max_workspace_size`

通常是 **TensorRT 10 + 旧 Ultralytics（如 8.0.135）** API 不兼容。  
本项目实测在同一机器（RTX 4090、`torch 2.0.0+cu117`、TensorRT 10.13）上，将 Ultralytics 升级到 **`8.1.46`** 后可直接成功导出：

```bash
python -m pip install "ultralytics==8.1.46"
python -c "import ultralytics; print(ultralytics.__version__)"   # 8.1.46
yolo export model=best.pt format=engine imgsz=640 half=True device=0 workspace=8
```

该次导出结果（关键日志）：

- ONNX export success（`best.onnx`）
- TensorRT export success（`best.engine`）
- Engine generation completed in ~205s，`best.engine` 约 `9.1 MB`

建议：若你当前仍是 `8.0.135`，优先先升到 **`8.1.46`** 复测；若仍有兼容问题再升级到更高 8.x。

---

## 1. 方案 A：Ultralytics → TensorRT Engine（推荐，步骤最少）

适用于：**640×480 相机图** + **单卡 4090**，与仓库内 `yolo_seg_backend.py`（Ultralytics API）兼容。

### 1.1 关于分辨率与 `imgsz`

| 概念 | 说明 |
|------|------|
| 相机 **640×480** | 原始 RGB 宽高；保存为 `.png` / 传给 `predict` 即可。 |
| **`imgsz=640`** | Ultralytics 常用设置：按 stride 将图像 **letterbox** 进网络（不必等于相机分辨率）。 |
| 与训练对齐 | **导出时的 `imgsz` 应与训练该 `best.pt` 时一致**（常见为 `640`）。若你训练用的是别的尺寸，下面命令里的 `640` 改成相同值。 |

结论：**相机 640×480 时仍通常使用 `imgsz=640` 导出与推理**，无需设为 `480` 除非训练就是如此。

### 1.2 步骤一：进入权重目录并指定 GPU

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights

export CUDA_VISIBLE_DEVICES=0
```

确认 `best.pt` 存在：

```bash
ls -la best.pt
```

### 1.3 步骤二：导出 FP16 TensorRT 引擎

**首选（一条命令）：**

```bash
yolo export model=best.pt format=engine imgsz=640 half=True device=0 workspace=8
```

- **`half=True`**：FP16，4090 上速度与显存更友好。  
- **`workspace=8`**：builder workspace（Ultralytics 内部语义随版本变化）。  
- 成功后在当前目录生成 **`best.engine`**（与 `best.pt` 同名换后缀）：

```bash
ls -la best.engine
```

若 **`yolo export ... format=engine`** 因 **TensorRT 10 + 旧 Ultralytics** 失败（见 **§0.2**），但已成功生成 **`best.onnx`**，则用 **`trtexec`** 生成同名 **`best.engine`** 即可，推理方式不变。

若报错找不到 TensorRT：按 **§0.1** 安装 Python 包；导出引擎阶段亦可完全依赖 **§2 / `trtexec`**。

### 1.4 步骤三：用 640×480 图像做一次验证

准备一张 **640×480** 的 RGB（或任意分辨率测试图）：

```bash
# 将路径换成你的样例图（例如相机保存的 rgb.png）
yolo predict \
  task=segment \
  model=best.engine \
  source=/path/to/your/rgb_640x480.png \
  imgsz=640 \
  conf=0.25 \
  classes=0 \
  device=0 \
  verbose=False

# 结果默认写在 runs/segment/predict* 下，可看可视化 mask
```

实测（2026-05-09）：

- 在 `ultralytics==8.1.46` + `best.engine` 下，若不显式传 `task=segment`，CLI 可能出现自动任务猜测为 detect，并触发 `KeyError`。
- 显式加上 `task=segment` 后可稳定跑通，日志示例：
  - `Loading ... best.engine for TensorRT inference...`
  - `Results saved to runs/segment/predict`
  - 无报错退出。

对比 PyTorch 基线（可选，用于确认精度）：

```bash
yolo predict \
  model=best.pt \
  source=/path/to/your/rgb_640x480.png \
  imgsz=640 \
  conf=0.25 \
  device=0 \
  verbose=False
```

### 1.5 步骤四：接入 SAM-6D（YOLO 分割后端）

仓库入口：`SAM-6D/yolo_seg_backend.py`，内部为 **`YOLO(weights_path)`**。将权重路径指向引擎文件即可，例如：

```bash
export SAM6D_YOLO_WEIGHTS=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine
```

HTTP 调用时在服务端保证上述环境变量（或与代码里传入的 `yolo_weights` 一致）。本仓库 `warmup_http_service` / `sam6d_http_service` 使用 **`yolo_imgsz`（默认 `640`）** 传给 Ultralytics，请与导出时的 **`imgsz=640`** 保持一致；若改分辨率，需 **重新导出 `best.engine`** 并在请求里传相同 `yolo_imgsz`。

### 1.6 性能预期（简要）

相对同一环境下的 **`best.pt`**：**GPU 核心推理**一般由 TensorRT 加速；分割还有 **mask 解码与缩放**，仍在 Ultralytics pipeline 中完成，整体仍有明显提升，具体以你在 **640×480** 图上实测耗时为准（建议 warmup 后循环上百帧取平均）。

### 1.7 `pef.py`：`predict` 耗时对比（实测，2026-05-09）

脚本：`Pose_Estimation_Model/yolo_trt/pef.py`。测量 **`YOLO(..., task="segment").predict(...)`** 整段耗时（前后 `torch.cuda.synchronize()`），**不包含** 一次性 `yolo export` 建引擎时间。

环境与权重：

- GPU：RTX 4090，`device=0`
- `ultralytics==8.1.46`，`torch 2.0.0+cu117`
- `best.pt` / `best.engine`：`user_data/yolo_runs/tray_seg/weights/`
- 测试图：`user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png`
- 参数：`warmup=10`，`runs=50`，`imgsz=640`，`conf=0.25`

命令：

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/Pose_Estimation_Model/yolo_trt

python pef.py \
  --pt /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.pt \
  --engine /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.engine \
  --rgb /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png \
  --warmup 10 \
  --runs 50 \
  --imgsz 640 \
  --conf 0.25 \
  --device 0 \
  --baseline-s 0.16
```

结果摘要（`pef.py` 终端输出）：

| 模型 | mean (ms) | median (ms) | min / max (ms) | std (ms) |
|------|-----------|-------------|----------------|----------|
| `best.pt` | 7.438 | 7.471 | 7.260 / 7.759 | 0.122 |
| `best.engine` | 6.102 | 6.125 | 5.842 / 6.763 | 0.142 |

- **TensorRT engine 相对 PT**：按均值约 **1.22×**（`pt/engine`，engine 更快）。
- **`--baseline-s 0.16`**（160 ms）：脚本用于对照 HTTP 里记录的 **`yolo_s` 量级**；本实测 **`predict` 仅约 6～8 ms**，说明服务端 **`yolo_s≈0.16 s`** 除 GPU 推理外，还包括 Python 调度、I/O、后处理等与 **`pef.py` 窄口径计时** 不一致的部分，**不宜直接把 160 ms 当作单次 `predict` 纯算力耗时**。

---


## 5. 服务端优化实测（2026-05-09）

本节记录 `warmup_http_service.py + yolo_seg_backend.py` 链路的优化改动与 `/infer` 实测结果。

### 5.1 改动点（已落地）

1. **分割任务显式指定**
   - `YOLO(..., task="segment")`
   - `model.predict(..., task="segment")`
   - 目的：避免 `.engine` 被自动误判为 detect，导致 `KeyError` / `masks` 异常。

2. **服务启动即预加载 + warmup**
   - `warmup_http_service.py` 启动阶段读取 `SAM6D_YOLO_WEIGHTS`。
   - 调用 `preload_yolo_model()` 预加载模型到 cache，并执行一次 dummy warmup。
   - 目的：降低首请求抖动。

3. **后处理耗时优化**
   - `_mask_to_rle` 改为 `pycocotools.mask.encode`（替代手写 Python 循环）。
   - 固定当前链路分辨率为 `640x480`（注意 `cv2.resize` 参数顺序为 `(width, height)`）。
   - 当前代码中默认不保存 YOLO 可视化图（减少 I/O）。

4. **耗时可观测性**
   - 增加 `_load_model` / `model.predict` / `load+predict` 的毫秒级打印。

### 5.2 调用命令（实测）

```bash
curl -X POST "http://127.0.0.1:8001/infer" \
  -F "rgb=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png" \
  -F "depth=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png" \
  -F "camera=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json" \
  -F "seg_backend=yolo_seg" \
  -F "yolo_conf=0.25" \
  -F "yolo_imgsz=640" \
  -F "yolo_class_id=0" \
  -F "det_score_thresh=0.0"
```

### 5.3 本次返回时延

- `yolo_s`: `0.008888s`（约 `8.9ms`）
- `pose_s`: `0.606515s`
- `pipeline_s`: `0.615689s`
- `upload_s`: `0.001106s`
- `total_s`: `0.616795s`

相较此前 warm 记录中 `yolo_s ≈ 0.1618s`，YOLO 分割阶段耗时显著下降。

> 备注：本次返回 `score` 偏低（约 `0.0051`）。上线前需继续做精度回归（`score`、`xyzrxryrz`、业务容差），避免仅看时延。