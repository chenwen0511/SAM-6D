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

### 0.0.1 可选：清理旧包（卸载）

若环境里历史版本较多，建议先清理再安装，避免混装冲突：

```bash
python -m pip uninstall -y ultralytics tensorrt tensorrt-cu11 tensorrt-cu12 tensorrt-cu13 onnx onnxsim onnxruntime-gpu
python -m pip cache purge
```

> 注意：卸载会影响当前环境的推理能力。建议先导出 `pip freeze > freeze_backup.txt`，或在新环境操作。

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


**5）只想先绕开「export 阶段」的 TensorRT Python：** 可用 **§2**：Ultralytics 只导出 **`best.onnx`**（一般不需要 `tensorrt`），再用 NVIDIA Tar 包里的 **`trtexec`** 生成 **`best.engine`**。注意：**Ultralytics 用 `YOLO('best.engine')` 做推理时，多数环境仍需要装好 Python 版 TensorRT**；若长期完全不装，需改用自有 C++/runtime 加载引擎。

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

### 0.3 备用路径：`trtexec` / TRT8.6

若日志类似：

`IBuilderConfig object has no attribute max_workspace_size'`

原因：**TensorRT 10** 已移除该字段；**Ultralytics 8.0.x** 等旧版本仍按 TensorRT 8 API 写死，二者不兼容。

**路径 A（推荐，立刻可用）：** 你已导出 **`best.onnx`** 时，直接用 **`trtexec`**（随 TensorRT 安装，需在 `PATH`）生成引擎，**不必**再走失败的 Python builder：

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights

# TensorRT 10：workspace 用 memPoolSize（MiB）；按需调整数值
trtexec \
  --onnx=best.onnx \
  --saveEngine=best.engine \
  --fp16 \
  --memPoolSize=workspace:8192 \
  --verbose
```

成功后：`yolo predict model=best.engine ...` 或 `SAM6D_YOLO_WEIGHTS=.../best.engine`。

**路径 B：** **升级 Ultralytics** 到支持 TensorRT 10 的版本后再执行 **`yolo export ... format=engine`**（可先从 `8.1.46` 起测）：

```bash
python -m pip install -U ultralytics
yolo export model=best.pt format=engine imgsz=640 half=True device=0 workspace=8
```

升级后请 regression：`predict`、与你 SAM-6D YOLO 分段流水线是否仍兼容。

**路径 C：** 安装 **TensorRT 8.6.x** 一类仍提供 `max_workspace_size` 的旧 Python 绑定（与当前驱动/CUDA 是否兼容需自行核对），一般不如路径 A/B 省事。

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

---

## 2. 方式 B：ONNX → TensorRT（`trtexec`，便于接 C++/PyBind）

### 2.1 导出 ONNX

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights

export CUDA_VISIBLE_DEVICES=0

yolo export model=best.pt format=onnx imgsz=640 opset=17 simplify=True dynamic=False
```

得到 `best.onnx`（分割模型会有多个输出：检测框 / mask 系数 / proto 等，具体以 Netron 打开 ONNX 为准）。

### 2.2 构建 FP16 Engine（固定 batch=1、固定输入 640×640 示例）

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights

# 根据 ONNX 实际输入名/shape 调整；YOLOv8 分割常见输入名为 images，形状 1x3x640x640
trtexec \
  --onnx=best.onnx \
  --saveEngine=best_seg_fp16.engine \
  --fp16 \
  --workspace=8192 \
  --memPoolSize=workspace:8192 \
  --verbose

# 若 trtexec 报 shape/profile 错误，用下面命令查看 ONNX 输入输出：
# polygraphy inspect model best.onnx
# 或 onnxruntime / Netron 手动核对后再补 --minShapes/--optShapes/--maxShapes
```

动态 batch 时需增加 profile（示例，**名称与维度必须与你的 ONNX 一致**）：

```bash
trtexec \
  --onnx=best.onnx \
  --saveEngine=best_seg_fp16_dynamic.engine \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:4x3x640x640 \
  --workspace=8192
```

---

## 3. PyBind11 + TensorRT（自定义加速路径）

引擎文件：`best.engine` 或 `best_seg_fp16.engine`（由 §1 或 §2 生成）。

典型步骤（无统一单行命令，需在工程里 CMake）：

1. C++ 侧：`nvinfer1::IRuntime` → `deserializeCudaEngine` → `createExecutionContext`，绑定输入输出 buffer，`enqueueV3`（或对应 API）。
2. Python 侧：用 **pybind11** 暴露例如 `infer(float_ptr)` / `infer_numpy(np.ndarray)`。
3. CMake 大致依赖：**TensorRT `include` + `libnvinfer.so`**、**CUDA**、**pybind11**，编译为 `.so`，`import your_module`。

官方可参考 TensorRT samples（`sampleOnnxMNIST` 等）改写成模块；PyBind 绑定范例见 [pybind11 docs](https://pybind11.readthedocs.io)。

分割后处理（mask = protos @ coeffs + sigmoid + resize）若仍在 Python 里做，通常足够快；瓶颈多在 backbone + neck 的 GPU 推理。

---

## 4. 与当前 SAM-6D 仓库的衔接

HTTP / pipeline 里 YOLO 入口见：`SAM-6D/yolo_seg_backend.py`，当前为 **`Ultralytics YOLO(best.pt)`**。换成 TensorRT 需要：

- 要么 **`YOLO("best.engine")`**（优先验证）；
- 要么改为加载你的 **PyBind 扩展**，在扩展内跑 engine，并在 Python 侧拼出与现逻辑一致的 mask/bbox（再写 `detection_ism.json`）。

---

## 5. 4090 常见问题

| 现象 | 处理 |
|------|------|
| `yolo export ... engine` 失败 | 确认 `pip show tensorrt` 与 CUDA 版本匹配；或用 §2 ONNX + `trtexec`。 |
| `trtexec` 找不到输入名 | `polygraphy inspect model best.onnx`，按输出改 `--shapes`。 |
| FP16 精度下降 | 先用 `--fp32` 建引擎对比 mask；再决定是否 FP16。 |
