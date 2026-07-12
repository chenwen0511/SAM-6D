# PEM TensorRT 加速指南（RTX 4090）

本文针对当前 `run_warmup_inference_custom.py` 的热加载方案，给出在 RTX 4090 上进一步使用 TensorRT 的落地步骤。

---

## 0. 先理解当前“热加载”在做什么

`run_warmup_inference_custom.py` 里的关键机制：

- 启动时执行 `_maybe_autopreload_default_pem()`，默认会预加载 PEM。
- `preload_default_pem_model()` 会：
  - 创建 `pose_estimation_model` 网络；
  - 加载 `sam-6d-pem-base.pth`；
  - 缓存到 `_CACHED_DEFAULT_MODEL`。
- `run_pose_inference()` 在默认参数下会复用该缓存模型，避免重复加载大权重。

这一步已经解决了“重复加载 1.3G checkpoint”的开销，后续 TRT 重点是降低前向推理时延。

---

## 1. 先做可行性判断（非常重要）

PEM 并不是纯 CNN，包含：

- ViT/Transformer 子模块
- 点云/几何匹配分支
- `pointnet2` 自定义算子（CUDA 扩展）

**结论**：通常不建议直接把整个 PEM 一键导出成单个 TRT 引擎。  
推荐采用“分阶段加速”：

1) 优先把可导出的 dense 子图（常见是图像特征提取/部分 MLP）转 TRT；  
2) 其余不稳定部分（点云匹配、几何求解）继续走 PyTorch。

这类混合方案在工程上最稳。

---

## 2. 环境建议（4090）

建议版本（任选一组，保持一致）：

- CUDA 11.8 + TensorRT 8.6.x + PyTorch 2.0/2.1
- 或 CUDA 12.x + TensorRT 10.x + 对应 PyTorch

你当前日志显示 `torch 2.0.0+cu117`，若走 TRT，建议升级到更常见组合（如 cu118 + trt8.6）再做导出验证，减少兼容问题。

---

## 3. 基线测试（必须先做）

在不改推理逻辑前，固定输入跑 N 次记录：

- `pose_s` 均值 / P95
- GPU 利用率与显存

建议脚本化压测，至少 30 次请求，保存为 `baseline.csv`；同时保留同一次运行生成的 `baseline_results.json`，便于与 TRT 对齐 **score** 与 **xyzrxryrz**。  
后续 TRT 结果必须与这个基线对比。

### 3.1 使用 `trt/baseline.py` 做函数级基线

仓库已提供：

- 脚本：`Pose_Estimation_Model/trt/baseline.py`
- 输出：
  - `Pose_Estimation_Model/trt/baseline.csv`：汇总时延 + 每次运行的 `latency_ms`、`score`、**xyz 与 rx/ry/rz**（`x_mm`…`rz_rad`，与 HTTP PEM 的 `xyzrxryrz` 一致）
  - `Pose_Estimation_Model/trt/baseline_results.json`（默认路径：与 `baseline.csv` 同目录、文件名为 `basename(baseline.csv)_results.json`）：完整 pose 快照、`reference_for_trt_compare`（默认取 **最后一次 benchmark 推理**）、`per_run` 逐次记录

该脚本特点：

- 直接 import `run_warmup_inference_custom.py` 的 `run_pose_inference`；
- 使用 `preload_default_pem_model` 先做模型预加载；
- 先执行若干次 warmup（默认 5 次），再进行 benchmark（默认 30 次）；
- 使用 `time.perf_counter_ns()` 纳秒级计时；
- CSV 写入汇总指标（mean/median/p95/min/max/std）以及每次样本的时延与六位姿；
- JSON 写入与 `sam6d_http_service` PEM 字段对齐的 `xyz_mm`、`rotation_euler_zyx_rad`、`xyzrxryrz` 等，供 TensorRT 数值对比。

示例：

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/Pose_Estimation_Model/trt

python baseline.py \
  --output_dir /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260508_101743_b16ac733 \
  --cad_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/models/tray_180mm_centered_mesh_v2.ply \
  --rgb_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png \
  --depth_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png \
  --cam_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json \
  --seg_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260508_101743_b16ac733/sam6d_results/detection_ism.json \
  --det_score_thresh 0.0 \
  --gpus 0 \
  --warmup_runs 5 \
  --benchmark_runs 30 \
  --csv_path /home/mui/projects/smt/SAM-6D/SAM-6D/Pose_Estimation_Model/trt/baseline.csv
```

可选：

- `--results_json /path/to/baseline_results.json` 指定 JSON 路径（不设则默认生成 `baseline_results.json`）；
- `--rd_seed`：进程启动时的全局随机种子（默认 `1`），需与 PEM 配置里的 `rd_seed`（如 `config/base.yaml`）保持一致；`baseline_results.json` 的 `inputs.rd_seed` 会记录该值。

建议固定输入与环境变量后再跑，确保 TRT 前后对比公平。

### 3.2 当前 baseline 结论（来自 `trt/baseline.csv`，2026-05-09 一次完整运行）

脚本参数（与 `baseline_results.json` 中 `inputs` 一致）：`warmup_runs=5`，`benchmark_runs=30`，`det_score_thresh=0.0`，`gpus=0`。

时延汇总（单位 ms）：

| 指标 | 数值 |
|------|------|
| mean | 473.669504 |
| median | 467.470258 |
| p95 | 500.846045 |
| min | 466.564187 |
| max | 547.636675 |
| std | 17.640594 |

样本数：`30` 次。

结论与解读：

- 多数样本落在约 `466~468 ms`；
- 少数样本明显偏高（例如约 `500~548 ms`），拉高 mean / p95 / std；若需压测结论更稳，可增大 `benchmark_runs` 或排查当时 GPU 抢占、功耗与温度；
- TRT 前后对比时，建议沿用同一套输入路径与 `--warmup_runs` / `--benchmark_runs`，并同时对比下文 **3.3** 中的位姿参考。

### 3.3 精度对照参考（来自 `trt/baseline_results.json`）

同一次运行生成的 JSON：`schema` 为 `sam6d_pem_baseline_v1`，记录时间（UTC）：`2026-05-09T00:47:08.329473+00:00`。

- **`reference_for_trt_compare`**：本轮取 **第 30 次** benchmark 的最佳检测快照（与 HTTP PEM 约定一致：`xyz_mm` + ZYX 欧拉 `rotation_euler_zyx_rad`，合并为 **`xyzrxryrz`**，单位 `mm_rad`）。
- 该参考一次典型值为：
  - `score`: `0.8950991034507751`
  - `xyzrxryrz`（`x,y,z` mm；`rx,ry,rz` rad）：  
    `[-40.00130844116211, -46.60285186767578, 423.06646728515625, 2.9259664290113028, -0.31181312340277084, 1.47031892828877]`
- **`per_run`**：每次 benchmark 的 `latency_ms` 与完整 `pose`。

输入目录说明（JSON 内 `inputs`，便于复现实验）：`output_dir` 指向含 `templates/` 的运行输出；`seg_path` 为该次 PEM 使用的 `detection_ism.json`。

### 3.4 随机种子相关改动与对齐结论（当前仓库）

以下为与本 TRT 指南直接相关的 **代码与结论**，便于做 PyTorch 基线 vs TensorRT 时的预期管理。

**已做的修改（要点）**

1. **`run_warmup_inference_custom.py` → `run_pose_inference()`**  
   每次推理在原有 `random.seed` / `torch.manual_seed` 之外，增加 **`np.random.seed(cfg.rd_seed)`** 与 **`torch.cuda.manual_seed_all(cfg.rd_seed)`**。  
   原因：数据管线里大量 **`np.random.choice`**（观测点/模板点采样）以及 **`mesh.sample`** 等依赖 **NumPy 全局 RNG**；仅设 Python `random` 与 `torch` 无法固定这些采样，会导致「同一套输入文件、每次进网络的点集不同」，进而 **`score` 与 `xyzrxryrz` 大幅抖动**。

2. **`run_inference_custom.py`（命令行 PEM）**  
   与上相同的种子逻辑，保证 CLI 与 API 路径行为一致。

3. **`trt/baseline.py`**  
   在 `preload_default_pem_model` 之前调用与 PEM 一致的全局设种（`random` / `torch` / `numpy` / `CUDA`），并提供 **`--rd_seed`**（默认 `1`），与 **`config/base.yaml` 的 `rd_seed`** 对齐；`baseline_results.json` 的 `inputs` 中记录 `rd_seed`。

**现象与结论**

| 阶段 | 现象 |
|------|------|
| 修正前 | 相同输入下多轮 benchmark，`score` 与六位姿可 **差异很大**（主要源于 **未固定的 NumPy 采样**）。 |
| 修正后 | 多轮结果 **总体非常接近**；若仍有微小差异，多为 **GPU 浮点累加顺序、cuDNN/自定义 CUDA 算子的非确定性**，属常见数值噪声。 |

**TRT / 基线数值对齐建议**

- **不必追求逐比特完全一致**；对 `xyzrxryrz` 采用 **容差**（mm / rad）更实际。
- 实践经验：**`score ≥ 0.85`** 的样本上，位姿与置信度 **已基本接近**，适合作为主对照集；低分样本解更模糊，微小数值扰动更容易放大，宜放宽容差或单独分析。

---

## 4. 选择 TRT 目标子模块

建议先从收益大、导出稳定的部分开始：

1. 图像特征提取（ViT_AE / descriptor 相关块）  
2. 纯 MLP / Linear 堆叠模块  

暂不建议第一阶段就导出：

- 带复杂动态 shape 的点云匹配全链路
- 依赖 `pointnet2` CUDA 扩展的路径
- 包含几何后处理（SVD/匹配筛选）的端到端整体图

---

## 5. 导出 ONNX（按子模块）

为目标模块准备 `export_onnx_*.py`，核心步骤：

1. 加载和 warmup 相同的权重；
2. 构造代表性 dummy input（覆盖实际 shape 范围）；
3. `torch.onnx.export(...)` 导出；
4. 用 `onnxsim` 简化；
5. 用 `onnxruntime` 对齐 PyTorch 输出（误差阈值如 `1e-3 ~ 1e-2`）。

注意：

- 明确 dynamic axes（batch、点数等）；
- 尽量减少不必要 dynamic 维度；
- 先做 FP32 对齐，再考虑 FP16。

---

## 6. 从 ONNX 构建 TensorRT 引擎

可用 `trtexec` 快速验证（示例）：

```bash
trtexec \
  --onnx=feature_encoder.onnx \
  --saveEngine=feature_encoder_fp16.engine \
  --fp16 \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:1x3x224x224 \
  --maxShapes=input:4x3x224x224 \
  --timingCacheFile=timing.cache
```

若有多输入（例如点云分支），为每个输入都设置 `min/opt/max`。

---

## 7. 在现有 warm 逻辑中接入 TRT

建议新增一个 TRT Runner（示意）：

- `trt/load_engine.py`：加载 engine + 创建 context
- `trt/run_engine.py`：封装输入绑定、执行、输出拷贝
- `trt/compare.py`：和 PyTorch 输出对齐检查

在 `run_warmup_inference_custom.py` 中：

1. 预加载阶段同时加载 TRT engine（类似当前 `_CACHED_DEFAULT_MODEL` 思路）；
2. 在 `run_pose_inference()` 里把目标子模块替换为 TRT 调用；
3. 其余链路保持不变（先保证可用）。

---

## 8. 精度与稳定性验证

至少做三类验证：

1. **数值对齐**：PyTorch vs TRT 子模块输出误差；
2. **任务精度**：最终 `score / R / t`（或与 HTTP 一致的 **`xyzrxryrz`**）统计差异；
3. **吞吐与时延**：`pose_s` 均值、P95、显存变化。

建议阈值（可按业务调）：

- 位姿偏差在可接受范围内（与当前评估标准一致）；
- 可优先在 **`score` 较高（例如 ≥ 0.85）** 的样本上收紧对齐要求，低分样本适当放宽（详见 **§3.4**）；
- 时延收益 >= 20% 才建议上线。

---

## 9. 常见问题（4090）

1. **ONNX 导出失败**  
   - 通常是自定义算子或动态 shape 不受支持；
   - 先切小子图导出，不要一开始追求端到端。

2. **TRT 构建成功但结果飘**  
   - 先 FP32 验证，再切 FP16；
   - 校准输入预处理和 dtype。

3. **速度没提升反而变慢**  
   - engine 太小、拷贝开销大；
   - 需要增大可融合子图，减少 PyTorch/TRT 来回切换。

4. **显存爆炸**  
   - dynamic shape 范围设置过大；
   - 缩小 `maxShapes`，按真实请求分布给 profile。

---

## 10. 推荐实施顺序

1. 固定 baseline（`baseline.csv` + `baseline_results.json`，并与 **`rd_seed`**、输入路径一致，参见 **§3.4**）  
2. 选 1 个最稳定子模块导出 ONNX  
3. TRT FP32 对齐  
4. TRT FP16 性能验证  
5. 灰度接入到 warm 服务  
6. 扩大加速覆盖面

---

## 11. 与当前仓库的对应关系

- 热加载与 PEM API：`Pose_Estimation_Model/run_warmup_inference_custom.py`（含 **`run_pose_inference()`** 内每次推理的随机种子设置）
- CLI PEM：`Pose_Estimation_Model/run_inference_custom.py`
- 函数级基线与 pose 记录：`Pose_Estimation_Model/trt/baseline.py`
- HTTP 入口：`warmup_http_service.py`
- 当前可直接观察指标：`/infer` 返回中的 `timing.pose_s`

建议把每轮 TRT 实验结果追加到 `Pose_Estimation_Model/infer.md`，便于回归对比。
