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

建议脚本化压测，至少 30 次请求，保存为 `baseline.csv`。  
后续 TRT 结果必须与这个基线对比。

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
2. **任务精度**：最终 `score / R / t` 统计差异；
3. **吞吐与时延**：`pose_s` 均值、P95、显存变化。

建议阈值（可按业务调）：

- 位姿偏差在可接受范围内（与当前评估标准一致）；
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

1. 固定 baseline（当前 warm 方案）  
2. 选 1 个最稳定子模块导出 ONNX  
3. TRT FP32 对齐  
4. TRT FP16 性能验证  
5. 灰度接入到 warm 服务  
6. 扩大加速覆盖面

---

## 11. 与当前仓库的对应关系

- 热加载主逻辑：`Pose_Estimation_Model/run_warmup_inference_custom.py`
- HTTP 入口：`warmup_http_service.py`
- 当前可直接观察指标：`/infer` 返回中的 `timing.pose_s`

建议把每轮 TRT 实验结果追加到 `Pose_Estimation_Model/infer.md`，便于回归对比。
