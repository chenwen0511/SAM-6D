# ViT_AE（`rgb_net`）TensorRT FP16 导出与加速备忘

本文记录：**固定输入 `1×3×224×224`**、**权重来自加载 `sam-6d-pem-base.pth` 后的 `feature_extraction.rgb_net`（`ViT_AE`）**、推理精度 **FP16（TRT engine）** 的可执行步骤。与 `get_img_feats` 中实际使用的 **`self.rgb_net(img)[0]`**（dense 特征图）对齐。

更完整的 PEM 时延与 HTTP 记录见同目录 [`README.md`](README.md)。

---

## 1. 两个 checkpoint 与本次导出的关系

| 文件 | 作用 | 与 TRT 导出 |
|------|------|-------------|
| `checkpoints/mae_pretrain_vit_base.pth` | MAE 预训练 ViT-Base，在 `ViT_AE.__init__` 中用于初始化 `self.vit` | **导出 TRT 时不要单独用它**；仅构建网络且 `pretrained=True` 时需要该文件存在 |
| `checkpoints/sam-6d-pem-base.pth` | PEM 整网训练权重，含 **`rgb_net`（ViT_AE）** 微调后的参数 | **`load_checkpoint` 后取出的 `rgb_net` 才是部署与导出的真值** |

结论：**ONNX / TensorRT 子图权重 = 加载 PEM checkpoint 后的 `rgb_net`**，与线上一致。

---

## 2. 环境与目录

- 系统：Ubuntu 22.04；GPU：NVIDIA RTX 4090（单卡示例）。
- Conda 环境：与日常跑 PEM 相同（如 `sam6d`）。
- 工作目录（以下命令默认在此执行）：

```bash
cd /path/to/SAM-6D/SAM-6D/Pose_Estimation_Model
```

将 `/path/to/SAM-6D` 换成你的仓库根路径（例如 `~/projects/smt/SAM-6D`）。

需已安装：`torch`、`onnx`（可选 `onnxsim`）、**TensorRT**（`tensorrt` Python 包 + **`trtexec`** 命令行，与 YOLO 导出环境一致即可）。

---

## 3. 步骤一：加载 PEM，取出 `rgb_net`（可选：单独存子模块权重）

在 `Pose_Estimation_Model` 下执行（路径按你机器修改）：

```bash
cd /path/to/SAM-6D/SAM-6D/Pose_Estimation_Model

python - <<'PY'
import os, sys, torch, importlib
import gorilla

ROOT = os.path.abspath(".")
os.chdir(ROOT)
for p in ("provider", "utils", "model", os.path.join("model", "pointnet2")):
    sys.path.insert(0, os.path.join(ROOT, p))
sys.path.insert(0, ROOT)

cfg = gorilla.Config.fromfile(os.path.join(ROOT, "config", "base.yaml"))
cfg.model_name = "pose_estimation_model"
cfg.gpus = "0"
gorilla.utils.set_cuda_visible_devices(gpu_ids="0")

MODEL = importlib.import_module(cfg.model_name)
net = MODEL.Net(cfg.model).cuda().eval()

ckpt = os.path.join(ROOT, "checkpoints", "sam-6d-pem-base.pth")
gorilla.solver.load_checkpoint(model=net, filename=ckpt)

rgb_net = net.feature_extraction.rgb_net
out = os.path.join(ROOT, "checkpoints", "export_rgb_net_only.pth")
torch.save(rgb_net.state_dict(), out)
print("saved:", out)
PY
```

说明：若 **`mae_pretrain_vit_base.pth`** 不在 `checkpoints/`，但 **`sam-6d-pem-base.pth` 已含完整 `rgb_net` 参数**，通常仍可成功加载整网；若从零构建 `ViT_AE` 且依赖 MAE 初始化，则需保留 MAE 文件。

---

## 4. 步骤二：导出 ONNX（仅 dense 特征图，与 `rgb_net(img)[0]` 一致）

推理里 **`get_img_feats` 只使用 `self.rgb_net(img)[0]`**，`cls_tokens` 未参与后续。使用仓库脚本 **`export_pem_rgb_net_onnx.py`**（位于 **`Pose_Estimation_Model/`** 根目录，与 `config/`、`checkpoints/` 同级）。

脚本要点：

- **`DenseOnly`**：与文档前述一致，只导出 dense 支路。
- **SDPA / ONNX**：PyTorch 2.0 的 ONNX 仍常无法导出 **`aten::scaled_dot_product_attention`**。脚本在导出前 **`_patch_sdpa_for_onnx_export`**：将 **`torch.nn.functional.scaled_dot_product_attention`** 临时替换为 matmul/softmax 实现，导出结束 **`_restore_sdpa`**；并保留 **`_disable_fused_sdpa_for_onnx_export`** 作为辅助。

### 为什么 `_sdpa_export_safe` 能修复「`scaled_dot_product_attention` 无法导出」？

**1. ONNX 导出在做什么**

`torch.onnx.export` 会对 `forward` 做 **追踪（trace 或 script 相关路径）**，得到一张由 **ATen 算子**（如 `aten::matmul`、`aten::softmax`）组成的计算图，再靠 **symbolic 函数** 把每个 ATen 算子 **映射** 成 ONNX 里已有定义的算子（如 `MatMul`、`Softmax`）。**只有「在 PyTorch 的 ONNX 映射表里有符号化实现」的 ATen 节点才能变成合法 ONNX**；否则会报 *UnsupportedOperatorError*。

**2. 报错本质：融合算子进了图，但映射表不认**

从 PyTorch 2.0 起，注意力常用 **`torch.nn.functional.scaled_dot_product_attention`（SDPA）** 一条算子完成 QK^T、缩放、mask、softmax、乘 V 等逻辑。在图里它往往对应 **`aten::scaled_dot_product_attention`** 这一类 **「大而全」的融合节点**。

在你使用的组合（例如 **PyTorch 2.0.x + ONNX opset 17**）下，**旧版 ONNX 导出器对 `aten::scaled_dot_product_attention` 没有可用的 symbolic**，因此一旦图里出现该节点，导出就会直接失败——这与「数学上注意力算不算得对」无关，纯粹是 **「这个 ATen 算子当前没人翻译成 ONNX」**。

**3. 仅关 `enable_flash_sdp` / `enable_math_sdp` 为什么仍可能不够**

这些开关影响的是 **PyTorch 在 CUDA 上选用哪条 SDPA 后端实现**（Flash、memory-efficient、math 等），理想情况下希望走到可分解的 **math** 路径。但在不少版本里，**追踪到的图仍可能保留对 `scaled_dot_product_attention` 的单一调用**，而不是展开成一串基础 ATen；只要图里仍是那个 **未支持映射的融合节点**，ONNX 导出照样失败。

**4. `_sdpa_export_safe` 起什么作用**

`_sdpa_export_safe` 在 **Python 层** 接管 **`F.scaled_dot_product_attention` 这个名字**：在导出 `forward` 时，timm ViT 里原本调用的仍是 `F.scaled_dot_product_attention(...)`，但实际执行的是 **手写展开**：

- 用 **`matmul`**（或等价）显式计算 \(Q K^\top\) 并乘缩放因子；
- 按需处理 **因果 / `attn_mask`**（与原版语义对齐的常见写法）；
- 用 **`softmax`** 在最后一维得到注意力权重；
- 再 **`matmul`** 与 \(V\) 相乘得到输出。

这样追踪到的图里 **不再出现** `aten::scaled_dot_product_attention`，而是 **`aten::matmul`、`aten::softmax` 等「老熟人」**，它们都有成熟的 ONNX 符号化，因此 **导出可以继续**。

**5. 数值与部署含义**

对 **ViT 编码器** 典型设置（双向注意力、无 causal mask、推理等价于 `dropout_p=0`），上述展开与标准缩放点积注意力 **同一套公式**；权重仍来自 **`sam-6d-pem-base.pth` 加载后的 `rgb_net`**，**没有改 checkpoint**，只是 **导出瞬间** 换了一条 **ONNX 友好** 的实现路径。导出结束后 **`_restore_sdpa`** 把 `F.scaled_dot_product_attention` 还原，避免影响同进程里其它脚本。

执行：

```bash
cd /path/to/SAM-6D/SAM-6D/Pose_Estimation_Model
python export_pem_rgb_net_onnx.py
```

将 `/path/to/...` 换为你的仓库路径。成功后在 **`checkpoints/pem_rgb_net_b1_224.onnx`** 得到 ONNX。

**可选简化 ONNX**（需 `pip install onnx onnxsim`）：

```bash
onnxsim checkpoints/pem_rgb_net_b1_224.onnx checkpoints/pem_rgb_net_b1_224_sim.onnx
```

后续 `trtexec` 使用 `*_sim.onnx` 即可。

---

## 5. 步骤三：TensorRT FP16 engine

```bash
cd /path/to/SAM-6D/SAM-6D/Pose_Estimation_Model

trtexec --onnx=checkpoints/pem_rgb_net_b1_224_sim.onnx \
  --saveEngine=checkpoints/pem_rgb_net_b1_224_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096
```

若未做 `onnxsim`，将 `--onnx=` 改为 `pem_rgb_net_b1_224.onnx`。

---

## 6. 步骤四：数值验收（再接业务代码）

1. **PyTorch 基线**：同一 `x`（`1×3×224×224`），`dense_pt = DenseOnly(rgb_net)(x)`。
2. **TRT**：用 `tensorrt` + `pycuda`、`polygraphy run` 或自写推理脚本加载 **`pem_rgb_net_b1_224_fp16.engine`**，得到 `dense_trt`。
3. 对比 **`(dense_pt - dense_trt).abs().max()`** 及相对误差；可选：对两路输出分别做 **`get_chosen_pixel_feats`**，再对比下游 pose（与 `trt/README.md` 中 `xyzrxryrz` 对照思路一致）。

---

## 7. 与模板 42 视角循环的关系

`get_obj_feats` 对每个模板视角调用一次 **`rgb_net(tem)`**，每次均为 **`B=1`、`224×224`**。部署时：**同一 engine 连续调用 42 次**即可；若将来改为 **多视角拼 batch**，需另设 **dynamic shape / 多 profile**，不在本文「固定 B=1」范围内。

---

## 8. 常见失败与处理

| 现象 | 建议 |
|------|------|
| **`scaled_dot_product_attention` ONNX 不支持** | 当前脚本已内置 **`_patch_sdpa_for_onnx_export`**；若仍失败：检查 timm 是否把 SDPA 绑定到其它符号、或升级 PyTorch / 换 dynamo 导出路径 |
| `TracerWarning: Converting a tensor to a Python boolean` | 多为警告；若导出失败再针对断言分支处理 |
| ONNX 导出其它报错 | 提高/降低 `opset_version`；确认 `m.eval()`；尝试 `onnxsim` |
| `trtexec` parser 失败 | 换简化后 ONNX；查 TensorRT 版本与 PyTorch/CUDA 矩阵 |
| FP16 误差偏大 | ONNX 保持 FP32，仅 engine 用 `--fp16`；或缩小验收阈值后接受 |

---

## 附录：`pose_acc/a.log` 一次跑通记录解析

以下整理自同目录 **`a.log`**（Ubuntu、`sam6d` 环境、**PyTorch 2.0.0+cu117**、**TensorRT 10.16.1**、`trtexec` 报告 **`TensorRT v101601`**）。便于对照「本机是否也跑到同样阶段」以及 **`trtexec` 性能摘要怎么读**。

### A. ONNX 导出（`export_pem_rgb_net_onnx.py`）

| 项目 | 日志含义 |
|------|----------|
| MAE 加载 | 打印 `load pre-trained checkpoint from: checkpoints/mae_pretrain_vit_base.pth`，与 `ViT_AE` 构建一致。 |
| `TracerWarning` | `torch/__init__.py` 中 **tensor 转 Python bool** 的追踪警告；本次 **未阻止导出**，若换输入形状或分支仍建议做数值对照。 |
| ONNX Diagnostic | `0 WARNING 0 ERROR`，**导出成功**。 |
| 产物路径 | `checkpoints/pem_rgb_net_b1_224.onnx`（日志中为绝对路径，等价于相对 `Pose_Estimation_Model/` 的上述路径）。 |

### B. `onnxsim` 简化

- **提示**：图中含 **`Tile` / `ConstantOfShape`**，折叠常量可能使简化图「逻辑上更大」；若不符合预期可加官方建议的 **`--no-large-tensor`**（会少一些常量折叠机会）。
- **算子数量变化（摘录）**：`Constant` **313 → 166**；去掉 **`ConstantOfShape`、`Equal`、`Expand`、`Pow`、`Shape`、`Where`** 等；**`MatMul` 73、`Softmax` 12、`LayerNormalization` 28** 等与 ViT 结构一致的核心算子 **数量不变**。
- **模型体积**：**Original / Simplified 均为 375.4 MiB**（权重与大张量占主导，简化主要去冗余节点而非砍参数）。

### C. `trtexec` 建 engine 与解析 ONNX

| 项目 | 日志含义 |
|------|----------|
| 弱类型网络 | `[W] Weakly-typed networks have been deprecated`：TRT 10 的提示，**不表示失败**；长期可查阅 AutoCast / strongly typed 迁移。 |
| ONNX 元数据 | **IR 0.0.8**、**Opset 17**、**Producer: pytorch 2.0.0**，与导出脚本一致。 |
| 解析耗时 | `Parse time: ~0.237 s`。 |
| 建 engine | `Engine generation completed in ~14.73 s`，`Engine built in ~15.10 s`；**权重显存约 197 MiB**（`Total Weights Memory: 197184256 bytes`），**Activation ~5.2 MiB**。 |
| 磁盘 engine | 日志中 **`Created engine with size: 189.97 MiB`**；`ll` 显示 **`pem_rgb_net_b1_224_fp16.engine` 约 199198460 B（≈190 MiB）**，与上面一致量级。 |
| 结果 | **`&&&& PASSED TensorRT.trtexec`**，engine 已成功写出。 |

### D. 绑定形状与精度（与文档约定对齐）

- **输入**：`images`，**`1×3×224×224`**，`fp32`，**`CHW`**。
- **输出**：`dense_feat`，**`1×256×224×224`**，`fp32`（与 **`rgb_net(img)[0]`** 的 dense 特征图一致）。
- **构建精度**：日志为 **`FP32+FP16`**（`--fp16` 下 TRT 对层做混合精度选择，**I/O 仍为 fp32** 属常见配置）。

### E. `trtexec` 默认 benchmark 性能摘要（如何读）

日志 **`Performance summary`**（约 **3 s**、**977** 次有效计时；**200 ms warmup**）要点：

| 指标 | 数值（摘自 log） | 说明 |
|------|------------------|------|
| **Throughput** | **324.684 qps** | 含 **H2D / 计算 / D2H** 的端到端吞吐（默认 **开启 data transfers**）。 |
| **Latency（mean）** | **~3.92 ms** | **端到端 host 视角** 延迟，**不是**纯 GPU kernel 时间。 |
| **GPU Compute Time（mean）** | **~0.756 ms** | **GPU 上实际计算** 均值；与上行的 3.9 ms 差距大，说明时间主要不在 kernel。 |
| **D2H Latency（mean）** | **~3.07 ms** | **输出从设备拷回主机** 占大头；与 TRT 警告一致。 |
| **H2D Latency（mean）** | **~0.087 ms** | 输入上传相对较小。 |

**TRT 给出的结论（重要）**：

- **吞吐可能受 D2H（输出回传）限制，GPU 算力未必吃满**（日志：`* Throughput may be bound by device-to-host transfers...`）。dense 特征 **`1×256×224×224` fp32** 约 **49M 元素/帧**，PCIe 回读本身就很重；业务里若 **输出留在 GPU** 给下游 CUDA / 与 PEM 其它子图拼接，端到端会远好于「每帧拷全图回 CPU」的 `trtexec` 默认测法。
- **GPU 计算时间波动**：`coefficient of variance ≈ 3.83%`，日志建议 **锁 GPU 频率** 或 **`--useSpinWait`** 以稳定计时；对 **相对对比** 有意义，对「是否部署」无阻塞。

### F. 小结

- **`a.log` 表明**：ONNX 导出 → `onnxsim` → **`trtexec` 建 FP16 engine** 全流程 **已通过**；绑定与 **`dense_feat` 形状** 与 **`vit_acc.md` 前文描述一致**。
- **解读 benchmark 时**：默认 **mean ~3.9 ms** 是 **含大块输出 D2H** 的 `trtexec` 场景；**~0.76 ms** 更接近「仅 GPU 前向」的量级；与 **`pose_acc/README.md`** 里整机 PEM 时延对比时，注意是否包含 **回传与后处理**。

---

## 9. 将 `pem_rgb_net_b1_224_fp16.engine` 接入项目

本节说明：**在不大改 PEM 其它模块的前提下**，用已导出的 **`checkpoints/pem_rgb_net_b1_224_fp16.engine`** 替换 **`rgb_net` 的 dense 特征**（即原 **`self.rgb_net(img)[0]`**），使 **`get_chosen_pixel_feats` → `sample_pts_feats` → coarse/fine** 等后续逻辑保持不变。

### 9.1 代码入口（必须改动的唯一语义点）

`ViTEncoder` 里 **`get_img_feats`** 实现为（启用 **`SAM6D_PEM_RGB_TRT_ENGINE`** 且加载成功时走 TRT）：

```197:202:SAM-6D/Pose_Estimation_Model/model/feature_extraction.py
    def get_img_feats(self, img, choose):
        if self._pem_rgb_trt is not None:
            dense = self._pem_rgb_trt.forward_dense(img)
        else:
            dense = self.rgb_net(img)[0]
        return get_chosen_pixel_feats(dense, choose)
```

接入 TRT 时，把 **`self.rgb_net(img)[0]`** 换成 **与之一张量语义相同的 `dense_feat`**（形状见下节），**仍调用** `get_chosen_pixel_feats(dense_feat, choose)`。  
**不要**动 `get_chosen_pixel_feats` / `sample_pts_feats` / `forward` 里点云与模板几何逻辑；**不要**要求 TRT 直接输出「采样后点特征」，那是 PyTorch 里 `gather` 的事。

**仓库实现（与下文环境变量一致）**：**`trt/pem_rgb_trt.py`** 中 **`PemRgbNetTrt`**；**`model/feature_extraction.py`** 的 **`ViTEncoder`** 在 **`SAM6D_PEM_RGB_TRT_ENGINE`** 指向合法 `.engine` 时构造该 runner，并在 **`get_img_feats`** 中调用 **`forward_dense`**，否则仍走 **`self.rgb_net(img)[0]`**。（原设计备选 **A** 的环境变量开关 + **B** 的独立 TRT 模块已合并为该实现。）

### 9.2 与数据预处理对齐（否则数值全错）

模板与观测 RGB 在进入 `rgb_net` 前，在 `run_warmup_inference_custom.py` / `run_inference_custom.py` 中与 **`rgb_transform`** 一致：

- **`cv2.resize` 到 `cfg.img_size`（默认 224）**；
- **`transforms.ToTensor()`**（0–1）+ **`Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`**；
- 张量布局 **`N×3×224×224`**（CHW），已在 GPU 上。

TRT 子图是在 **上述分布的 `img`** 上导出的，接入时 **禁止** 再改归一化或通道顺序；若你自定义了别的预处理，需 **重新导出 ONNX/engine**。

### 9.3 Engine 的 I/O 契约（与 `trtexec` / ONNX 一致）

| 绑定名 | 方向 | 形状（本仓库导出） | dtype（日志） | 说明 |
|--------|------|-------------------|---------------|------|
| **`images`** | 输入 | **`1×3×224×224`** | **fp32** | 与 ONNX `export` 时 `input_names` 一致。 |
| **`dense_feat`** | 输出 | **`1×256×224×224`** | **fp32** | 即原 **`rgb_net(img)[0]`**，通道 **256**，空间 **224×224**。 |

构建时使用 **`--fp16`** 时，TensorRT 多为 **层内 FP16、I/O 仍为 fp32**（以你本地 `trtexec` 日志为准）；Python 侧绑定 **按 fp32 分配缓冲区** 即可，与当前 engine 一致。

### 9.4 Batch：固定 B=1 时的调用方式

当前 engine 按 **`export_pem_rgb_net_onnx.py`** 为 **固定 batch 1** 导出。

- **模板分支 `get_obj_feats`**：对每个视角调用 **`get_img_feats(tem, tem_choose)`**，其中 **`tem` 一般为 `1×3×224×224`**，与 engine 一致，**逐视角调用 TRT 即可**。
- **观测分支 `forward` → `get_img_feats(rgb, rgb_choose)`**：`get_test_data` 中 **`rgb` 可能为 `ninstance×3×224×224`**（多检测实例）。此时需 **按 batch 维循环**，每次喂 **`1×3×224×224`**，得到 **`1×256×224×224`**，再在 batch 维 **`torch.cat`** 成 **`ninstance×256×224×224`**，再交给 **`get_chosen_pixel_feats`**（该函数支持 **4D** `B×C×H×W`，内部会 reshape 为 `B×C×H*W` 再 `gather`）。

若希望 **单次 enqueue 多 B**，需 **重新导出 ONNX**（dynamic axis）并 **重建 engine**，不在当前固定形状 engine 范围内。

### 9.5 Python 侧集成步骤（建议顺序）

1. **依赖**：与建 engine 时一致安装 **`tensorrt`** Python 包；CUDA 设备与 **`CUDA_VISIBLE_DEVICES`** 与 PEM 一致。  
2. **进程内单例**：在加载 PEM **`Net`** 之后（例如 `preload_default_pem_model` / `run_pose_inference` 创建 `model` 之后），若设置了 **`SAM6D_PEM_RGB_TRT_ENGINE`**，则 **反序列化 engine 一次**，创建 **`IExecutionContext`**，可做 **1～2 次 dummy infer** 做 warmup。  
3. **设备内存**：为 **`images` / `dense_feat`** 分配 **GPU** 内存（`cudaMalloc` / `torch.cuda` 空张量 `.data_ptr()` 等均可），保证与 **`get_chosen_pixel_feats`** 后续张量 **同卡**。  
4. **推理**：`memcpy H2D`（若输入已在 GPU，可用设备侧拷贝）→ **`context.execute_async_v3` 或 v2**（视 TensorRT 版本 API）→ 得到 **`dense_feat`**。  
5. **少拷贝优化**：若 TRT 输出缓冲区直接是 **`torch.cuda.FloatTensor` 的 storage**（或 DLPack 互操作），可避免 **`dense_feat` 全量回 CPU 再上传**，与附录 E 中「D2H 限吞吐」的分析一致；**`get_chosen_pixel_feats` 只需 GPU 上索引**。  
6. **开关与回滚**：未设置环境变量或文件不存在时，走 **`self.rgb_net(img)[0]`**，保证未部署 TRT 的环境行为不变。

**启用示例**（在 `Pose_Estimation_Model` 目录下启动推理或 HTTP 前导出，路径可为绝对路径或相对 **仓库内 `Pose_Estimation_Model/`** 根目录）：

```bash
export SAM6D_PEM_RGB_TRT_ENGINE=checkpoints/pem_rgb_net_b1_224_fp16.engine
# 或: export SAM6D_PEM_RGB_TRT_ENGINE=/abs/path/to/pem_rgb_net_b1_224_fp16.engine
```

### 9.6 精度验收（上线前建议必做）

1. 固定同一 **`img`**（模板或观测裁剪后 **`1×3×224×224`**），比较 **`self.rgb_net(img)[0]`** 与 TRT **`dense_feat`** 的 **`abs().max()`** / 相对误差。  
2. 再对两路 **`dense`** 分别做 **`get_chosen_pixel_feats(..., choose)`**，比较 gather 后特征。  
3. 可选：全链路 **`run_pose_inference`** 对比 **`detection_pem.json`** / **`xyzrxryrz`**（与 `trt/README.md`、**`pose_acc/README.md` §3** 思路一致）。

### 9.7 与 HTTP 服务的关系

**`warmup_http_service.py` / `sam6d_http_service.py`** 通过 **`run_pose_inference`** 调 PEM；只要在 **同一进程** 内 **`ViTEncoder.get_img_feats`**（或其调用的 TRT 封装）已按上文初始化，**无需改 HTTP 路由**。注意：**模板 GPU 缓存**（`SAM6D_PEM_PRELOAD_TEMPLATES`）与 TRT engine **各自独立**，可同时开启。

### 9.8 常见坑

| 现象 | 可能原因 |
|------|----------|
| 特征噪声很大 / pose 飘 | 预处理与导出时不一致；或 engine 与当前 **`sam-6d-pem-base.pth` 权重** 不是同一套。 |
| TRT 报错 shape | 输入不是 **`1×3×224×224`**` fp32`；或多实例未按 §9.4 拆循环。 |
| 多进程服务每进程都要加载 engine | 每个 worker **各自反序列化** 一份 context；考虑 **启动阶段** 加载避免首次请求延迟。 |

---

## 10. 文档与脚本索引

- 导出 ONNX / SDPA 说明：**本文 §4**；脚本 **`Pose_Estimation_Model/export_pem_rgb_net_onnx.py`**。  
- **PEM rgb TensorRT 运行库**：**`trt/pem_rgb_trt.py`**（**`PemRgbNetTrt`**，环境变量 **`SAM6D_PEM_RGB_TRT_ENGINE`**）。  
- PEM 推理数据流与 HTTP：**`trt/PEM_INFER.md`**、**`pose_acc/README.md`**。  
- 本目录 **`a.log` 解析**：**本文附录**。
