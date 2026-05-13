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

推理里 **`get_img_feats` 只使用 `self.rgb_net(img)[0]`**，`cls_tokens` 未参与后续。建议用 **Wrapper 只导出第一路输出**，减小图、降低 TRT 解析失败概率。

在 `Pose_Estimation_Model` 下将下面保存为 **`export_pem_rgb_net_onnx.py`**（或整段用 `python -` 执行），与步骤一相同 **`load_checkpoint` 后的 `rgb_net`**：

```python
import os, sys, torch, importlib
import gorilla

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
for p in ("provider", "utils", "model", os.path.join("model", "pointnet2")):
    sys.path.insert(0, os.path.join(ROOT, p))
sys.path.insert(0, ROOT)

class DenseOnly(torch.nn.Module):
    def __init__(self, rgb_net):
        super().__init__()
        self.rgb_net = rgb_net

    def forward(self, x):
        dense, _cls = self.rgb_net(x)
        return dense

def main():
    cfg = gorilla.Config.fromfile(os.path.join(ROOT, "config", "base.yaml"))
    cfg.model_name = "pose_estimation_model"
    cfg.gpus = "0"
    gorilla.utils.set_cuda_visible_devices(gpu_ids="0")

    MODEL = importlib.import_module(cfg.model_name)
    net = MODEL.Net(cfg.model).cuda().eval()
    ckpt = os.path.join(ROOT, "checkpoints", "sam-6d-pem-base.pth")
    gorilla.solver.load_checkpoint(model=net, filename=ckpt)

    m = DenseOnly(net.feature_extraction.rgb_net).cuda().eval()
    dummy = torch.randn(1, 3, 224, 224, device="cuda", dtype=torch.float32)

    onnx_path = os.path.join(ROOT, "checkpoints", "pem_rgb_net_b1_224.onnx")
    torch.onnx.export(
        m,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["dense_feat"],
        opset_version=17,
        do_constant_folding=True,
    )
    print("ONNX:", onnx_path)

if __name__ == "__main__":
    main()
```

执行：

```bash
cd /path/to/SAM-6D/SAM-6D/Pose_Estimation_Model
python export_pem_rgb_net_onnx.py
```

（若脚本放在别处，把 `ROOT = os.path.dirname(...)` 改成 `ROOT = "/path/to/.../Pose_Estimation_Model"`。）

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
| ONNX 导出报错 | 提高/降低 `opset_version`；确认 `m.eval()`；尝试 `onnxsim` |
| `trtexec` parser 失败 | 换简化后 ONNX；查 TensorRT 版本与 PyTorch/CUDA 矩阵 |
| FP16 误差偏大 | ONNX 保持 FP32，仅 engine 用 `--fp16`；或缩小验收阈值后接受 |

---

## 9. 接入推理（后续工作，本文不展开）

在 **`get_img_feats`** 中用 TRT 输出替换 **`self.rgb_net(img)[0]`**，其余 **`get_chosen_pixel_feats` / `sample_pts_feats`** 仍走 PyTorch；建议 **环境变量开关**，便于与当前 PyTorch 路径 A/B 及 `pose_acc/README.md` 中的 **`get_obj_feats` 耗时** 对比。
