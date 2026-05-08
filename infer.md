# SAM-6D 本地调用命令

## 1) 健康检查

```bash
curl "http://127.0.0.1:8000/health"
```

## 2) 使用默认分割后端（sam6d_ism）

```bash
curl -X POST "http://127.0.0.1:8000/infer" \
  -F "rgb=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png" \
  -F "depth=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png" \
  -F "camera=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json" \
  -F "segmentor_model=sam" \
  -F "det_score_thresh=0.3"
```

## 3) 使用 YOLO 分割后端（可选）

```bash
curl -X POST "http://127.0.0.1:8000/infer" \
  -F "rgb=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png" \
  -F "depth=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png" \
  -F "camera=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json" \
  -F "seg_backend=yolo_seg" \
  -F "yolo_weights=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.pt" \
  -F "yolo_conf=0.25" \
  -F "yolo_imgsz=640" \
  -F "yolo_class_id=0" \
  -F "det_score_thresh=0.0"
```

## 4) 仅输出关键字段（可选，需安装 jq）

```bash
curl -s -X POST "http://127.0.0.1:8000/infer" \
  -F "rgb=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png" \
  -F "depth=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png" \
  -F "camera=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json" \
  -F "segmentor_model=sam" \
  -F "det_score_thresh=0.3" | jq '{score, xyz_mm, rotation_euler_zyx_rad, xyzrxryrz, timing}'
```

## 5) 实际调用结果记录（2026-05-07）

### health 返回

```json
{
  "status": "ok",
  "root_dir": "/home/mui/projects/smt/SAM-6D/SAM-6D",
  "env_dir": "/home/mui/.micromamba/envs/sam6d",
  "cad_path": "/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/models/tray_180mm_centered_mesh_v2.ply",
  "cad_exists": true,
  "output_root": "/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs",
  "default_seg_backend": "yolo_seg",
  "yolo_weights": "/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/yolo_runs/tray_seg/weights/best.pt",
  "yolo_weights_exists": true
}
```

### infer 结果摘要

- score: `0.8672866225242615`
- xyz_mm: `[-39.67422866821289, -42.124290466308594, 418.56719970703125]`
- rotation_euler_zyx_rad: `[-1.0479800707645823, -1.2273715248092116, 2.596818143045228]`
- xyzrxryrz(mm+rad): `[-39.67422866821289, -42.124290466308594, 418.56719970703125, -1.0479800707645823, -1.2273715248092116, 2.596818143045228]`

### 输出文件

- result_dir: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99`
- detection_ism_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/sam6d_results/detection_ism.json`
- detection_pem_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/sam6d_results/detection_pem.json`
- vis_ism_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/sam6d_results/vis_ism.png`
- vis_pem_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/sam6d_results/vis_pem.png`

### 耗时（秒）

- upload_s: `0.000293968987534754`
- templates_s: `0.00007770099909976125` (模板缓存命中)
- yolo_s: `0.1684631289972458`
- pose_s: `4.143830660992535`
- pipeline_s: `4.31237149098888`
- total_s: `4.312665459976415`

## 6) 不走 HTTP，直接运行实例分割脚本

### 与 `sam6d_http_service.py` 199-220 等价命令

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/Instance_Segmentation_Model && \
python run_inference_custom.py \
  --segmentor_model sam \
  --output_dir /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/manual_run_$(date +%Y%m%d_%H%M%S) \
  --cad_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/models/tray_180mm_centered_mesh_v2.ply \
  --rgb_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png \
  --depth_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png \
  --cam_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json
```

### 指定 `sam6d` 环境 python（更稳）

```bash
cd /home/mui/projects/smt/SAM-6D/SAM-6D/Instance_Segmentation_Model && \
/home/mui/.micromamba/envs/sam6d/bin/python run_inference_custom.py \
  --segmentor_model sam \
  --output_dir /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/manual_run_$(date +%Y%m%d_%H%M%S) \
  --cad_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/models/tray_180mm_centered_mesh_v2.ply \
  --rgb_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png \
  --depth_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png \
  --cam_path /home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json
```

## 7) 不走 HTTP，直接运行位姿估计脚本（PEM）

### 常见报错与原因

- 报错: `unrecognized arguments: --segmentor_model sam`
  - 原因: `--segmentor_model` 是实例分割脚本参数，不是 PEM 参数。
  - 修复: PEM 使用 `--seg_path` 传入分割结果 JSON。

- 报错: `FileNotFoundError: .../templates/rgb_0.png`
  - 原因: `--output_dir` 下缺少 `templates`，或软链接建在 A 目录但运行时用了新的 B 目录（时间戳变了）。
  - 修复: 使用同一个固定 `OUT` 变量，先准备 `OUT/templates` 再运行 PEM。

### 复用已有模板（方案2）+ 直接跑 PEM（已验证可用）

```bash
OUT=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/manual_run_$(date +%Y%m%d_%H%M%S)
SRC=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/templates
CAD=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/models/tray_180mm_centered_mesh_v2.ply
RGB=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png
DEPTH=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png
CAM=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json
SEG=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/sam6d_results/detection_ism.json

mkdir -p "$OUT"
ln -s "$SRC" "$OUT/templates"

# 可选自检（确认模板可访问）
ls -l "$OUT/templates/rgb_0.png"

cd /home/mui/projects/smt/SAM-6D/SAM-6D/Pose_Estimation_Model && \
python run_inference_custom.py \
  --output_dir "$OUT" \
  --cad_path "$CAD" \
  --rgb_path "$RGB" \
  --depth_path "$DEPTH" \
  --cam_path "$CAM" \
  --seg_path "$SEG" \
  --det_score_thresh 0.0
```

### 带开始/结束时间与总耗时打印（推荐）

```bash
OUT=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/manual_run_$(date +%Y%m%d_%H%M%S)
SRC=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/templates
CAD=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/models/tray_180mm_centered_mesh_v2.ply
RGB=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png
DEPTH=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png
CAM=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json
SEG=/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_194839_79feda99/sam6d_results/detection_ism.json

mkdir -p "$OUT"
ln -s "$SRC" "$OUT/templates"
ls -l "$OUT/templates/rgb_0.png"

START_TS=$(python - <<'PY'
import time
print(f"{time.perf_counter():.6f}")
PY
)
echo "[PEM] start: $(date '+%F %T.%3N')"

cd /home/mui/projects/smt/SAM-6D/SAM-6D/Pose_Estimation_Model && \
python run_inference_custom.py \
  --output_dir "$OUT" \
  --cad_path "$CAD" \
  --rgb_path "$RGB" \
  --depth_path "$DEPTH" \
  --cam_path "$CAM" \
  --seg_path "$SEG" \
  --det_score_thresh 0.0
RET=$?

END_TS=$(python - <<'PY'
import time
print(f"{time.perf_counter():.6f}")
PY
)
echo "[PEM] end:   $(date '+%F %T.%3N')"
python - <<PY
start = float("$START_TS")
end = float("$END_TS")
print(f"[PEM] elapsed_s: {end - start:.3f}")
print(f"[PEM] elapsed_ms: {(end - start)*1000:.1f}")
PY
echo "[PEM] exit_code: $RET"
```

### 关键注意点

- `ln -s` 和 `python run_inference_custom.py` 必须使用同一个 `OUT` 变量。
- 不要在 `--output_dir` 里再次写 `manual_run_$(date ...)`，否则会和建链接目录不一致。
- 你的 `SEG` 来自 YOLO 后端时，建议 `--det_score_thresh 0.0`（与服务逻辑一致）。

## 8) PEM 权重加载位置说明

`Pose_Estimation_Model/run_inference_custom.py` 推理时会涉及两个 `.pth`：

- 主权重：`Pose_Estimation_Model/checkpoints/sam-6d-pem-base.pth`
- MAE 预训练权重：`Pose_Estimation_Model/checkpoints/mae_pretrain_vit_base.pth`

### 8.1 主权重（显式加载）

在 `Pose_Estimation_Model/run_inference_custom.py` 主流程中直接加载：

```python
checkpoint = os.path.join(os.path.dirname((os.path.abspath(__file__))), 'checkpoints', 'sam-6d-pem-base.pth')
gorilla.solver.load_checkpoint(model=model, filename=checkpoint)
```

说明：这里是把 PEM 的训练权重加载到 `MODEL.Net(cfg.model)` 实例中。

### 8.2 MAE 权重（模型初始化时隐式加载）

`run_inference_custom.py` 在创建模型时会触发特征提取模块初始化：

```python
MODEL = importlib.import_module(cfg.model_name)
model = MODEL.Net(cfg.model)
```

随后进入 `Pose_Estimation_Model/model/feature_extraction.py` 的 `ViT_AE.__init__`。
当配置 `pretrained: True` 时，会在该处加载：

```python
vit_checkpoint = os.path.join('checkpoints', 'mae_pretrain_'+ self.vit_type +'.pth')
checkpoint = torch.load(vit_checkpoint, map_location='cpu')
msg = self.vit.load_state_dict(checkpoint_model, strict=False)
```

说明：默认配置 `vit_type: vit_base`，因此对应文件是 `mae_pretrain_vit_base.pth`。


(sam6d) mui@ubuntu-System-Product-Name:~/projects/smt/SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints$ ls -lh
总计 1.6G
-rw------- 1 mui mui 328M  4月 27 18:58 mae_pretrain_vit_base.pth
-rw-rw-r-- 1 mui mui 1.3G  3月  1  2024 sam-6d-pem-base.pth


## 9) Warm 方案改动与调用记录（2026-05-08）

### 9.1 改动方案（Warm）

- 新增 PEM 可导入接口脚本：`SAM-6D/Pose_Estimation_Model/run_warmup_inference_custom.py`
  - 将原 CLI 推理流程封装成可被 HTTP 服务直接调用的函数。
  - 支持默认模型/默认 checkpoint 的预加载与缓存复用（减少重复加载耗时）。
  - 增加预加载日志（环境、路径、缓存命中、CUDA信息、异常阶段）。
- `SAM-6D/warmup_http_service.py` 采用 warm 方式调用 PEM（不再每次都重新初始化完整 PEM 环境）。
- 接口保持与原服务一致：`POST /infer` 的输入输出字段不变。

### 9.2 本次调用命令

```bash
curl -X POST "http://127.0.0.1:8001/infer" \
  -F "rgb=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/rgb.png" \
  -F "depth=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/depth.png" \
  -F "camera=@/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260507_103518_e7ebc86f/inputs/camera.json" \
  -F "segmentor_model=sam" \
  -F "det_score_thresh=0.3"
```

### 9.3 返回结果（摘要）

- score: `0.7004115581512451`
- xyz_mm: `[-35.76496505737305, -50.191524505615234, 419.96319580078125]`
- rotation_euler_zyx_rad: `[-2.113831981043488, 1.3517449756622901, -0.6791343908055002]`
- xyzrxryrz(mm+rad): `[-35.76496505737305, -50.191524505615234, 419.96319580078125, -2.113831981043488, 1.3517449756622901, -0.6791343908055002]`

### 9.4 输出目录与文件

- result_dir: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260508_101743_b16ac733`
- detection_ism_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260508_101743_b16ac733/sam6d_results/detection_ism.json`
- detection_pem_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260508_101743_b16ac733/sam6d_results/detection_pem.json`
- vis_ism_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260508_101743_b16ac733/sam6d_results/vis_ism.png`
- vis_pem_path: `/home/mui/projects/smt/SAM-6D/SAM-6D/user_data/outputs/20260508_101743_b16ac733/sam6d_results/vis_pem.png`

### 9.5 耗时（秒）

- upload_s: `0.00031983799999579787`
- templates_s: `0.00028569900314323604`
- yolo_s: `0.1617757609928958`
- pose_s: `0.6119539139908738`
- pipeline_s: `0.7740153739869129`
- total_s: `0.7743352119869087`

### 9.6 对比说明

- 本次 `pose_s` 降到约 `0.612s`，相较于之前记录的 `4.14s` 明显降低。
- 当前返回中 `ism_s: null`、`yolo_s` 有值，说明此次分割后端走的是 YOLO 分支。
