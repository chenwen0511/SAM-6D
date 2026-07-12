# PEM 推理详细逻辑（结合流程图与代码）

本文基于以下两部分信息：

- 流程图：`pics/overview_sam_6d.png`、`pics/overview_pem.png`
- 代码实现：`Pose_Estimation_Model/run_warmup_inference_custom.py` 与 `Pose_Estimation_Model/model/*`

目标是把 PEM（Pose Estimation Model）推理链路拆成“输入 -> 特征 -> 粗匹配 -> 细匹配 -> 位姿输出”的可落地执行过程。

流程图展示：

### 图 1：SAM-6D 全流程概览

![SAM-6D overview](../../../pics/overview_sam_6d.png)

### 图 2：PEM 模块内部流程

![PEM overview](../../../pics/overview_pem.png)

---

## 1. PEM 在 SAM-6D 全链路里的位置

从 `overview_sam_6d.png` 看，SAM-6D 的主流程是：

1. Segment Anything（实例候选/分割）
2. Object Matching（目标类别/模板关联）
3. Coarse Point Matching（粗位姿）
4. Fine Point Matching（精位姿）

其中 3、4 两步由 PEM 完成，输入来自上游分割结果（`detection_ism.json`）和模板渲染结果（`templates/*.png, *.npy`）。

### 1.1 上游分割（ISM/YOLO）到 PEM 的输入过程

PEM 不直接做目标分割，它消费上游生成的检测结果文件（默认 `sam6d_results/detection_ism.json`）。

上游流程可概括为：

1. `Instance_Segmentation_Model/run_inference_custom.py` 或 YOLO 分支输出实例列表；
2. 每个实例至少包含：
   - `score`
   - `segmentation`（COCO RLE 或可解码结构）
   - 目标类别信息（如 `category_id`）
3. PEM 在 `get_test_data(...)` 中读取该 JSON，并按 `det_score_thresh` 过滤低分实例；
4. 对每个实例执行：
   - 解码 mask（`pycocotools.mask.decode`）
   - 与有效深度区域做交集（`mask && depth>0`）
   - 计算 bbox，裁剪出该实例对应的 RGB 与点云 ROI

因此，PEM 的输入不是“整图”，而是“上游筛选后的实例集合 + 每个实例的掩码约束”。

### 1.2 模板渲染到 PEM 的输入过程

PEM 依赖离线/在线渲染生成的模板目录（通常在 `output_dir/templates`），由 `Render/render_custom_templates.py` 生成。

模板目录中每个视角包含三类文件：

- `rgb_i.png`：该视角的模板外观
- `mask_i.png`：该视角的模板前景掩码
- `xyz_i.npy`：模板像素对应的 3D 坐标（用于构建模板点云）

在 PEM 中：

1. `get_templates(...)` 按 `n_template_view` 选取视角（默认从 42 视角均匀采样）；
2. `_get_template(...)` 对每个视角执行：
   - mask 求 bbox 并裁剪
   - RGB resize + 归一化
   - 基于 mask 采样模板点（`n_sample_template_point`）
   - 生成 `rgb_choose`（2D-3D 对齐索引）
3. 最终得到模板侧多视角特征输入，供后续 coarse/fine matching 使用。

可以把这一步理解为：先把 CAD 转成“多视角可匹配特征库”，PEM 推理时直接查询该特征库，而不是在线重建 CAD 几何。

---

## 2. 入口与热加载

在 `run_warmup_inference_custom.py` 中：

- `preload_default_pem_model()`：服务启动时预加载默认 PEM 权重（`sam-6d-pem-base.pth`）并缓存模型实例。
- `run_pose_inference(...)`：实际单次推理入口。

`run_pose_inference(...)` 的主要阶段：

1. 构建 cfg（`build_inference_cfg`）
2. 加载或复用模型（`_CACHED_DEFAULT_MODEL`）
3. 提取模板特征（`get_templates` + `model.feature_extraction.get_obj_feats`）
4. 读取观测数据（`get_test_data`）
5. 模型前向（`out = model(input_data)`）
6. 保存 `detection_pem.json` 和 `vis_pem.png`

---

## 3. 输入数据是如何构造的

### 3.1 模板分支（Object side）

`get_templates(path, cfg)` 读取每个模板视角的：

- `rgb_i.png`
- `mask_i.png`
- `xyz_i.npy`

处理步骤：

- 由 mask 求 bbox，裁剪 ROI
- 将 RGB resize 到网络输入尺寸（默认 224）
- 根据 mask 采样模板点（`n_sample_template_point`）
- 用 `get_resize_rgb_choose` 建立“点 -> 图像特征索引”的对应

输出为模板图像张量、模板点云、以及图像采样索引。

### 3.2 观测分支（Measured side）

`get_test_data(...)` 读取：

- 当前 RGB/Depth
- 相机内参 `cam_K` 和 `depth_scale`
- 上游分割结果 `seg_path`
- CAD 网格（`cad_path`）

关键步骤：

1. 用 mask + depth 有效值过滤实例像素
2. 由深度反投影得到观测点云
3. 根据 CAD 采样点云半径做几何裁剪（去离群）
4. 对每个实例采样固定数量观测点（`n_sample_observed_point`）
5. 构造 `input_data`：
   - `pts`：观测点云
   - `rgb`：裁剪后的观测图像
   - `rgb_choose`：图像特征索引
   - `score`：来自检测分数
   - `model`：CAD 采样点
   - `K`：相机内参

---

## 4. 网络前向主干（对应 overview_pem.png）

PEM 主网络在 `model/pose_estimation_model.py`，结构为：

1. `feature_extraction`
2. `coarse_point_matching`
3. `fine_point_matching`

### 4.1 Feature Extraction（图中左上）

输入：

- 观测 RGB + 点云
- 模板 RGB + 点云

输出：

- 观测 dense 点特征（`dense_pm, dense_fm`）
- 模板 dense 点特征（`dense_po, dense_fo`）

实现里使用 ViT 特征提取并做点级对齐采样（图像特征与 3D 点一一对应）。

### 4.2 Coarse Point Matching（图中右上）

目标：先得到一个可用的初始位姿。

核心逻辑（对应代码中的 coarse matching）：

- 构造观测/模板 sparse 点及特征
- 使用几何感知 transformer 做跨域匹配
- 生成匹配分数矩阵（correspondence score）
- 从高分匹配中采样/打分候选位姿
- 输出粗位姿 `init_R, init_t`

这一步的本质是：在大范围姿态空间快速收敛到“附近”。

### 4.3 Fine Point Matching（图中下半）

目标：在粗位姿基础上精修。

核心逻辑：

- 基于 `init_R, init_t` 将观测点对齐到模板附近
- 做更密集、更细粒度的匹配（sparse-to-dense）
- 用加权对应关系再求一次位姿
- 输出最终：
  - `pred_R`
  - `pred_t`
  - `pred_pose_score`

图中 `Weighted SVD` 对应最终刚体求解步骤（由匹配对/权重估计 R,t）。

---

## 5. 分数融合与输出

在 `run_pose_inference` 中：

- 若存在 `pred_pose_score`，最终分数为：
  - `pose_scores = pred_pose_score * detection_score`
- 否则回退用 `detection_score`

然后把每个 detection 写回：

- `score`
- `R`（3x3）
- `t`（mm）

保存到：

- `sam6d_results/detection_pem.json`
- `sam6d_results/vis_pem.png`

并返回 `best_detection`（最高分实例）。

---

## 6. 和 TensorRT 优化最相关的瓶颈点

从当前实现看，耗时主要集中在：

1. 模型前向（coarse + fine transformer）
2. 模板特征提取（每次请求如果不缓存会有额外开销）

已做优化：

- 默认模型预加载 + 缓存复用（减少权重加载成本）

进一步 TRT 建议：

- 优先尝试可导出的密集子图（特征提取/线性层堆叠）
- 点云匹配与几何求解路径先保留 PyTorch
- 最终走混合推理（TRT + PyTorch）逐步替换

---

## 7. 一句话总结

PEM 的核心是：  
**先用几何感知粗匹配找到初始姿态，再用细匹配 + 加权刚体求解精修姿态，最后融合检测置信度输出稳定的 6D 位姿结果。**

---

## 8. Baseline 测试结果（来自 `trt/baseline.csv`）

基于 `baseline.py`（预加载 + warmup 后多次测量）得到：

- mean: `469.624216 ms`
- median: `467.528242 ms`
- p95: `469.098053 ms`
- min: `466.429132 ms`
- max: `529.504554 ms`
- std: `11.140928 ms`

样本数：`30` 次（`run_index=1..30`）。

补充观察：

- 大多数样本稳定在 `466~469 ms` 区间；
- 第 4 次出现一次高值 `529.504554 ms`，是当前方差主要来源；
- 若用于 TRT 前后对比，建议继续采样到 `N>=100` 并同时记录 GPU 温度/功耗，避免单次抖动影响结论。
