# My FoundationPose Workflow (Lyn)

本文档说明了基于现有代码框架的数据录制、位姿估计、滤波平滑以及结果分析的可复现流程。在每一步操作时，请确保位于 `FoundationPose` 工作空间根目录。

## 整体流程概述
1. **数据捕获 (Data Capture)**: 使用 Kinect 相机录制对齐的 RGB、Depth 数据及相机内参 (`cam_K.txt`)。
2. **位姿估计/跟踪 (Pose Estimation)**: 运行 `lyn_run_demo.py`，结合录制的数据和 3D 模型 (.obj) 进行 6DoF 位姿追踪。
3. **姿态滤波 (Pose Filtering)**: (当存在抖动时) 使用 `pose_filter.py` 提供 SE(3) 扩展卡尔曼滤波进行平滑处理。
4. **结果分析与评估 (Result Analysis)**: 运行 `pose_analyzer.py` 对估计好的位姿进行可视化渲染与量化评估。

---

## 阶段1：数据捕获
**相关脚本**：`tools/capture_kinect_data_v4.py` (或其他 `capture_kinect_*.py` 脚本)

**执行流程**：
使用 Azure Kinect (PyK4A) 录制高精度数据点并完成去畸变处理。
内部指定了分辨率以及深度模式，并在录制时自动生成对齐后的图片。
一般执行指令为（进入目录或直接运行）：
```bash
python tools/capture_kinect_data_v4.py
```

**数据位置与结构**：
录制到的数据通常保存在 `FoundationPose_manual/` 的子目录下（例如：`FoundationPose_manual/kinect_data_with_calibrated_K/`）。
标准的数据结构必须包含以下内容：
```
FoundationPose_manual/<scene_name>/
├── rgb/                # 对应时间戳/序号的RGB图像 (*.jpg 或 *.png)
├── depth/              # 对齐（或变换后）的深度图像 (*.png)
└── cam_K.txt           # 3x3 相机内参矩阵
```

---

## 阶段2：位姿估计(跟踪)
**相关脚本**：`lyn_run_demo.py`

**配置与修改**：
如果录制了新数据或是使用了新的 Mesh 模型，可以在 `lyn_run_demo.py` 的参数解析处修改对应的默认路径：
- `mesh_file`: 你的 .obj 三维模型路径（当前默认：`FoundationPose_manual/geminiHead/mesh/new_head_fp_canonical.obj`）
- `test_scene_dir`: 你的场景数据文件夹路径（通常为阶段 1 中产生或整理存放的文件夹，例如 `FoundationPose_manual/kinect_data_with_calibrated_K`）
- `debug_dir`: 姿态结果与中间可视化的输出路径。

**执行指令**：
```bash
python lyn_run_demo.py
```

**输出数据位置**：
生成的 6DoF 位姿 (.txt文件，名称为对应的帧编号) 会被输出在设定好的 `debug_dir/ob_in_cam/` 目录下（例如 `debug_kinect_data_with_calibrated_K_mesh/ob_in_cam/`），同时可以查看到可视化绘制的投影边缘 (`track_vis`)。

---

## 阶段3：姿态滤波（可选操作）
**相关脚本**：`pose_filter.py`

**主要作用**：
当由于深度噪点等原因导致输出的网格/姿态发生微小抖动 (Jittering) 或闪刺时，可以调用封装好的 `<SE3KalmanFilter>` 对输出的随时间变化的 6DoF 位姿流进行平滑。

*注：当前通常作为一个模块，被融合引入主程序或分析模块中进行离线处理。*

---

## 阶段4：分析与评估
**相关脚本**：`pose_analyzer.py`

**执行流程**：
用于读取计算出的姿态、原始的 RGB-D 数据和测试网格，进行误差的量化评估（例如基于一定的 Fitness Threshold 求指标），或是对追踪序列完成整体检验与视频生成。

代码内的参数配置可根据需要更改，如 `test_scene_dir`、`est_pose_dir` 和 `output_dir`：
```python
# 示例配置对应项
file_dir = 'kinect_data_with_calibrated_K'
```

**执行指令**：
```bash
python pose_analyzer.py
```

**结果位置**：
评估的结果、生成的图表与渲染图片将存放在配置项 `output_dir` 所代表的目录，如：
`FoundationPose_manual/kinect_data_with_calibrated_K/analysis_results/` 
