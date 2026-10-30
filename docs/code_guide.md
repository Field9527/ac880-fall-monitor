# 代码导读

## 当前运行链路

`manage.py -> realtime.py -> 相机和帧缓冲 -> 摔倒及手势任务 -> 按目标判决与手势确认 -> HDMI、外设、网页和事件记录`。

| 模块 | 职责 |
| --- | --- |
| `sw/manage.py` | 手动启动、进程锁、状态和安全停止 |
| `sw/fall_detect.py` | VDMA、寄存器接入和四级时序判决；旧 CNN 接口仅为兼容保留 |
| `sw/realtime.py` | 硬件主循环、原图快照、显示、报警与控制应用 |
| `sw/fall_lite.py` | 输入预处理、候选解码和最新帧任务 |
| `sw/ncnn_backend.py` | NCNN C 接口、输出布局与内存所有权 |
| `sw/shared_detector.py` | 当前 v6 ROI 复核分支与识别设置 |
| `sw/semantic_features.py` | 共享特征描述、ROI 复核器和坐标变换 |
| `sw/dual_inference.py` | 摔倒、手势独立最新帧调度 |
| `sw/hand_models.py`、`hand_geometry.py` | 掌部、21 点与坐标恢复 |
| `sw/hand_region_review.py` | 新鲜、高置信度手部的局部复核 |
| `sw/tracked_fall.py` | 各目标独立推进报警状态 |
| `sw/gesture_control.py` | 手势分类、稳定确认、单次触发及冷却 |
| `sw/frame_copy.py`、`image_processing.py` | 视频搬运和 PL 图像处理驱动 |
| `sw/environment.py` | 有效温湿度读取与显示缓存 |
| `sw/home_web.py`、`sw/web/` | 网页服务、共享视频和控制界面 |
| `scripts/lan_gateway.py` | 电脑共享 RGB565 接收与 JPEG 编码 |
| `cloud/video_relay.py` | 云端按需共享 MJPEG，不运行识别 |

路径以 `ac880_fall/` 为起点。实际打包模块由 `scripts/build_semantic_runtime.py` 的模块列表确定，不需将全部开发脚本复制到开发板。

## 训练与导出

`ml/train_fall_lite.py` 训练基础检测器；`shared_semantic_model.py`、`cache_shared_features.py` 和 `train_shared_heads.py` 提供冻结骨干与区域复核工具；`cache_camera_augmented_rois.py`、`refine_region_verifier.py` 处理相机取景及负例复核；`export_fall_region.py` 导出板端模型。`prepare_gesture_ncnn.py` 转换掌部和关键点模型。

训练入口先导入 `cache_env.py`，实际图像数据、划分、特征缓存和检查点放在 E 盘，不包含在源码包中。监督掩码、样本来源和开发集边界属于训练逻辑，不能为了简化文件而删除。五姿态数据不属于当前项目输入。

## 测试与运行配置

`sw/test_*.py` 是无需连接开发板的软件回归；`scripts/validate_*.py`、`benchmark_*.py`、`regress_*.py` 用于实机或样本验证，按各自参数执行，不能把这些脚本当作日常启动入口。

`sw/config_current.json` 为当前配置；同目录其余配置还被兼容性测试引用，保留作回归夹具，不作为当前启动设置。

`pose_rules.py` 的空背景前景规则、`semantic_policy.py`、`semantic_probe.py` 及 Nano/通用物体分支是兼容或开发研究代码，当前配置不以它们替代最终报警策略。硬件入口为 `hardware/current/`。

## 注释与维护

自编运行模块使用简洁中文说明接口、边界与时序，保留英文类名和变量名，避免破坏已有导入。Python 排版按统一缩进和行宽整理；本轮不更改阈值、控制逻辑、硬件地址或模型权重。第三方原始版权、许可及来源说明继续保留。
