# 代码导读

板端程序只保留一份，集中在 `board_runtime/app/`。`board_runtime/` 包含可直接部署的程序、配置、模型、bit/hwh 和 ARM NCNN 动态库。

## 板端运行链路

`run.sh → manage.py → fall_detect.py / realtime.py → 摄像头与帧缓冲 → 跌倒和手势任务 → 时序确认 → HDMI、LED、蜂鸣器及网页状态`。

| 模块 | 职责 |
| --- | --- |
| `manage.py` | 手动启动、进程锁、状态和安全停止 |
| `fall_detect.py` | VDMA、寄存器接入及报警状态机 |
| `realtime.py` | 硬件主循环、原始图像快照、显示叠加与控制 |
| `display_ui.py` | RGB565 状态栏、环境与手势信息、报警边框及缓存排版 |
| `fall_lite.py`、`ncnn_backend.py` | 输入预处理、候选解码、NCNN C 接口 |
| `shared_detector.py`、`semantic_features.py`、`semantic_decode.py` | 共享特征、区域复核、坐标恢复 |
| `tracked_fall.py` | 按目标确认报警，过滤重复与过期证据 |
| `dual_inference.py` | 按 SW0 选择任务，排空旧任务并隔离跨模式结果 |
| `recognition_mode.py` | SW0 读取、消抖、接口故障时暂停识别 |
| `hand_models.py`、`hand_geometry.py` | 掌部及 21 点关键点推理与坐标变换 |
| `hand_region_review.py` | 新鲜、高置信度手部区域局部复核 |
| `gesture_control.py` | 手势分类、稳定确认、单次触发与冷却 |
| `frame_copy.py`、`image_processing.py` | 视频搬运与 PL 图像处理驱动 |
| `environment.py` | 温湿度有效性检查和显示缓存 |
| `home_web.py`、`web/` | 板端网页、状态、视频和控制接口 |

表中模块位于 `board_runtime/app/`。`pose_rules.py` 是主程序仍需导入的兼容依赖；实际配置使用 NCNN 检测，不要求采集空背景。

## 硬件、网关与训练

- `hardware/current/`：当前顶层、视频与外设 RTL、Block Design、XDC 及工程重建脚本。
- `ac880_fall/scripts/`：电脑网关、SSH 隧道、云端部署和运行包构建工具。
- `ac880_fall/cloud/`：HTTPS、登录、共享视频转发及服务模板。
- `ac880_fall/ml/`：检测骨干、共享特征、区域复核训练及部署导出；工具先导入 `cache_env.py`，输出和缓存留在 E 盘。
- `ac880_fall/tests/`：当前视频、外设、调度、判决与网页的软件回归；`config.json` 只作测试夹具，部署配置为 `board_runtime/config.json`。

运行文件先用 `python tools/verify_runtime.py` 核验。启动、安全停止、网页和新机器部署见 [使用说明](reproduce.md)。
