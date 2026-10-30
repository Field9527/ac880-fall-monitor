# 复现与部署说明

## 环境与文件

需要 AC880、OV5640、HDMI 显示器、DHT11、可用的板载 LED/蜂鸣器、网线及电源。板端沿用已验证的 PYNQ/Linux 环境，需 NumPy、OpenCV、Pillow；NCNN 使用配套 ARMv7 动态库，不用安装不匹配的推理 wheel。电脑网关需要 Python、NumPy、OpenCV、Pillow、requests 和现有 OpenSSH。

参赛包中的 `board_runtime/` 包含当前程序、配对 bit/hwh、refined_v5 与 v4 参考复核器、掌部及关键点模型、NCNN、许可和静态文件 SHA256 清单。保留源码目录原有相对结构，训练模型和运行时文件可通过 `scripts/build_semantic_runtime.py` 重新打包。

## 手动运行

1. 断电完成 OV5640、DHT11、HDMI 和网线连接。DHT11 VCC 接 P7-29，GND 接 P7-30，DATA/S 接 P7-31，核对模块丝印和 3.3V 供电。
2. 启动板端系统，连接到开发板，确认没有其他进程占用摄像头或 DMA。
3. 当前已部署目录运行 `./run.sh`；用 `./run.sh status` 查询，停止时执行 `./run.sh stop`。不要通过强杀硬件主线程切换运行目录。
4. 新安装时将完整运行包放到约定目录，核对 `config.json` 中的 overlay、日志及模型路径。先校验 manifest，再启动；不能运行中覆盖文件。
5. 局域网网页由电脑网关转发，远程网页还需要云服务器和 SSH 隧道，具体见 cloud 说明。凭据自行配置，不使用仓库中的示例路径代替密钥。

当前配置为摔倒与手势并行、固定核关闭、显示上限 12 FPS、网页原始视频宽度 640、发送上限 15 FPS。网关以 JPEG60 编码。手势确认窗口为 0.5 秒且至少两个新样本，窗口不是完整动作延迟。

## 从源码重建

硬件：在新的 E 盘目录执行 `hardware/current/Scripts/build.tcl`，独立重建项目并验证 BD；输出 `AC880_SOURCE_REBUILD_PASS` 表示重建检查通过。需要重新综合和实现才能得到新硬件结果，不能套用已有 bit 的裕量。

运行包：在源码根目录执行以下命令，路径替换为自己的文件位置：

```powershell
python ac880_fall/scripts/build_semantic_runtime.py E:/build/runtime --model E:/models/fall_regions_refined_v5 --gesture E:/models/gesture_mediapipe_v1 --library E:/models/libncnn_semantic.so --hardware E:/build/semantic_video.bit --config ac880_fall/sw/config_current.json --board-root /home/xilinx/fall_gesture_trial
```

模型、硬件与库必须是成套校验版本。打包程序核验模型和库哈希，拒绝已知不匹配的组合，不会自动部署或下载开发板。

## 验证顺序

先执行 `sw/test_*.py` 软件回归，再核对模型与库数值、bit/hwh、实机视频和外设回执，最后做真实人员独立场景测试及远程负载测试。重复帧、过期结果、误报、漏报及完整动作时间应单独统计。手机屏幕回放仅作集成测试，训练重放不作为独立现场验收。

## 当前边界

仍有手部误报、部分跌倒漏报和手势偏慢；DHT11 湿度校准待完善；硬件方法学与 CDC 问题未完整关闭。当前程序采用手动启动，不启用开机自动识别。
