# HTTPS 远程监控部署

当前网址为 https://shuaidao.online，香港服务器运行 Nginx 和共享视频转发。开发板完成全部实时识别；电脑网关共享 RGB565 接收与 JPEG 编码，SSH 隧道向云端提供视频和状态。

## 当前链路

`OV5640 / AC880 -> 电脑 127.0.0.1:8089 网关 -> SSH 隧道 -> 云端共享 MJPEG -> Nginx HTTPS -> 浏览器`

现有链路经北京服务器中继 SSH；两个云实例及电脑、开发板均为运行依赖。当前配置：网页 640×360、JPEG75、发送上限 15 FPS，显示与识别独立更新。

## 手动启动

1. 在开发板已部署目录执行 `./run.sh`，用 `./run.sh status` 查看状态，停止用 `./run.sh stop`。
2. 电脑执行 `scripts/start_home_cloud.ps1`。脚本检查网关及隧道是否已运行，避免重复启动；不设置开机自启。
3. 打开域名，使用自行配置的网页登录凭据。保持电脑、开发板和隧道运行。

新环境先调整脚本中的 Python、日志、密钥和中继主机路径。私钥与网页登录文件保存在源码外，不上传。SSH 使用现有 OpenSSH 和固定主机记录；没有对应私钥和 known_hosts 时不能直接复用部署命令。

## 文件

| 文件 | 用途 |
| --- | --- |
| nginx-public.conf.in | HTTPS、登录验证与代理模板 |
| nginx-challenge.conf.in | 首次证书域名验证模板 |
| ac880-home-public.service | Nginx 服务 |
| video_relay.py | 按需拉取一路上游，向观看端共享最新帧 |
| ac880-video-relay.service | 回环监听的视频转发服务 |
| scripts/deploy_cloud_public.py | 首次引导和 HTTPS 激活 |
| scripts/verify_cloud_public.py | 从外部检查 TLS、登录、控制令牌及视频 |

转发器只监听回环地址，由 Nginx 对外发布。慢客户端跳过旧帧；无观看者时停止持续拉流；断线后重连，过期画面不作新帧发送。视频、状态、资源及控制接口统一验证登录，控制操作另需令牌。

## 已有验证与限制

2026-10-29 检查通过：有效 TLS 证书、HTTP 到 HTTPS 跳转、8 条未登录路由 401、缺少控制令牌 403 和递增的实机视频帧。最终 60 秒公网接收样本约 9.76 FPS，约 1.13 Mbps 纯视频载荷；视频帧率不等于识别频率或完整动作延迟。

证书使用 Let's Encrypt 与 Certbot 定时续期。云实例、磁盘、公网流量和域名按产品规则计费；证书续期不等于服务器或域名续费。当前仍需扩大独立移动网络和长时间运行验证。
