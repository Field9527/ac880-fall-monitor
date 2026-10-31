# 当前视频与外设硬件

目标为 Vivado 2025.2、xc7z020clg484-2。Sources/RTL 保存当前顶层与视频、环境、外设逻辑，Sources/BD/system.tcl 保存已经移除旧卷积端点的 Block Design，Sources/Constraints 保存本次硬件沿用的约束。

在新的 E 盘输出目录重建工程：

```powershell
vivado -mode batch -source hardware/current/Scripts/build.tcl -tclargs E:/Zynq_cache/temp/ac880_rebuild
```

入口默认只重建与校验项目。Sources/IP 的两个脚本重建系统与 HDMI 时钟 IP，时钟参数保持不变。追加 `bitstream` 参数可生成配置文件及资源、时序报告：

```powershell
vivado -mode batch -source hardware/current/Scripts/build.tcl -tclargs E:/Zynq_cache/temp/ac880_build bitstream
```

输出 bit 位于 `ac880_video.runs/impl_1/fall_top.bit`，对应 hwh 位于 `ac880_video.gen/sources_1/bd/system/hw_handoff/system.hwh`，部署时成对复制并更新运行包校验清单。脚本不连接或下载开发板。已有方法学及 CDC 问题仍需独立处理；正裕量仅对应已约束路径。

本目录包含与运行文件对应的顶层、BD、外设逻辑及约束。重建时保持模块、引脚和时钟关系一致。

用户拨码 SW0 接 M16，经过 `home_io` 两级同步进入 AXI 域。`motion_regfile` 的 0x110 为 SWM1 能力标识，0x114 的 bit0 为 SW0 电平，软件负责 150 毫秒消抖。独立自检包含同步延迟、两个电平和 AXI 读反压：

```powershell
vivado -mode batch -source hardware/current/Scripts/simulate_switch.tcl -tclargs E:/Zynq_cache/temp/ac880_switch_sim
```

通过标记为 `AC880_SWITCH_SIMULATION_PASS`，仿真不代替实际拨动、HDMI 和外设验收。
