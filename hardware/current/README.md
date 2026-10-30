# 当前视频与外设硬件

目标为 Vivado 2025.2、xc7z020clg484-2。Sources/RTL 保存当前顶层与视频、环境、外设逻辑，Sources/BD/system.tcl 保存已经移除旧卷积端点的 Block Design，Sources/Constraints 保存本次硬件沿用的约束。

在新的 E 盘输出目录重建工程：

```powershell
vivado -mode batch -source hardware/current/Scripts/build.tcl -tclargs E:/Zynq_cache/temp/ac880_rebuild
```

入口只重建与校验项目，不启动综合、实现或下载开发板。重建后重新执行综合与实现，才能获得该次构建的 bit/hwh、资源和时序结果。原布线报告中仍有方法学及 CDC 问题；正裕量仅对应已约束路径。

本目录包含与运行文件对应的顶层、BD、外设逻辑及约束。重建时保持模块、引脚和时钟关系一致。
