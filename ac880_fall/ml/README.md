# 训练与模型导出

当前板端为 v6 摔倒骨干、refined_v5 区域复核及 v4 参考复核器，使用 FP32 ARM NCNN。手势采用固定版本的掌部检测和 21 点模型。全部实时识别在 AC880 上；本目录仅用于离线训练、审计与导出。

## 主要入口

| 文件 | 作用 |
| --- | --- |
| cache_env.py | 在模型库导入前配置 E 盘缓存 |
| train_fall_lite.py、fall_lite_model.py | 检测骨干训练与固定形状导出 |
| cache_shared_features.py、shared_semantic_model.py | 冻结骨干特征及新增分支 |
| train_shared_heads.py | 训练共享特征上的复核头 |
| cache_camera_augmented_rois.py | 训练侧相机宽高比与 RGB565 增强 |
| refine_region_verifier.py | 仅训练复核器，检查既有倒地保留情况 |
| export_fall_region.py | 导出摔倒候选与复核所需特征 |
| prepare_gesture_ncnn.py | 掌部及关键点网络转换 |

本目录只保留当前检测骨干、区域复核、手势模型的训练和导出工具。各工具的输入输出参数见 `--help`。

## 数据与评估边界

原始训练数据为 `shuaidao/FallDataset/FallDataset`。缺失 XML/TXT 不等于正常场景，未知标注应排除或按监督掩码处理。沿用已记录的分组和验证边界，避免通过重新随机划分引入泄漏；近重复图和同场景风险需另行审计。

相机手部及局部人体负例只用于对应训练或开发回归。辅助视觉复核不是独立人工真值，应保留数据来源、复核状态、SHA256 及监督掩码。代码中的历史来源标识用于兼容已有索引，不能移除后将样本宣称为独立标注。训练重放和反复查看的验证集不作为新的独立验收。

补充数据与模型许可见 `../../board_runtime/assets/DATA_SOURCES.md`、`../../board_runtime/assets/licenses/` 及模型来源清单。五姿态数据不属于当前项目输入。完整图像、特征缓存及检查点未放入源码包；运行包包含成套部署模型，不要求重新训练。

## 缓存与部署

模型库导入前先导入 `cache_env.py`。缓存统一为 `E:/Zynq_cache`，临时目录为 `E:/Zynq_cache/temp`；数据、训练输出和构建文件留在 E 盘工作区。

模型输入为 320×192 RGB，等比补边和归一化需与部署一致。导出后核对 ONNX、NCNN 图、权重、输出布局和 ARM 动态库哈希，再进行数值对照。转换通过不代表识别准确率通过；实际运行配置见 `../../board_runtime/config.json`，打包入口见 `scripts/build_semantic_runtime.py`。
