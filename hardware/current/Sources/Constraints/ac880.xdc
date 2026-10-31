################################################################
# AC880 M1 - 约束文件
# 器件: xc7z020clg484-2
################################################################

# ==============================================================
# 1. 时钟约束
# ==============================================================

# 板载 50MHz 晶振
create_clock -name clk_50m_in -period 20.000 [get_ports clk_50m_in]

# 摄像头像素时钟 (OV5640 1280x720@60fps, PCLK ≈ 74.25MHz)
create_clock -name camera_pclk -period 13.468 [get_ports camera_pclk]

# 摄像头 PCLK 为普通 IO 输入，非时钟专用引脚
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets camera_pclk_IBUF]

# ==============================================================
# 2. 跨时钟域 False Path
# ==============================================================

# DVP 采集域 (cam_pclk) ↔ 系统域 (100MHz)
set_false_path -from [get_clocks camera_pclk] -to [get_clocks clk_50m_in]
set_false_path -from [get_clocks clk_50m_in]  -to [get_clocks camera_pclk]

# AXI 系统域 (PS FCLK_CLK0 100MHz, 时序报告中为 clk_fpga_0)
# ↔ 摄像头像素域 (camera_pclk) 全部按异步处理。
# motion_regfile 的 reg_*→cfg_* 配置寄存器跨域、snap_* 状态寄存器跨域
# 均为慢变 CDC，软件侧用 SNAP_SEQ 协议保证一致性。
# 缺少本约束时 Vivado 按同步路径分析这些跨域路径，
# 产生 WNS≈-3.7ns / 248 条违例（M6 构建时序报告）。
set_clock_groups -asynchronous \
  -group [get_clocks clk_fpga_0] \
  -group [get_clocks camera_pclk]

# 注：PLL 输出时钟（200M/100M/50M/24M）由工具自动推导
# 注：DVI PLL 输出（disp_clk / disp_clk5x）由工具自动推导
# 注：PS FCLK_CLK0 由 PS7 IP 约束自动导出

# ==============================================================
# 3. IO 标准 - 系统
# ==============================================================

set_property IOSTANDARD LVCMOS33 [get_ports clk_50m_in]
set_property IOSTANDARD LVCMOS33 [get_ports reset_n]
set_property IOSTANDARD LVCMOS33 [get_ports led]

# ==============================================================
# 4. IO 标准 - 摄像头 DVP
# ==============================================================

set_property IOSTANDARD LVCMOS33 [get_ports camera_xclk]
set_property IOSTANDARD LVCMOS33 [get_ports camera_pclk]
set_property IOSTANDARD LVCMOS33 [get_ports camera_vsync]
set_property IOSTANDARD LVCMOS33 [get_ports camera_href]
set_property IOSTANDARD LVCMOS33 [get_ports camera_sclk]
set_property IOSTANDARD LVCMOS33 [get_ports camera_sdat]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[6]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[5]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[4]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[3]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[2]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[1]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_data[0]}]

# ==============================================================
# 5. IO 标准 - HDMI_2 TMDS
# ==============================================================

set_property IOSTANDARD TMDS_33 [get_ports hdmi_clk_p]
set_property IOSTANDARD TMDS_33 [get_ports hdmi_clk_n]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_tx2_p[2]}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_tx2_n[2]}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_tx2_p[1]}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_tx2_n[1]}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_tx2_p[0]}]
set_property IOSTANDARD TMDS_33 [get_ports {hdmi_tx2_n[0]}]

# ==============================================================
# 6. 引脚分配 - 系统
# ==============================================================

set_property PACKAGE_PIN Y9  [get_ports clk_50m_in]
set_property PACKAGE_PIN P17 [get_ports reset_n]
set_property PACKAGE_PIN N22 [get_ports led]
set_property PACKAGE_PIN M16 [get_ports recognition_sw0]
set_property IOSTANDARD LVCMOS33 [get_ports recognition_sw0]
# 用户拨码输入只对第一级同步器放宽异步输入路径。
set_false_path -from [get_ports recognition_sw0] -to [get_pins -hier -filter {NAME == "u_home_io/switch_sync_reg[0]/D"}]

# ==============================================================
# 7. 引脚分配 - 摄像头 DVP
# ==============================================================

set_property PACKAGE_PIN AA18 [get_ports camera_xclk]
set_property PACKAGE_PIN Y18  [get_ports camera_pclk]
set_property PACKAGE_PIN U21  [get_ports camera_vsync]
set_property PACKAGE_PIN T21  [get_ports camera_href]
set_property PACKAGE_PIN U22  [get_ports camera_sclk]
set_property PACKAGE_PIN T22  [get_ports camera_sdat]

set_property PACKAGE_PIN W21  [get_ports {camera_data[7]}]
set_property PACKAGE_PIN W20  [get_ports {camera_data[6]}]
set_property PACKAGE_PIN V19  [get_ports {camera_data[5]}]
set_property PACKAGE_PIN V18  [get_ports {camera_data[4]}]
set_property PACKAGE_PIN Y21  [get_ports {camera_data[3]}]
set_property PACKAGE_PIN Y20  [get_ports {camera_data[2]}]
set_property PACKAGE_PIN V17  [get_ports {camera_data[1]}]
set_property PACKAGE_PIN U17  [get_ports {camera_data[0]}]

# camera_xclk 驱动能力
set_property DRIVE 12 [get_ports camera_xclk]

# ==============================================================
# 8. 引脚分配 - HDMI_2 TMDS
#    _p 引脚手动约束，_n 由工具自动匹配差分对
# ==============================================================

set_property PACKAGE_PIN Y6   [get_ports hdmi_clk_p]
set_property PACKAGE_PIN AB7  [get_ports {hdmi_tx2_p[2]}]
set_property PACKAGE_PIN AB5  [get_ports {hdmi_tx2_p[1]}]
set_property PACKAGE_PIN Y4   [get_ports {hdmi_tx2_p[0]}]
