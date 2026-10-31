# HDMI 时钟：100 MHz 输入，74.25/371.25 MHz 输出。
# 参数按已有 Clocking Wizard 导出，需由 build.tcl 创建项目后调用。
set clk_wiz_dvi [create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz_dvi]

set_property -dict [list \
  CONFIG.CLKOUT1_DRIVES {BUFG} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {74.250} \
  CONFIG.CLKOUT1_USED {true} \
  CONFIG.CLKOUT2_DRIVES {BUFG} \
  CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {371.250} \
  CONFIG.CLKOUT2_USED {true} \
  CONFIG.NUM_OUT_CLKS {2} \
  CONFIG.PRIMITIVE {MMCM} \
  CONFIG.PRIM_IN_FREQ {100.000} \
  CONFIG.PRIM_SOURCE {Global_buffer} \
  CONFIG.RESET_PORT {resetn} \
  CONFIG.RESET_TYPE {ACTIVE_LOW} \
  CONFIG.USE_LOCKED {true} \
  CONFIG.USE_RESET {true} \
] [get_ips clk_wiz_dvi]

set_property -dict {
  GENERATE_SYNTH_CHECKPOINT {0}
} $clk_wiz_dvi
