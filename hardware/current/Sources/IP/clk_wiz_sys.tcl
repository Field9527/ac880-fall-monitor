# 系统时钟：50 MHz 输入，200/100/50/24 MHz 输出。
# 参数按已有 Clocking Wizard 导出，需由 build.tcl 创建项目后调用。
set clk_wiz_sys [create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz_sys]

set_property -dict [list \
  CONFIG.CLKOUT1_DRIVES {BUFG} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {200.000} \
  CONFIG.CLKOUT1_USED {true} \
  CONFIG.CLKOUT2_DRIVES {BUFG} \
  CONFIG.CLKOUT2_REQUESTED_OUT_FREQ {100.000} \
  CONFIG.CLKOUT2_USED {true} \
  CONFIG.CLKOUT3_DRIVES {BUFG} \
  CONFIG.CLKOUT3_REQUESTED_OUT_FREQ {50.000} \
  CONFIG.CLKOUT3_USED {true} \
  CONFIG.CLKOUT4_DRIVES {BUFG} \
  CONFIG.CLKOUT4_REQUESTED_OUT_FREQ {24.000} \
  CONFIG.CLKOUT4_USED {true} \
  CONFIG.NUM_OUT_CLKS {4} \
  CONFIG.PRIMITIVE {MMCM} \
  CONFIG.PRIM_IN_FREQ {50.000} \
  CONFIG.PRIM_SOURCE {Single_ended_clock_capable_pin} \
  CONFIG.RESET_PORT {resetn} \
  CONFIG.RESET_TYPE {ACTIVE_LOW} \
  CONFIG.USE_LOCKED {true} \
  CONFIG.USE_RESET {true} \
] [get_ips clk_wiz_sys]

set_property -dict {
  GENERATE_SYNTH_CHECKPOINT {0}
} $clk_wiz_sys
