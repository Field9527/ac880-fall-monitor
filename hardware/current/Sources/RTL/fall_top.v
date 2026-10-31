`timescale 1ns / 1ps
/////////////////////////////////////////////////////////////////////////////////
// Module Name   : fall_top
// Description   : AC880 M1 里程碑顶层模块
//                 OV5640 摄像头采集 → AXI VDMA → PS DDR3 帧缓冲 → HDMI 显示
//                 使用 AXI VDMA 替换自研 fifo_axi4_adapter 通路
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

module fall_top(
  // 系统时钟与复位
  input           clk_50m_in       ,  // 板载50MHz晶振，Y9
  input           reset_n          ,  // 复位按键，低有效，P17
  output          led              ,  // LED，N22
  output [7:1]    home_leds        ,  // remaining user LEDs
  inout           dht11_data       ,
  input           light_do          ,
  output          beep             ,  // passive buzzer, J22
  input           recognition_sw0  ,  // 用户拨码 SW0，M16

  // 摄像头 DVP 接口
  output          camera_xclk      ,  // 摄像头时钟输出，AA18
  input           camera_pclk      ,  // 摄像头像素时钟输入，Y18
  input           camera_vsync     ,  // 场同步，U21
  input           camera_href      ,  // 行有效，T21
  input  [7:0]    camera_data      ,  // 数据[7:0]
  output          camera_sclk      ,  // I2C时钟，U22
  inout           camera_sdat      ,  // I2C数据，T22

  // HDMI_2 TMDS 输出
  output          hdmi_clk_p       ,  // TMDS时钟+，Y6
  output          hdmi_clk_n       ,  // TMDS时钟-，Y5
  output [2:0]    hdmi_tx2_p       ,  // TMDS数据+ [2:0]，AB7/AB5/Y4
  output [2:0]    hdmi_tx2_n       ,  // TMDS数据- [2:0]，AB6/AB4/AA4

  // DDR3 接口 (PS侧)
  inout  [31:0]   ddr3_dq          ,
  inout  [3:0]    ddr3_dqs_n       ,
  inout  [3:0]    ddr3_dqs_p       ,
  output [14:0]   ddr3_addr        ,
  output [2:0]    ddr3_ba          ,
  output          ddr3_ras_n       ,
  output          ddr3_cas_n       ,
  output          ddr3_we_n        ,
  output          ddr3_reset_n     ,
  output [0:0]    ddr3_ck_p        ,
  output [0:0]    ddr3_ck_n        ,
  output [0:0]    ddr3_cke         ,
  output [0:0]    ddr3_cs_n        ,
  output [3:0]    ddr3_dm          ,
  output [0:0]    ddr3_odt         ,

  // PS FIXED_IO
  inout           FIXED_IO_ddr_vrn ,
  inout           FIXED_IO_ddr_vrp ,
  inout  [53:0]   FIXED_IO_mio     ,
  inout           FIXED_IO_ps_clk  ,
  inout           FIXED_IO_ps_porb ,
  inout           FIXED_IO_ps_srstb
);

  parameter IMAGE_WIDTH  = 1280;
  parameter IMAGE_HEIGHT = 720;

// =============================================
// 内部连线定义
// =============================================
  // PS 输出时钟与复位
  wire          fclk_clk0;          // PS FCLK_CLK0，100MHz
  wire          fclk_reset0_n;      // PS FCLK_RESET0_N，低有效

  // 系统 PLL (clk_wiz_sys) 输出
  wire          pll_locked;
  wire          pll_clk_200m;
  wire          pll_clk_100m;
  wire          pll_clk_50m;
  wire          pll_clk_24m;

  // DVI PLL (clk_wiz_dvi) 输出
  wire          dvi_pll_locked;
  wire          disp_clk;           // 74.25MHz
  wire          disp_clk5x;         // 371.25MHz

  // 复位信号（各时钟域）
  wire          sys_rst;            // 系统复位，高有效
  wire          cam_rst;            // cam_pclk 域复位，高有效
  wire          disp_rst;           // disp_clk 域复位，高有效
  wire          pll_100m_rst_n;     // 100MHz 域复位，低有效

  // 摄像头相关
  wire          camera_init_done;
  wire          cam_pclk_bufg;
  wire [15:0]   cam_pixel_data;
  wire          cam_pixel_valid;
  wire          cam_pixel_hs;
  wire          cam_pixel_vs;
  wire [11:0]   cam_xaddr;          // X 坐标
  wire [11:0]   cam_yaddr;          // Y 坐标

  // AXI-Stream: DVP → pixel_pack (16bit → 32bit)
  wire [15:0]   s_pack_tdata;
  wire          s_pack_tvalid;
  wire          s_pack_tuser;       // SOF
  wire          s_pack_tlast;       // EOL
  wire          s_pack_tready;

  // AXI-Stream: pixel_pack → axis_broadcast (32bit)
  wire [31:0]   m_pack_tdata;
  wire          m_pack_tvalid;
  wire          m_pack_tuser;
  wire          m_pack_tlast;
  wire          m_pack_tready;

  // AXI-Stream: axis_broadcast → VDMA S2MM (通道0)
  wire [31:0]   s2mm_axis_tdata;
  wire          s2mm_axis_tvalid;
  wire          s2mm_axis_tuser;
  wire          s2mm_axis_tlast;
  wire          s2mm_axis_tready;
  wire [3:0]    s2mm_axis_tkeep;  // 【v5】S2MM 字节使能, 恒 4'hF

  // AXI-Stream: axis_broadcast → motion_detect (通道1，M1悬空)
  wire [31:0]   mot_axis_tdata;
  wire          mot_axis_tvalid;
  wire          mot_axis_tuser;
  wire          mot_axis_tlast;
  wire          mot_axis_tready;

  // AXI-Stream: motion_detect ROI → CNN 加速器 (8bit, 96×96)

  // AXI-Stream: VDMA MM2S 输出 (16bit)
  wire [15:0]   mm2s_axis_tdata;
  wire          mm2s_axis_tvalid;
  wire          mm2s_axis_tuser;
  wire          mm2s_axis_tlast;
  wire          mm2s_axis_tready;

  // AXI-Lite: motion_detect 寄存器接口（100MHz 域）
  wire [11:0]   md_axi_awaddr;
  wire          md_axi_awvalid;
  wire          md_axi_awready;
  wire [31:0]   md_axi_wdata;
  wire [3:0]    md_axi_wstrb;
  wire          md_axi_wvalid;
  wire          md_axi_wready;
  wire [1:0]    md_axi_bresp;
  wire          md_axi_bvalid;
  wire          md_axi_bready;
  wire [11:0]   md_axi_araddr;
  wire          md_axi_arvalid;
  wire          md_axi_arready;
  wire [31:0]   md_axi_rdata;
  wire [1:0]    md_axi_rresp;
  wire          md_axi_rvalid;
  wire          md_axi_rready;

  // 显示驱动相关
  wire [23:0]   disp_data_rgb888;
  wire          disp_data_req;
  wire          disp_hs;
  wire          disp_vs;
  wire          disp_de;
  wire          disp_pclk;
  wire [7:0]    disp_red;
  wire [7:0]    disp_green;
  wire [7:0]    disp_blue;
  wire          frame_begin;

// =============================================
// 复位同步逻辑
// =============================================
  // 系统复位：PS复位 + 外部复位 + PLL锁定
  wire          ext_rst_n;
  assign ext_rst_n = reset_n & fclk_reset0_n;

  // 100MHz 域复位同步（20级移位寄存器）
  reg [19:0]    rst_sync_100m;
  wire          rst_100m_pre;
  assign rst_100m_pre = ~pll_locked | ~ext_rst_n;

  always @(posedge pll_clk_100m or posedge rst_100m_pre) begin
    if (rst_100m_pre)
      rst_sync_100m <= {20{1'b1}};
    else
      rst_sync_100m <= {rst_sync_100m[18:0], 1'b0};
  end

  assign pll_100m_rst_n = ~rst_sync_100m[19];
  assign sys_rst        =  rst_sync_100m[19];

  // fclk_clk0 域上电复位（不依赖 PS 的 fclk_reset0_n，
  // 因为 FCLK_RESET0_N 默认低电平且 PYNQ 不释放，会导致 AXI-Lite 挂死）
  reg [3:0]     rst_sync_fclk;
  always @(posedge fclk_clk0) begin
    if (rst_sync_fclk != 4'hF)
      rst_sync_fclk <= rst_sync_fclk + 1'b1;
  end
  wire axi_aresetn = (rst_sync_fclk == 4'hF);

  // cam_pclk 域复位同步
  reg [3:0]     rst_sync_cam;
  always @(posedge cam_pclk_bufg or negedge pll_100m_rst_n) begin
    if (!pll_100m_rst_n)
      rst_sync_cam <= 4'b1111;
    else
      rst_sync_cam <= {rst_sync_cam[2:0], 1'b0};
  end
  assign cam_rst = rst_sync_cam[3];

  // disp_clk 域复位同步
  reg [3:0]     rst_sync_disp;
  always @(posedge disp_clk or negedge dvi_pll_locked) begin
    if (!dvi_pll_locked)
      rst_sync_disp <= 4'b1111;
    else
      rst_sync_disp <= {rst_sync_disp[2:0], 1'b0};
  end
  assign disp_rst = rst_sync_disp[3];

// =============================================
// 系统 PLL: 50MHz 输入 → 200M/100M/50M/24M 输出
// =============================================
  clk_wiz_sys pll_sys (
    .clk_out1  (pll_clk_200m  ),  // 200MHz
    .clk_out2  (pll_clk_100m  ),  // 100MHz
    .clk_out3  (pll_clk_50m   ),  // 50MHz
    .clk_out4  (pll_clk_24m   ),  // 24MHz
    .resetn    (ext_rst_n     ),
    .locked    (pll_locked    ),
    .clk_in1   (clk_50m_in    )
  );

  assign camera_xclk = pll_clk_24m;

// =============================================
// DVI PLL: 100MHz 输入 → 74.25M / 371.25M 输出
// =============================================
  clk_wiz_dvi dvi_pll (
    .clk_out1  (disp_clk      ),  // 74.25MHz
    .clk_out2  (disp_clk5x    ),  // 371.25MHz
    .resetn    (pll_100m_rst_n),
    .locked    (dvi_pll_locked),
    .clk_in1   (pll_clk_100m  )
  );

// =============================================
// 摄像头 I2C 初始化
// =============================================
  camera_init #(
    .CAMERA_TYPE    ( "ov5640"     ),
    .IMAGE_TYPE     ( 0            ),  // 0: RGB; 1: JPEG
    .IMAGE_WIDTH    ( IMAGE_WIDTH  ),
    .IMAGE_HEIGHT   ( IMAGE_HEIGHT ),
    .IMAGE_FLIP_EN  ( 0            ),
    .IMAGE_MIRROR_EN( 0            )
  ) u_camera_init (
    .Clk         (pll_clk_50m      ),
    .Rst_n       (pll_100m_rst_n   ),
    .Init_Done   (camera_init_done ),
    .camera_rst_n(                 ),
    .camera_pwdn (                 ),
    .i2c_sclk    (camera_sclk      ),
    .i2c_sdat    (camera_sdat      )
  );

// =============================================
// 摄像头 DVP 采集
// =============================================
  BUFG BUFG_cam_pclk (
    .O (cam_pclk_bufg ),
    .I (camera_pclk   )
  );

  wire fifo_full;
  DVP_Capture u_dvp_capture (
    .Rst_p      (cam_rst          ),
    .PCLK       (cam_pclk_bufg    ),
    .Vsync      (camera_vsync     ),
    .Href       (camera_href      ),
    .Data       (camera_data      ),

    .ImageState (                 ),
    .DataValid  (cam_pixel_valid  ),
    .DataPixel  (cam_pixel_data   ),
    .DataHs     (cam_pixel_hs     ),
    .DataVs     (cam_pixel_vs     ),
    .Xaddr      (cam_xaddr        ),
    .Yaddr      (cam_yaddr        )
  );

  // DVP 输出 → AXI-Stream 从接口信号映射 (v3, 修复 EOL 误触发)
  //
  // 【v1 bug】eol_pulse <= cam_pixel_valid & ~camera_href 是寄存器输出,
  //   比最后一个有效像素晚一拍, 该拍 tvalid=0, tlast 无法搭载到有效数据上。
  // 【v2 bug】改用"延迟拍有效 & 原始流当前拍无效", 但仍然错误:
  //   DVP_Capture 的 DataValid 是"每像素一个单拍脉冲"(行内每 2 个 PCLK
  //   出 1 个), 不是连续电平。因此 valid 的下降沿每行出现 640 次,
  //   tlast 被误发 640 次/行。VDMA 数到第 720 个 tlast 仍未见下一个
  //   tuser → SOFLate(SR bit11) + ErrInternal, 帧永不完成,
  //   S2MM FrameCnt 停在 1, DDR 三帧缓冲全零。
  //
  // 【v3 方案】EOL 改用列号判据。DVP_Capture 的 Xaddr 是行内像素计数器,
  //   与 DataValid 脉冲一一对应, 有效拍上取值 1..IMAGE_WIDTH。
  //   当列号达到 IMAGE_WIDTH 即为本行最后一个像素 → 每行恰好 1 个 tlast。
  //   SOF 沿用挂起标志方案 (DataVs 上升沿挂起, 随帧内第一个有效像素发出)。
  reg  [15:0] dvp_d_pixel;
  reg         dvp_d_valid;
  reg  [11:0] dvp_d_xaddr;
  reg         dvp_sof_pending;
  reg         r_cam_pixel_vs;

  always @(posedge cam_pclk_bufg or posedge cam_rst) begin
    if (cam_rst) begin
      dvp_d_pixel     <= 16'd0;
      dvp_d_valid     <= 1'b0;
      dvp_d_xaddr     <= 12'd0;
      dvp_sof_pending <= 1'b0;
      r_cam_pixel_vs  <= 1'b0;
    end else begin
      dvp_d_pixel <= cam_pixel_data;
      dvp_d_valid <= cam_pixel_valid;
      dvp_d_xaddr <= cam_xaddr;
      r_cam_pixel_vs <= cam_pixel_vs;
      // DataVs 上升沿 = 消隐→有效 = 新帧开始, 挂起 SOF
      if (cam_pixel_vs & ~r_cam_pixel_vs)
        dvp_sof_pending <= 1'b1;
      else if (dvp_d_valid & dvp_sof_pending)
        dvp_sof_pending <= 1'b0;   // SOF 已随第一个有效像素发出
    end
  end

  // EOL: 本行最后一个像素 —— 列号达到每行像素数 (每行恰好 1 次)
  wire dvp_eol_now = dvp_d_valid & (dvp_d_xaddr == IMAGE_WIDTH[11:0]);
  // SOF: 帧起始后第一个有效像素
  wire dvp_sof_now = dvp_d_valid & dvp_sof_pending;

  assign s_pack_tdata  = dvp_d_pixel;
  assign s_pack_tvalid = dvp_d_valid;
  assign s_pack_tuser  = dvp_sof_now;
  assign s_pack_tlast  = dvp_eol_now;

  // -------------------------------------------------------------
  // 采集链路调试计数器 (cam_pclk 域)
  //   用于确认摄像头真实输出的行像素数 / 每帧行数 / 打包拍数,
  //   经 motion_regfile 只读寄存器 0x1B~0x1F 暴露给 PS
  // -------------------------------------------------------------
  reg [15:0] dbg_max_xaddr;    // 观测到的最大列号 = 每行像素数
  reg [15:0] dbg_line_cnt;     // 当前帧内 EOL(tlast) 数
  reg [15:0] dbg_lines_last;   // 上一帧的 EOL 数 = 每帧行数
  reg [15:0] dbg_sof_cnt;      // SOF 累计数
  reg [15:0] dbg_pix_lost_cur; // 当前帧内因反压丢失的像素数
  reg [15:0] dbg_pix_lost;     // 上一帧丢失的像素数 (关键指标, 应为 0)
  reg [15:0] dbg_fifo_ovf;     // FIFO 满(溢出)累计数 (关键指标, 应为 0)
  reg [15:0] dbg_axis_frames;  // AXI-Stream 侧帧计数

  always @(posedge cam_pclk_bufg or posedge cam_rst) begin
    if (cam_rst) begin
      dbg_max_xaddr    <= 16'd0;
      dbg_line_cnt     <= 16'd0;
      dbg_lines_last   <= 16'd0;
      dbg_sof_cnt      <= 16'd0;
      dbg_pix_lost_cur <= 16'd0;
      dbg_pix_lost     <= 16'd0;
      dbg_fifo_ovf     <= 16'd0;
      dbg_axis_frames  <= 16'd0;
    end else begin
      if (dvp_d_valid && ({4'd0, dvp_d_xaddr} > dbg_max_xaddr))
        dbg_max_xaddr <= {4'd0, dvp_d_xaddr};
      if (dvp_eol_now)
        dbg_line_cnt <= dbg_line_cnt + 16'd1;
      // DVP 侧无反压机制: 打包器 tready=0 时该像素被丢弃
      if (dvp_d_valid & ~s_pack_tready)
        dbg_pix_lost_cur <= dbg_pix_lost_cur + 16'd1;
      // FIFO 满: 打包输出无法写入, 数据丢失
      if (m_pack_tvalid & fifo_full)
        dbg_fifo_ovf <= dbg_fifo_ovf + 16'd1;
      if (dvp_sof_now) begin
        dbg_sof_cnt      <= dbg_sof_cnt + 16'd1;
        dbg_lines_last   <= dbg_line_cnt;
        dbg_pix_lost     <= dbg_pix_lost_cur;
        dbg_line_cnt     <= 16'd0;
        dbg_pix_lost_cur <= 16'd0;
        dbg_axis_frames  <= dbg_axis_frames + 16'd1;
      end
    end
  end

  // -------------------------------------------------------------
  // 【v4 新增】VDMA S2MM 引脚侧的在线计数器 (cam_pclk 域)
  //   前面 6 个计数器测的是"打包器之前"的 DVP 流, 已经确认干净
  //   (1280 像素/行, 720 行/帧, 丢数=0)。但 VDMA 仍报 SOFLate/EOLLate +
  //   ErrInternal, 说明问题出在"VDMA 实际收到什么"这一环。
  //   这一组计数器直接挂在 s2mm_axis_* 引脚上, 用 tvalid&tready 握手计数,
  //   是判定问题的决定性测量:
  //     beats/行 期望 640 (=2560B/行 / 4B)
  //     lines/帧 期望 720
  //   经 motion_regfile 只读寄存器 0x84~0xA0 暴露给 PS
  // -------------------------------------------------------------
  wire s2mm_hs = s2mm_axis_tvalid & s2mm_axis_tready;

  reg [15:0] vdma_beats_cur;    // 当前行已送出的 32bit 拍数
  reg [15:0] vdma_beats_last;   // 上一行拍数 (期望 640)
  reg [15:0] vdma_max_beats;    // 观测到的最大拍数/行
  reg [15:0] vdma_lines_cur;    // 当前帧已送出的行数
  reg [15:0] vdma_lines_last;   // 上一帧行数 (期望 720)
  reg [15:0] vdma_sof_cnt;      // 送到 VDMA 的 SOF(tuser) 累计数
  reg [15:0] vdma_frames;       // VDMA 侧帧计数
  reg [15:0] vdma_rdy_low;      // tvalid=1 但 tready=0 的周期数(反压强度)

  always @(posedge cam_pclk_bufg or posedge cam_rst) begin
    if (cam_rst) begin
      vdma_beats_cur  <= 16'd0;
      vdma_beats_last <= 16'd0;
      vdma_max_beats  <= 16'd0;
      vdma_lines_cur  <= 16'd0;
      vdma_lines_last <= 16'd0;
      vdma_sof_cnt    <= 16'd0;
      vdma_frames     <= 16'd0;
      vdma_rdy_low    <= 16'd0;
    end else begin
      // 反压强度统计
      if (s2mm_axis_tvalid & ~s2mm_axis_tready)
        vdma_rdy_low <= vdma_rdy_low + 16'd1;

      if (s2mm_hs) begin
        vdma_beats_cur <= vdma_beats_cur + 16'd1;

        if (s2mm_axis_tlast) begin
          // 本行结束: 锁存行长并复位拍计数
          if ((vdma_beats_cur + 16'd1) > vdma_max_beats)
            vdma_max_beats <= vdma_beats_cur + 16'd1;
          vdma_beats_last <= vdma_beats_cur + 16'd1;
          vdma_beats_cur  <= 16'd0;
          vdma_lines_cur  <= vdma_lines_cur + 16'd1;
        end

        if (s2mm_axis_tuser) begin
          // 新帧开始: 锁存上一帧行数
          vdma_sof_cnt    <= vdma_sof_cnt + 16'd1;
          vdma_lines_last <= vdma_lines_cur;
          vdma_lines_cur  <= 16'd0;
          vdma_frames     <= vdma_frames + 16'd1;
        end
      end
    end
  end

// =============================================
// 像素打包：16bit → 32bit (2像素/拍)
// =============================================
  pixel_pack_16to32 u_pixel_pack (
    .aclk          (cam_pclk_bufg ),
    .aresetn       (~cam_rst      ),

    .s_axis_tdata  (s_pack_tdata  ),
    .s_axis_tvalid (s_pack_tvalid ),
    .s_axis_tuser  (s_pack_tuser  ),
    .s_axis_tlast  (s_pack_tlast  ),
    .s_axis_tready (s_pack_tready ),

    .m_axis_tdata  (m_pack_tdata  ),
    .m_axis_tvalid (m_pack_tvalid ),
    .m_axis_tuser  (m_pack_tuser  ),
    .m_axis_tlast  (m_pack_tlast  ),
    .m_axis_tready (m_pack_tready )
  );

// =============================================
// 打包输出 → 同步 FIFO → VDMA S2MM
// =============================================
// 【v3 修复】原先的 axis_broadcast(1进2出) 被整体移除:
//   1) motion_detect 走的是并行像素接口 cam_data/xaddr/yaddr,
//      广播出去的 m_axis_1 是死路径, 广播本身毫无必要;
//   2) 该广播模块存在真实丢数 bug —— both_ready 分支里清除 buf_valid 时
//      没有接收当前拍输入, 而 s_axis_tready 却置 1, 上游误以为数据已被
//      接收, 数据凭空丢失。
//
// 改为: 打包器 → 同步 FIFO(1024×34bit) → VDMA S2MM
//   FIFO 用于吸收 VDMA 写 DDR 时的周期性反压。摄像头 DVP 侧没有反压
//   机制(像素流不会等待), 若无弹性缓冲, 反压期间像素必然被丢弃,
//   行长度不一致 → VDMA 报 SOFLate/EOLLate + ErrInternal。
//
// FIFO 数据位格式: [33]=tuser(SOF) [32]=tlast(EOL) [31:0]=tdata
  wire [33:0] fifo_din  = {m_pack_tuser, m_pack_tlast, m_pack_tdata};
  wire [33:0] fifo_dout;
  wire        fifo_valid;

  wire fifo_wr_en = m_pack_tvalid & ~fifo_full;
  wire fifo_rd_en = fifo_valid & s2mm_axis_tready;

  assign m_pack_tready = ~fifo_full;

  axis_fifo_sync #(
    .DATA_WIDTH    (34            ),
    // 2048 entries ≈ 8KB。原为 1024; 行缓冲修复后理论上不再溢出,
    // 加深一倍作为余量 (BRAM 成本仅多 1 个 RAMB36)
    .ADDR_WIDTH    (11            )
  ) u_axis_fifo (
    .clk           (cam_pclk_bufg ),
    .rst           (cam_rst       ),

    .din           (fifo_din      ),
    .wr_en         (fifo_wr_en    ),
    .full          (fifo_full     ),

    .dout          (fifo_dout     ),
    .valid         (fifo_valid    ),
    .rd_en         (fifo_rd_en    )
  );

  assign s2mm_axis_tdata  = fifo_dout[31:0];
  assign s2mm_axis_tuser  = fifo_dout[33];
  assign s2mm_axis_tlast  = fifo_dout[32];
  assign s2mm_axis_tvalid = fifo_valid;
  // 【v5 修复】tkeep 之前完全悬空 (Synth 8-7071 告警), 综合后被绑到常量。
  // AXI VDMA 的 S2MM 数据通路用 tkeep 判定每拍有效字节数, tkeep 不对会
  // 直接导致行字节计数错误 -> 行/帧判定异常 -> 帧指针不推进。
  // 32bit 全部有效, 恒为 4'hF。
  assign s2mm_axis_tkeep  = 4'hF;

// =============================================
// VDMA MM2S → RGB565 转 RGB888 → 显示驱动
// =============================================
  // VDMA MM2S 的 tready 由显示驱动的数据请求控制（Disp_DE 期间请求数据）
  wire [11:0] disp_haddr, disp_vaddr;
  wire [15:0] aligned_pixel;
  wire aligned_valid;
  axis_video_sync u_axis_video_sync (
    .clk(disp_clk), .rst(disp_rst),
    .s_data(mm2s_axis_tdata), .s_valid(mm2s_axis_tvalid),
    .s_user(mm2s_axis_tuser), .s_last(mm2s_axis_tlast),
    .s_ready(mm2s_axis_tready), .request(disp_data_req),
    .first_pixel(disp_data_req && disp_haddr == 12'd1 && disp_vaddr == 12'd0),
    .last_pixel(disp_data_req && disp_haddr == IMAGE_WIDTH),
    .pixel(aligned_pixel), .pixel_valid(aligned_valid)
  );

  wire [23:0] mm2s_rgb888_data;
  wire        mm2s_rgb888_valid;

  rgb565_to_rgb888 u_rgb565_to_rgb888 (
    .clk           (disp_clk        ),
    .rst_n         (~disp_rst       ),
    .pixel_data    (aligned_pixel ),
    .pixel_valid   (aligned_valid),
    .rgb888_data   (mm2s_rgb888_data),
    .rgb888_valid  (mm2s_rgb888_valid)
  );

  assign disp_data_rgb888 = mm2s_rgb888_data;

// =============================================
// 显示驱动
// =============================================
  disp_driver u_disp_driver (
    .ClkDisp     (disp_clk        ),
    .Rst_p       (disp_rst        ),
    .Data        (disp_data_rgb888),
    .DataReq     (disp_data_req   ),
    .H_Addr      (disp_haddr      ),
    .V_Addr      (disp_vaddr      ),
    .Disp_HS     (disp_hs         ),
    .Disp_VS     (disp_vs         ),
    .Disp_Red    (disp_red        ),
    .Disp_Green  (disp_green      ),
    .Disp_Blue   (disp_blue       ),
    .Frame_Begin (frame_begin     ),
    .Disp_DE     (disp_de         ),
    .Disp_PCLK   (disp_pclk       )
  );

// =============================================
// DVI / TMDS 编码器
// =============================================
  dvi_encoder u_dvi_encoder (
    .pixelclk    (disp_clk     ),
    .pixelclk5x  (disp_clk5x   ),
    .rst_p       (disp_rst     ),
    .blue_din    (disp_blue    ),
    .green_din   (disp_green   ),
    .red_din     (disp_red     ),
    .hsync       (disp_hs      ),
    .vsync       (disp_vs      ),
    .de          (disp_de      ),
    .tmds_clk_p  (hdmi_clk_p   ),
    .tmds_clk_n  (hdmi_clk_n   ),
    .tmds_data_p (hdmi_tx2_p   ),
    .tmds_data_n (hdmi_tx2_n   )
  );

// =============================================
// LED 指示：摄像头初始化完成
// =============================================
  wire [31:0] env_status;
  wire [31:0] env_payload;
  wire [31:0] env_sample_seq;
  wire [31:0] env_age_ms;
  wire [31:0] env_errors;
  wire [31:0] env_attempts;
  environment_sensors u_environment (
    .clk(fclk_clk0), .resetn(fclk_reset0_n),
    .dht11_data(dht11_data), .light_do(light_do),
    .status(env_status),
    .payload(env_payload),
    .sample_seq(env_sample_seq),
    .age_ms(env_age_ms),
    .errors(env_errors),
    .attempts(env_attempts)
  );
  wire [31:0] image_control, image_frames;
  wire [63:0] image_tx_tdata,image_rx_tdata;
  wire [7:0] image_tx_tkeep,image_rx_tkeep;
  wire image_tx_tvalid,image_tx_tready,image_tx_tlast;
  wire image_rx_tvalid,image_rx_tready,image_rx_tlast,image_resetn;
  axis_image_filter64 u_image_filter (
    .clk(fclk_clk0),.resetn(fclk_reset0_n & image_resetn),.control(image_control),
    .s_data(image_tx_tdata),.s_keep(image_tx_tkeep),.s_valid(image_tx_tvalid),
    .s_ready(image_tx_tready),.s_last(image_tx_tlast),
    .m_data(image_rx_tdata),.m_keep(image_rx_tkeep),.m_valid(image_rx_tvalid),
    .m_ready(image_rx_tready),.m_last(image_rx_tlast),.frame_count(image_frames)
  );
  wire [9:0] home_ctrl;
  wire home_feed;
  wire [31:0] home_status;
  wire recognition_mode_sw;
  wire [7:0] home_led_value;
  home_io u_home_io (
    .clk(fclk_clk0), .resetn(axi_aresetn),
    .control(home_ctrl), .feed(home_feed), .camera_ready(camera_init_done),
    .switch0(recognition_sw0), .mode_switch(recognition_mode_sw),
    .leds(home_led_value), .beep(beep), .status(home_status)
  );
  assign led = home_led_value[0];
  assign home_leds = home_led_value[7:1];

// =============================================
// PS Block Design 实例化 (system_wrapper)
// =============================================
  system_wrapper u_system_wrapper (
    .image_resetn(image_resetn),
    .image_tx_tdata(image_tx_tdata),
    .image_tx_tkeep(image_tx_tkeep),
    .image_tx_tvalid(image_tx_tvalid),
    .image_tx_tready(image_tx_tready),
    .image_tx_tlast(image_tx_tlast),
    .image_rx_tdata(image_rx_tdata),
    .image_rx_tkeep(image_rx_tkeep),
    .image_rx_tvalid(image_rx_tvalid),
    .image_rx_tready(image_rx_tready),
    .image_rx_tlast(image_rx_tlast),

    .DDR_addr            (ddr3_addr           ),
    .DDR_ba              (ddr3_ba             ),
    .DDR_cas_n           (ddr3_cas_n          ),
    .DDR_ck_n            (ddr3_ck_n           ),
    .DDR_ck_p            (ddr3_ck_p           ),
    .DDR_cke             (ddr3_cke            ),
    .DDR_cs_n            (ddr3_cs_n           ),
    .DDR_dm              (ddr3_dm             ),
    .DDR_dq              (ddr3_dq             ),
    .DDR_dqs_n           (ddr3_dqs_n          ),
    .DDR_dqs_p           (ddr3_dqs_p          ),
    .DDR_odt             (ddr3_odt            ),
    .DDR_ras_n           (ddr3_ras_n          ),
    .DDR_reset_n         (ddr3_reset_n        ),
    .DDR_we_n            (ddr3_we_n           ),
    .FIXED_IO_ddr_vrn    (FIXED_IO_ddr_vrn    ),
    .FIXED_IO_ddr_vrp    (FIXED_IO_ddr_vrp    ),
    .FIXED_IO_mio        (FIXED_IO_mio        ),
    .FIXED_IO_ps_clk     (FIXED_IO_ps_clk     ),
    .FIXED_IO_ps_porb    (FIXED_IO_ps_porb    ),
    .FIXED_IO_ps_srstb   (FIXED_IO_ps_srstb   ),

    // PS → PL 时钟与复位
    .fclk_clk0           (fclk_clk0           ),
    .fclk_reset0_n       (fclk_reset0_n       ),

    // VDMA S2MM AXI-Stream (PL → PS)
    .s_axis_s2mm_tdata   (s2mm_axis_tdata     ),
    .s_axis_s2mm_tvalid  (s2mm_axis_tvalid    ),
    .s_axis_s2mm_tuser   (s2mm_axis_tuser     ),
    .s_axis_s2mm_tlast   (s2mm_axis_tlast     ),
    .s_axis_s2mm_tkeep   (s2mm_axis_tkeep     ),  // 【v5】之前漏接导致悬空
    .s_axis_s2mm_tready  (s2mm_axis_tready    ),
    .s_axis_s2mm_aclk    (cam_pclk_bufg       ),

    // VDMA MM2S AXI-Stream (PS → PL)
    .m_axis_mm2s_tdata   (mm2s_axis_tdata     ),
    .m_axis_mm2s_tvalid  (mm2s_axis_tvalid    ),
    .m_axis_mm2s_tuser   (mm2s_axis_tuser     ),
    .m_axis_mm2s_tlast   (mm2s_axis_tlast     ),
    .m_axis_mm2s_tready  (mm2s_axis_tready    ),
    .m_axis_mm2s_aclk    (disp_clk            ),

    // motion_detect 用的 AXI-Lite 主接口（SmartConnect M01_AXI 引出）
    .m_axi_lite_md_awaddr  (md_axi_awaddr     ),
    .m_axi_lite_md_awvalid (md_axi_awvalid    ),
    .m_axi_lite_md_awready (md_axi_awready    ),
    .m_axi_lite_md_wdata   (md_axi_wdata      ),
    .m_axi_lite_md_wstrb   (md_axi_wstrb      ),
    .m_axi_lite_md_wvalid  (md_axi_wvalid     ),
    .m_axi_lite_md_wready  (md_axi_wready     ),
    .m_axi_lite_md_bresp   (md_axi_bresp      ),
    .m_axi_lite_md_bvalid  (md_axi_bvalid     ),
    .m_axi_lite_md_bready  (md_axi_bready     ),
    .m_axi_lite_md_araddr  (md_axi_araddr     ),
    .m_axi_lite_md_arvalid (md_axi_arvalid    ),
    .m_axi_lite_md_arready (md_axi_arready    ),
    .m_axi_lite_md_rdata   (md_axi_rdata      ),
    .m_axi_lite_md_rresp   (md_axi_rresp      ),
    .m_axi_lite_md_rvalid  (md_axi_rvalid     ),
    .m_axi_lite_md_rready  (md_axi_rready     )

  );

// =============================================
// motion_detect 一级运动检测引擎
// =============================================
  motion_detect_top u_motion_detect (
    // AXI-Lite 时钟/复位（用 PS 的 fclk_clk0，与 SmartConnect 同域）
    .s_axi_aclk       (fclk_clk0          ),
    .s_axi_aresetn    (axi_aresetn        ),
    .home_ctrl        (home_ctrl),
    .home_feed        (home_feed),
    .home_status      (home_status),
    .recognition_mode_sw (recognition_mode_sw),
    .image_control(image_control),.image_frames(image_frames),
    .env_status (env_status),
    .env_payload (env_payload),
    .env_sample_seq (env_sample_seq),
    .env_age_ms (env_age_ms),
    .env_errors (env_errors),
    .env_attempts (env_attempts),


    // AXI-Lite 从接口（接 BD 引出的 AXI-Lite 主）
    .s_axi_awaddr     (md_axi_awaddr      ),
    .s_axi_awvalid    (md_axi_awvalid     ),
    .s_axi_awready    (md_axi_awready     ),
    .s_axi_wdata      (md_axi_wdata       ),
    .s_axi_wstrb      (md_axi_wstrb       ),
    .s_axi_wvalid     (md_axi_wvalid      ),
    .s_axi_wready     (md_axi_wready      ),
    .s_axi_bresp      (md_axi_bresp       ),
    .s_axi_bvalid     (md_axi_bvalid      ),
    .s_axi_bready     (md_axi_bready      ),
    .s_axi_araddr     (md_axi_araddr      ),
    .s_axi_arvalid    (md_axi_arvalid     ),
    .s_axi_arready    (md_axi_arready     ),
    .s_axi_rdata      (md_axi_rdata       ),
    .s_axi_rresp      (md_axi_rresp       ),
    .s_axi_rvalid     (md_axi_rvalid      ),
    .s_axi_rready     (md_axi_rready      ),

    // 像素时钟/复位（pclk 域）
    .aclk_cam         (cam_pclk_bufg      ),
    .aresetn_cam      (~cam_rst           ),

    // 相机输入
    .cam_data         (cam_pixel_data     ),
    .cam_valid        (cam_pixel_valid    ),
    .cam_xaddr        (cam_xaddr          ),
    .cam_yaddr        (cam_yaddr          ),
    .cam_vs           (cam_pixel_vs       ),

    // ROI 输出 → CNN 加速器 AXI-Stream
    .m_axis_roi_tdata (),
    .m_axis_roi_tvalid(),
    .m_axis_roi_tuser (),
    .m_axis_roi_tlast (),
    .m_axis_roi_tready(1'b1             ),

    // 采集链路调试计数器 (cam_pclk 域, 只读)
    .dbg_max_xaddr    (dbg_max_xaddr     ),
    .dbg_lines_last   (dbg_lines_last    ),
    .dbg_sof_cnt      (dbg_sof_cnt       ),
    .dbg_pix_lost     (dbg_pix_lost      ),
    .dbg_fifo_ovf     (dbg_fifo_ovf      ),
    .dbg_axis_frames  (dbg_axis_frames   ),

    // VDMA S2MM 引脚侧调试计数器 (cam_pclk 域, 只读)
    .dbg_vdma_beats   (vdma_beats_last   ),
    .dbg_vdma_maxbeat (vdma_max_beats    ),
    .dbg_vdma_lines   (vdma_lines_last   ),
    .dbg_vdma_sof     (vdma_sof_cnt      ),
    .dbg_vdma_frames  (vdma_frames       ),
    .dbg_vdma_rdylow  (vdma_rdy_low      ),
    .dbg_vdma_beatcur (vdma_beats_cur    ),
    .dbg_vdma_linecur (vdma_lines_cur    )
  );

endmodule
