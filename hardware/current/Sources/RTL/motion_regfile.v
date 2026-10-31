/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_regfile
// Description   : 运动检测 AXI-Lite 寄存器文件
//                 - 实现 0x00~0x68 共 27 个寄存器
//                 - AXI-Lite 从接口（32位，4K地址空间）
//                 - 跨时钟域：pclk 域快照 → AXI 域，快照+序号协议
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_regfile(
    // AXI-Lite 时钟/复位
    s_axi_aclk,
    s_axi_aresetn,
    home_ctrl,
    home_feed,
    home_status,
    recognition_mode_sw,
    image_control,image_frames,
    env_status,
    env_payload,
    env_sample_seq,
    env_age_ms,
    env_errors,
    env_attempts,


    // AXI-Lite 写地址通道
    s_axi_awaddr,
    s_axi_awvalid,
    s_axi_awready,

    // AXI-Lite 写数据通道
    s_axi_wdata,
    s_axi_wstrb,
    s_axi_wvalid,
    s_axi_wready,

    // AXI-Lite 写响应通道
    s_axi_bresp,
    s_axi_bvalid,
    s_axi_bready,

    // AXI-Lite 读地址通道
    s_axi_araddr,
    s_axi_arvalid,
    s_axi_arready,

    // AXI-Lite 读数据通道
    s_axi_rdata,
    s_axi_rresp,
    s_axi_rvalid,
    s_axi_rready,

    // pclk 域时钟/复位
    pclk,
    prst,

    // ===== pclk 域配置输出（AXI → pclk）=====
    cfg_en,
    cfg_sw_rst,
    cfg_clear_bg,
    cfg_roi_en,
    cfg_force_trig,
    cfg_thresh,
    cfg_min_area,
    cfg_roi_trig,
    cfg_ds_factor,
    cfg_morph_cfg,
    cfg_trig_n,
    cfg_mode,
    cfg_bg_alpha,
    cfg_roi_pad,
    cfg_pix_w,
    cfg_pix_h,
    cfg_row_th,
    cfg_col_th,

    // ===== pclk 域状态输入（pclk → AXI）=====
    sts_busy,
    sts_bg_valid,
    sts_roi_active,
    sts_roi_done,
    sts_motion,
    sts_bbox_upd,
    sts_bbox_x,
    sts_bbox_y,
    sts_bbox_w,
    sts_bbox_h,
    sts_area,
    sts_energy,
    sts_frame_cnt,
    sts_proc_lat,
    sts_roi_sent,
    sts_snap_seq,
    sts_frame_ms,

    // ===== 采集链路调试计数器（cam_pclk 域, 只读）=====
    sts_dbg_max_xaddr,
    sts_dbg_lines_last,
    sts_dbg_sof_cnt,
    sts_dbg_pix_lost,
    sts_dbg_fifo_ovf,
    sts_dbg_axis_frames,

    // ===== VDMA S2MM 引脚侧调试计数器（cam_pclk 域, 只读）=====
    sts_dbg_vdma_beats,
    sts_dbg_vdma_maxbeat,
    sts_dbg_vdma_lines,
    sts_dbg_vdma_sof,
    sts_dbg_vdma_frames,
    sts_dbg_vdma_rdylow,
    sts_dbg_vdma_beatcur,
    sts_dbg_vdma_linecur
);

// ============================================================
// 参数
// ============================================================
parameter ADDR_WIDTH = 12;              // 4K 地址空间
parameter DATA_WIDTH = 32;
parameter STRB_WIDTH = 4;

// ============================================================
// 端口声明
// ============================================================
input                       s_axi_aclk;
input                       s_axi_aresetn;
output reg [9:0]             home_ctrl;
output reg                   home_feed;
input [31:0]                 home_status;
input recognition_mode_sw;
output reg [31:0] image_control;
input [31:0] image_frames;
input [31:0] env_status;
input [31:0] env_payload;
input [31:0] env_sample_seq;
input [31:0] env_age_ms;
input [31:0] env_errors;
input [31:0] env_attempts;


input   [ADDR_WIDTH-1:0]    s_axi_awaddr;
input                       s_axi_awvalid;
output reg                  s_axi_awready;

input   [DATA_WIDTH-1:0]    s_axi_wdata;
input   [STRB_WIDTH-1:0]    s_axi_wstrb;
input                       s_axi_wvalid;
output reg                  s_axi_wready;

output  [1:0]               s_axi_bresp;
output reg                  s_axi_bvalid;
input                       s_axi_bready;

input   [ADDR_WIDTH-1:0]    s_axi_araddr;
input                       s_axi_arvalid;
output reg                  s_axi_arready;

output  [DATA_WIDTH-1:0]    s_axi_rdata;
output  [1:0]               s_axi_rresp;
output reg                  s_axi_rvalid;
input                       s_axi_rready;

input                       pclk;
input                       prst;

// 配置输出（pclk 域）
output reg                  cfg_en;
output reg                  cfg_sw_rst;
output reg                  cfg_clear_bg;
output reg                  cfg_roi_en;
output reg                  cfg_force_trig;
output reg [7:0]            cfg_thresh;
output reg [19:0]           cfg_min_area;
output reg                  cfg_roi_trig;
output reg [3:0]            cfg_ds_factor;
output reg [1:0]            cfg_morph_cfg;
output reg [3:0]            cfg_trig_n;
output reg                  cfg_mode;
output reg [3:0]            cfg_bg_alpha;
output reg [7:0]            cfg_roi_pad;
output reg [11:0]           cfg_pix_w;
output reg [11:0]           cfg_pix_h;
output reg [7:0]            cfg_row_th;
output reg [7:0]            cfg_col_th;

// 状态输入（pclk 域）
input                       sts_busy;
input                       sts_bg_valid;
input                       sts_roi_active;
input                       sts_roi_done;
input                       sts_motion;
input                       sts_bbox_upd;
input   [11:0]              sts_bbox_x;
input   [11:0]              sts_bbox_y;
input   [11:0]              sts_bbox_w;
input   [11:0]              sts_bbox_h;
input   [19:0]              sts_area;
input   [23:0]              sts_energy;
input   [15:0]              sts_frame_cnt;
input   [15:0]              sts_proc_lat;
input   [15:0]              sts_roi_sent;
input   [7:0]               sts_snap_seq;
input   [15:0]              sts_frame_ms;

// 采集链路调试计数器（cam_pclk 域）
input   [15:0]              sts_dbg_max_xaddr;
input   [15:0]              sts_dbg_lines_last;
input   [15:0]              sts_dbg_sof_cnt;
input   [15:0]              sts_dbg_pix_lost;
input   [15:0]              sts_dbg_fifo_ovf;
input   [15:0]              sts_dbg_axis_frames;

// VDMA S2MM 引脚侧调试计数器 (cam_pclk 域)
input   [15:0]              sts_dbg_vdma_beats;     // 0x84 上一行拍数 (期望640)
input   [15:0]              sts_dbg_vdma_maxbeat;   // 0x88 最大拍数/行
input   [15:0]              sts_dbg_vdma_lines;     // 0x8C 上一帧行数 (期望720)
input   [15:0]              sts_dbg_vdma_sof;       // 0x90 SOF 累计数
input   [15:0]              sts_dbg_vdma_frames;    // 0x94 VDMA 侧帧计数
input   [15:0]              sts_dbg_vdma_rdylow;    // 0x98 反压周期数
input   [15:0]              sts_dbg_vdma_beatcur;   // 0x9C 当前行拍数
input   [15:0]              sts_dbg_vdma_linecur;   // 0xA0 当前帧行数

// ============================================================
// 内部信号（AXI-Lite 域）
// ============================================================
// 写地址 FIFO
reg [ADDR_WIDTH-1:0]        aw_addr_reg;
reg                         aw_valid_reg;

// 写数据 FIFO
reg [DATA_WIDTH-1:0]        w_data_reg;
reg [STRB_WIDTH-1:0]        w_strb_reg;
reg                         w_valid_reg;

// 读地址 FIFO
reg [ADDR_WIDTH-1:0]        ar_addr_reg;
reg                         ar_valid_reg;

// 寄存器文件（AXI 域，可写寄存器）
reg                         reg_ctrl_en;
reg                         reg_ctrl_sw_rst;
reg                         reg_ctrl_clear_bg;
reg                         reg_ctrl_roi_en;
reg                         reg_ctrl_force_trig;
reg [7:0]                   reg_thresh;
reg [19:0]                  reg_min_area;
reg                         reg_roi_trig;
reg [3:0]                   reg_ds_factor;
reg [1:0]                   reg_morph_cfg;
reg [3:0]                   reg_trig_n;
reg                         reg_mode;
reg [3:0]                   reg_bg_alpha;
reg [7:0]                   reg_roi_pad;
reg [11:0]                  reg_pix_w;
reg [11:0]                  reg_pix_h;
reg [7:0]                   reg_row_th;
reg [7:0]                   reg_col_th;

// 只读寄存器（AXI 域，从 pclk 域同步过来的快照）
reg [31:0]                  snap_status;
reg [31:0]                  snap_bbox_x;
reg [31:0]                  snap_bbox_y;
reg [31:0]                  snap_bbox_w;
reg [31:0]                  snap_bbox_h;
reg [31:0]                  snap_area;
reg [31:0]                  snap_energy;
reg [31:0]                  snap_frame_cnt;
reg [31:0]                  snap_proc_lat;
reg [31:0]                  snap_roi_sent;
reg [31:0]                  snap_snap_seq;
reg [31:0]                  snap_frame_ms;

// 采集链路调试快照
reg [31:0]                  snap_dbg_max_xaddr;
reg [31:0]                  snap_dbg_lines_last;
reg [31:0]                  snap_dbg_sof_cnt;
reg [31:0]                  snap_dbg_pix_lost;
reg [31:0]                  snap_dbg_fifo_ovf;
reg [31:0]                  snap_dbg_axis_frames;

// VDMA S2MM 引脚侧调试快照
reg [31:0]                  snap_dbg_vdma_beats;
reg [31:0]                  snap_dbg_vdma_maxbeat;
reg [31:0]                  snap_dbg_vdma_lines;
reg [31:0]                  snap_dbg_vdma_sof;
reg [31:0]                  snap_dbg_vdma_frames;
reg [31:0]                  snap_dbg_vdma_rdylow;
reg [31:0]                  snap_dbg_vdma_beatcur;
reg [31:0]                  snap_dbg_vdma_linecur;

// ============================================================
// AXI-Lite 写通道
// ============================================================

// 写地址通道：接受地址，存入寄存器
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        aw_valid_reg <= 1'b0;
        aw_addr_reg  <= {ADDR_WIDTH{1'b0}};
        s_axi_awready <= 1'b1;
    end else begin
        if (s_axi_awvalid && s_axi_awready) begin
            aw_addr_reg  <= s_axi_awaddr;
            aw_valid_reg <= 1'b1;
            s_axi_awready <= 1'b0;
        end else if (s_axi_bvalid && s_axi_bready) begin
            // 写响应完成，可以接受新地址
            aw_valid_reg <= 1'b0;
            s_axi_awready <= 1'b1;
        end
    end
end

// 写数据通道：接受数据，存入寄存器
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        w_valid_reg <= 1'b0;
        w_data_reg  <= {DATA_WIDTH{1'b0}};
        w_strb_reg  <= {STRB_WIDTH{1'b0}};
        s_axi_wready <= 1'b1;
    end else begin
        if (s_axi_wvalid && s_axi_wready) begin
            w_data_reg  <= s_axi_wdata;
            w_strb_reg  <= s_axi_wstrb;
            w_valid_reg <= 1'b1;
            s_axi_wready <= 1'b0;
        end else if (s_axi_bvalid && s_axi_bready) begin
            w_valid_reg <= 1'b0;
            s_axi_wready <= 1'b1;
        end
    end
end

// 写响应通道：地址和数据都有效时，执行写并返回响应
reg  [31:0]  wr_reg_data;
wire [11:0]  wr_addr_word;
assign wr_addr_word = aw_addr_reg[11:0];

always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        s_axi_bvalid <= 1'b0;
        home_ctrl <= 10'd0;
        image_control <= 32'h00800000;
        home_feed <= 1'b0;
        // 可写寄存器初始值
        reg_ctrl_en        <= 1'b0;
        reg_ctrl_sw_rst    <= 1'b0;
        reg_ctrl_clear_bg  <= 1'b0;
        reg_ctrl_roi_en    <= 1'b1;
        reg_ctrl_force_trig<= 1'b0;
        reg_thresh         <= 8'd24;
        reg_min_area       <= 20'd200;
        reg_roi_trig       <= 1'b0;
        reg_ds_factor      <= 4'd8;
        reg_morph_cfg      <= 2'd3;
        reg_trig_n         <= 4'd3;
        reg_mode           <= 1'b0;
        reg_bg_alpha       <= 4'd4;
        reg_roi_pad        <= 8'd8;
        reg_pix_w          <= 12'd1280;
        reg_pix_h          <= 12'd720;
        reg_row_th         <= 8'd3;
        reg_col_th         <= 8'd3;
    end else begin
        // 自清零脉冲
        reg_ctrl_sw_rst    <= 1'b0;
        reg_ctrl_clear_bg  <= 1'b0;
        reg_ctrl_force_trig<= 1'b0;

        home_feed <= 1'b0;
        if (aw_valid_reg && w_valid_reg && !s_axi_bvalid) begin
            // 执行写操作
            case (wr_addr_word[11:2])  // 字地址
                5'h00: begin  // CTRL
                    if (w_strb_reg[0]) begin
                        reg_ctrl_en        <= w_data_reg[0];
                        reg_ctrl_sw_rst    <= w_data_reg[1];  // 自清零
                        reg_ctrl_clear_bg  <= w_data_reg[2];  // 自清零
                        reg_ctrl_roi_en    <= w_data_reg[3];
                        reg_ctrl_force_trig<= w_data_reg[4];  // 自清零
                    end
                end
                5'h02: begin  // THRESH
                    if (w_strb_reg[0]) reg_thresh <= w_data_reg[7:0];
                end
                5'h03: begin  // MIN_AREA
                    if (w_strb_reg[0]) reg_min_area[7:0]  <= w_data_reg[7:0];
                    if (w_strb_reg[1]) reg_min_area[15:8] <= w_data_reg[15:8];
                    if (w_strb_reg[2]) reg_min_area[19:16]<= w_data_reg[19:16];
                end
                5'h0B: begin  // ROI_TRIG
                    if (w_strb_reg[0]) reg_roi_trig <= w_data_reg[0];
                end
                5'h0C: begin  // DS_FACTOR
                    if (w_strb_reg[0]) reg_ds_factor <= w_data_reg[3:0];
                end
                5'h0D: begin  // MORPH_CFG
                    if (w_strb_reg[0]) reg_morph_cfg <= w_data_reg[1:0];
                end
                5'h0E: begin  // TRIG_N
                    if (w_strb_reg[0]) reg_trig_n <= w_data_reg[3:0];
                end
                5'h0F: begin  // MODE
                    if (w_strb_reg[0]) reg_mode <= w_data_reg[0];
                end
                5'h10: begin  // BG_ALPHA
                    if (w_strb_reg[0]) reg_bg_alpha <= w_data_reg[3:0];
                end
                5'h11: begin  // ROI_PAD
                    if (w_strb_reg[0]) reg_roi_pad <= w_data_reg[7:0];
                end
                5'h13: begin  // PIX_W
                    if (w_strb_reg[0]) reg_pix_w[7:0]   <= w_data_reg[7:0];
                    if (w_strb_reg[1]) reg_pix_w[11:8]  <= w_data_reg[11:8];
                end
                5'h14: begin  // PIX_H
                    if (w_strb_reg[0]) reg_pix_h[7:0]   <= w_data_reg[7:0];
                    if (w_strb_reg[1]) reg_pix_h[11:8]  <= w_data_reg[11:8];
                end
                5'h18: begin  // ROW_TH
                    if (w_strb_reg[0]) reg_row_th <= w_data_reg[7:0];
                end
                5'h19: begin  // COL_TH
                    if (w_strb_reg[0]) reg_col_th <= w_data_reg[7:0];
                end
                10'h041: begin // HOME_CTRL 0x104, AXI clock domain only
                    if (w_strb_reg[0]) home_ctrl[7:0] <= w_data_reg[7:0];
                    if (w_strb_reg[1]) home_ctrl[9:8] <= w_data_reg[9:8];
                    if (|w_strb_reg[1:0]) home_feed <= 1'b1;
                end
                10'h051: begin // IMG_CTRL, AXI clock domain
                    if(w_strb_reg[0]) image_control[7:0]<=w_data_reg[7:0];
                    if(w_strb_reg[1]) image_control[15:8]<=w_data_reg[15:8];
                    if(w_strb_reg[2]) image_control[23:16]<=w_data_reg[23:16];
                    if(w_strb_reg[3]) image_control[31:24]<=w_data_reg[31:24];
                end
                default: ;
            endcase
            s_axi_bvalid <= 1'b1;
        end else if (s_axi_bvalid && s_axi_bready) begin
            s_axi_bvalid <= 1'b0;
        end
    end
end

assign s_axi_bresp = 2'b00;  // OKAY

// ============================================================
// AXI-Lite 读通道
// ============================================================

// 读地址通道
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        ar_valid_reg   <= 1'b0;
        ar_addr_reg    <= {ADDR_WIDTH{1'b0}};
        s_axi_arready  <= 1'b1;
    end else begin
        if (s_axi_arvalid && s_axi_arready) begin
            ar_addr_reg   <= s_axi_araddr;
            ar_valid_reg  <= 1'b1;
            s_axi_arready <= 1'b0;
        end else if (s_axi_rvalid && s_axi_rready) begin
            ar_valid_reg  <= 1'b0;
            s_axi_arready <= 1'b1;
        end
    end
end

// 读数据通道：组合逻辑输出
reg [31:0] rdata_reg;
wire [11:0] rd_addr_word;
assign rd_addr_word = ar_addr_reg[11:0];

always @(*) begin
    case (rd_addr_word[11:2])
        5'h00:  rdata_reg = {27'd0, reg_ctrl_force_trig, reg_ctrl_roi_en, reg_ctrl_clear_bg, reg_ctrl_sw_rst, reg_ctrl_en};
        5'h01:  rdata_reg = snap_status;
        5'h02:  rdata_reg = {24'd0, reg_thresh};
        5'h03:  rdata_reg = {12'd0, reg_min_area};
        5'h04:  rdata_reg = snap_bbox_x;
        5'h05:  rdata_reg = snap_bbox_y;
        5'h06:  rdata_reg = snap_bbox_w;
        5'h07:  rdata_reg = snap_bbox_h;
        5'h08:  rdata_reg = snap_area;
        5'h09:  rdata_reg = snap_energy;
        5'h0A:  rdata_reg = snap_frame_cnt;
        5'h0B:  rdata_reg = {31'd0, reg_roi_trig};
        5'h0C:  rdata_reg = {28'd0, reg_ds_factor};
        5'h0D:  rdata_reg = {30'd0, reg_morph_cfg};
        5'h0E:  rdata_reg = {28'd0, reg_trig_n};
        5'h0F:  rdata_reg = {31'd0, reg_mode};
        5'h10:  rdata_reg = {28'd0, reg_bg_alpha};
        5'h11:  rdata_reg = {24'd0, reg_roi_pad};
        5'h12:  rdata_reg = snap_proc_lat;
        5'h13:  rdata_reg = {20'd0, reg_pix_w};
        5'h14:  rdata_reg = {20'd0, reg_pix_h};
        5'h15:  rdata_reg = snap_roi_sent;
        5'h16:  rdata_reg = 32'h0001_0000;  // VERSION: 1.0
        5'h17:  rdata_reg = snap_snap_seq;
        5'h18:  rdata_reg = {24'd0, reg_row_th};
        5'h19:  rdata_reg = {24'd0, reg_col_th};
        5'h1A:  rdata_reg = snap_frame_ms;
        // 采集链路调试寄存器 (只读)
        5'h1B:  rdata_reg = snap_dbg_max_xaddr;    // 每行像素数(最大列号)
        5'h1C:  rdata_reg = snap_dbg_lines_last;   // 每帧行数(EOL 计数)
        5'h1D:  rdata_reg = snap_dbg_pix_lost;     // 每帧丢失像素数(应为0)
        5'h1E:  rdata_reg = snap_dbg_fifo_ovf;     // FIFO 溢出累计(应为0)
        5'h1F:  rdata_reg = snap_dbg_axis_frames;  // AXI-Stream 侧帧计数
        6'h20:  rdata_reg = snap_dbg_sof_cnt;      // SOF 累计数
        // VDMA S2MM 引脚侧调试寄存器 (只读)
        6'h21:  rdata_reg = snap_dbg_vdma_beats;   // 上一行 32bit 拍数 (期望640)
        6'h22:  rdata_reg = snap_dbg_vdma_maxbeat; // 最大拍数/行
        6'h23:  rdata_reg = snap_dbg_vdma_lines;   // 上一帧行数 (期望720)
        6'h24:  rdata_reg = snap_dbg_vdma_sof;     // SOF 累计数
        6'h25:  rdata_reg = snap_dbg_vdma_frames;  // VDMA 侧帧计数
        6'h26:  rdata_reg = snap_dbg_vdma_rdylow;  // 反压周期数
        6'h27:  rdata_reg = snap_dbg_vdma_beatcur; // 当前行拍数
        6'h28:  rdata_reg = snap_dbg_vdma_linecur; // 当前帧行数
        10'h040: rdata_reg = 32'h484F4D31; // HOM1: explicit capability probe
        10'h041: rdata_reg = {22'd0, home_ctrl};
        10'h042: rdata_reg = home_status;
        10'h043: rdata_reg = 32'd2500; // fixed hardware tone, Hz
        10'h044: rdata_reg = 32'h53574D31; // SWM1，拨码模式接口
        10'h045: rdata_reg = {31'd0, recognition_mode_sw};
        10'h048: rdata_reg = 32'h454E5631; // ENV1
        10'h049: rdata_reg = env_status;
        10'h04a: rdata_reg = env_payload;
        10'h04b: rdata_reg = env_sample_seq;
        10'h04c: rdata_reg = env_age_ms;
        10'h04d: rdata_reg = env_errors;
        10'h04e: rdata_reg = env_attempts;
        10'h050: rdata_reg = 32'h494D4731; // IMG1
        10'h051: rdata_reg = image_control;
        10'h052: rdata_reg = image_frames;
        default: rdata_reg = 32'd0;
    endcase
end

always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        s_axi_rvalid <= 1'b0;
    end else begin
        if (ar_valid_reg && !s_axi_rvalid) begin
            s_axi_rvalid <= 1'b1;
        end else if (s_axi_rvalid && s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end
    end
end

assign s_axi_rdata = rdata_reg;
assign s_axi_rresp = 2'b00;  // OKAY

// ============================================================
// 跨时钟域：AXI → pclk（配置寄存器）
// ============================================================
// 使用脉冲同步 + 数据保持方式
// 配置寄存器变化缓慢，使用双寄存器同步

// 写更新脉冲（AXI 域）
reg wr_update_pulse;
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn)
        wr_update_pulse <= 1'b0;
    else
        wr_update_pulse <= aw_valid_reg && w_valid_reg && !s_axi_bvalid;
end

// 脉冲同步到 pclk 域（三级同步 + 边沿检测）
reg [2:0] wr_update_sync;
always @(posedge pclk or posedge prst) begin
    if (prst)
        wr_update_sync <= 3'd0;
    else
        wr_update_sync <= {wr_update_sync[1:0], wr_update_pulse};
end
wire wr_update_pclk = wr_update_sync[2] ^ wr_update_sync[1];

// 配置数据在 pclk 域的同步
// 由于数据稳定（写入后不变，直到下次写入），直接采样是安全的
// 这里使用打拍方式增强可靠性
always @(posedge pclk or posedge prst) begin
    if (prst) begin
        cfg_en         <= 1'b0;
        cfg_sw_rst     <= 1'b0;
        cfg_clear_bg   <= 1'b0;
        cfg_roi_en     <= 1'b1;
        cfg_force_trig <= 1'b0;
        cfg_thresh     <= 8'd24;
        cfg_min_area   <= 20'd200;
        cfg_roi_trig   <= 1'b0;
        cfg_ds_factor  <= 4'd8;
        cfg_morph_cfg  <= 2'd3;
        cfg_trig_n     <= 4'd3;
        cfg_mode       <= 1'b0;
        cfg_bg_alpha   <= 4'd4;
        cfg_roi_pad    <= 8'd8;
        cfg_pix_w      <= 12'd1280;
        cfg_pix_h      <= 12'd720;
        cfg_row_th     <= 8'd3;
        cfg_col_th     <= 8'd3;
    end else begin
        // 脉冲信号（自清零）
        cfg_sw_rst     <= 1'b0;
        cfg_clear_bg   <= 1'b0;
        cfg_force_trig <= 1'b0;

        if (wr_update_pclk) begin
            // 锁存配置值（数据在 AXI 域稳定，直接采样安全）
            cfg_en         <= reg_ctrl_en;
            cfg_sw_rst     <= reg_ctrl_sw_rst;     // 脉冲
            cfg_clear_bg   <= reg_ctrl_clear_bg;   // 脉冲
            cfg_roi_en     <= reg_ctrl_roi_en;
            cfg_force_trig <= reg_ctrl_force_trig; // 脉冲
            cfg_thresh     <= reg_thresh;
            cfg_min_area   <= reg_min_area;
            cfg_roi_trig   <= reg_roi_trig;
            cfg_ds_factor  <= reg_ds_factor;
            cfg_morph_cfg  <= reg_morph_cfg;
            cfg_trig_n     <= reg_trig_n;
            cfg_mode       <= reg_mode;
            cfg_bg_alpha   <= reg_bg_alpha;
            cfg_roi_pad    <= reg_roi_pad;
            cfg_pix_w      <= reg_pix_w;
            cfg_pix_h      <= reg_pix_h;
            cfg_row_th     <= reg_row_th;
            cfg_col_th     <= reg_col_th;
        end
    end
end

// ============================================================
// 跨时钟域：pclk → AXI（状态快照 + 序号协议）
// ============================================================
// snap_seq 使用格雷码 + 双寄存器同步
// 状态数据在帧间稳定，检测到 snap_seq 变化时直接采样

// pclk 域：格雷码转换
reg [7:0] snap_seq_gray_pclk;
always @(posedge pclk or posedge prst) begin
    if (prst)
        snap_seq_gray_pclk <= 8'd0;
    else
        snap_seq_gray_pclk <= sts_snap_seq ^ (sts_snap_seq >> 1);
end

// AXI 域：格雷码双同步
reg [7:0] snap_seq_gray_sync1;
reg [7:0] snap_seq_gray_sync2;
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        snap_seq_gray_sync1 <= 8'd0;
        snap_seq_gray_sync2 <= 8'd0;
    end else begin
        snap_seq_gray_sync1 <= snap_seq_gray_pclk;
        snap_seq_gray_sync2 <= snap_seq_gray_sync1;
    end
end

// 格雷码转二进制（AXI 域）
reg [7:0] snap_seq_bin_axi;
integer gi;
always @(*) begin
    snap_seq_bin_axi[7] = snap_seq_gray_sync2[7];
    for (gi = 6; gi >= 0; gi = gi - 1)
        snap_seq_bin_axi[gi] = snap_seq_bin_axi[gi+1] ^ snap_seq_gray_sync2[gi];
end

// 检测 snap_seq 变化，更新快照寄存器
reg [7:0] snap_seq_prev;
wire snap_update_axi;
assign snap_update_axi = (snap_seq_bin_axi != snap_seq_prev);

always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        snap_seq_prev   <= 8'd0;
        snap_status     <= 32'd0;
        snap_bbox_x     <= 32'd0;
        snap_bbox_y     <= 32'd0;
        snap_bbox_w     <= 32'd0;
        snap_bbox_h     <= 32'd0;
        snap_area       <= 32'd0;
        snap_energy     <= 32'd0;
        snap_frame_cnt  <= 32'd0;
        snap_proc_lat   <= 32'd0;
        snap_roi_sent   <= 32'd0;
        snap_snap_seq   <= 32'd0;
        snap_frame_ms   <= 32'd0;
        snap_dbg_max_xaddr   <= 32'd0;
        snap_dbg_lines_last  <= 32'd0;
        snap_dbg_sof_cnt     <= 32'd0;
        snap_dbg_pix_lost    <= 32'd0;
        snap_dbg_fifo_ovf    <= 32'd0;
        snap_dbg_axis_frames <= 32'd0;
        snap_dbg_vdma_beats    <= 32'd0;
        snap_dbg_vdma_maxbeat  <= 32'd0;
        snap_dbg_vdma_lines    <= 32'd0;
        snap_dbg_vdma_sof      <= 32'd0;
        snap_dbg_vdma_frames   <= 32'd0;
        snap_dbg_vdma_rdylow   <= 32'd0;
        snap_dbg_vdma_beatcur  <= 32'd0;
        snap_dbg_vdma_linecur  <= 32'd0;
    end else if (snap_update_axi) begin
        snap_seq_prev <= snap_seq_bin_axi;
        // 直接采样 pclk 域数据（数据在帧间稳定，安全）
        snap_status   <= {26'd0, sts_bbox_upd, sts_motion, sts_roi_done, sts_roi_active, sts_bg_valid, sts_busy};
        snap_bbox_x   <= {20'd0, sts_bbox_x};
        snap_bbox_y   <= {20'd0, sts_bbox_y};
        snap_bbox_w   <= {20'd0, sts_bbox_w};
        snap_bbox_h   <= {20'd0, sts_bbox_h};
        snap_area     <= {12'd0, sts_area};
        snap_energy   <= {8'd0, sts_energy};
        snap_frame_cnt<= {16'd0, sts_frame_cnt};
        snap_proc_lat <= {16'd0, sts_proc_lat};
        snap_roi_sent <= {16'd0, sts_roi_sent};
        snap_snap_seq <= {24'd0, sts_snap_seq};
        snap_frame_ms <= {16'd0, sts_frame_ms};
        // 采集链路调试：直接采样（计数器在帧间稳定，安全）
        snap_dbg_max_xaddr   <= {16'd0, sts_dbg_max_xaddr};
        snap_dbg_lines_last  <= {16'd0, sts_dbg_lines_last};
        snap_dbg_sof_cnt     <= {16'd0, sts_dbg_sof_cnt};
        snap_dbg_pix_lost    <= {16'd0, sts_dbg_pix_lost};
        snap_dbg_fifo_ovf    <= {16'd0, sts_dbg_fifo_ovf};
        snap_dbg_axis_frames <= {16'd0, sts_dbg_axis_frames};
        snap_dbg_vdma_beats    <= {16'd0, sts_dbg_vdma_beats};
        snap_dbg_vdma_maxbeat  <= {16'd0, sts_dbg_vdma_maxbeat};
        snap_dbg_vdma_lines    <= {16'd0, sts_dbg_vdma_lines};
        snap_dbg_vdma_sof      <= {16'd0, sts_dbg_vdma_sof};
        snap_dbg_vdma_frames   <= {16'd0, sts_dbg_vdma_frames};
        snap_dbg_vdma_rdylow   <= {16'd0, sts_dbg_vdma_rdylow};
        snap_dbg_vdma_beatcur  <= {16'd0, sts_dbg_vdma_beatcur};
        snap_dbg_vdma_linecur  <= {16'd0, sts_dbg_vdma_linecur};
    end
end

endmodule
