/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_detect_top
// Description   : 一级运动检测引擎顶层模块
//                 - 例化所有子模块并完成互连
//                 - AXI-Lite 寄存器接口（100MHz）
//                 - 像素处理流水线（pclk 域）
//                 - ROI AXI-Stream 输出（pclk 域，96x96 灰度）
//
// Dependencies  : motion_sampler, motion_framestore, motion_diff,
//                 motion_morph, motion_projection, motion_fsm,
//                 roi_scaler, motion_regfile
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_detect_top(
    // ===== AXI-Lite 时钟/复位 =====
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


    // ===== AXI-Lite 从接口 =====
    s_axi_awaddr,
    s_axi_awvalid,
    s_axi_awready,
    s_axi_wdata,
    s_axi_wstrb,
    s_axi_wvalid,
    s_axi_wready,
    s_axi_bresp,
    s_axi_bvalid,
    s_axi_bready,
    s_axi_araddr,
    s_axi_arvalid,
    s_axi_arready,
    s_axi_rdata,
    s_axi_rresp,
    s_axi_rvalid,
    s_axi_rready,

    // ===== 像素时钟/复位 =====
    aclk_cam,
    aresetn_cam,

    // ===== 相机输入（pclk 域）=====
    cam_data,
    cam_valid,
    cam_xaddr,
    cam_yaddr,
    cam_vs,

    // ===== ROI 输出（AXI-Stream，pclk 域）=====
    m_axis_roi_tdata,
    m_axis_roi_tvalid,
    m_axis_roi_tuser,
    m_axis_roi_tlast,
    m_axis_roi_tready,

    // ===== 采集链路调试计数器（cam_pclk 域，只读）=====
    dbg_max_xaddr,
    dbg_lines_last,
    dbg_sof_cnt,
    dbg_pix_lost,
    dbg_fifo_ovf,
    dbg_axis_frames,

    // ===== VDMA S2MM 引脚侧调试计数器（cam_pclk 域，只读）=====
    dbg_vdma_beats,
    dbg_vdma_maxbeat,
    dbg_vdma_lines,
    dbg_vdma_sof,
    dbg_vdma_frames,
    dbg_vdma_rdylow,
    dbg_vdma_beatcur,
    dbg_vdma_linecur
);

// ============================================================
// 端口声明
// ============================================================
// AXI-Lite
input           s_axi_aclk;
input           s_axi_aresetn;
output [9:0]    home_ctrl;
output          home_feed;
input [31:0]    home_status;
input recognition_mode_sw;
output [31:0] image_control;
input [31:0] image_frames;
input [31:0] env_status;
input [31:0] env_payload;
input [31:0] env_sample_seq;
input [31:0] env_age_ms;
input [31:0] env_errors;
input [31:0] env_attempts;


input   [11:0]  s_axi_awaddr;
input           s_axi_awvalid;
output          s_axi_awready;
input   [31:0]  s_axi_wdata;
input   [3:0]   s_axi_wstrb;
input           s_axi_wvalid;
output          s_axi_wready;
output  [1:0]   s_axi_bresp;
output          s_axi_bvalid;
input           s_axi_bready;
input   [11:0]  s_axi_araddr;
input           s_axi_arvalid;
output          s_axi_arready;
output  [31:0]  s_axi_rdata;
output  [1:0]   s_axi_rresp;
output          s_axi_rvalid;
input           s_axi_rready;

// 像素时钟域
input           aclk_cam;
input           aresetn_cam;

// 相机输入
input   [15:0]  cam_data;
input           cam_valid;
input   [11:0]  cam_xaddr;
input   [11:0]  cam_yaddr;
input           cam_vs;

// ROI AXI-Stream 输出
output  [7:0]   m_axis_roi_tdata;
output          m_axis_roi_tvalid;
output          m_axis_roi_tuser;
output          m_axis_roi_tlast;
input           m_axis_roi_tready;

// 采集链路调试计数器 (cam_pclk 域, 只读, 由 fall_top 驱动)
input   [15:0]  dbg_max_xaddr;
input   [15:0]  dbg_lines_last;
input   [15:0]  dbg_sof_cnt;
input   [15:0]  dbg_pix_lost;
input   [15:0]  dbg_fifo_ovf;
input   [15:0]  dbg_axis_frames;

// VDMA S2MM 引脚侧调试计数器 (cam_pclk 域, 只读, 由 fall_top 驱动)
input   [15:0]  dbg_vdma_beats;      // 上一行 32bit 拍数   (期望 640)
input   [15:0]  dbg_vdma_maxbeat;    // 观测到的最大拍数/行
input   [15:0]  dbg_vdma_lines;      // 上一帧行数          (期望 720)
input   [15:0]  dbg_vdma_sof;        // 送到 VDMA 的 SOF 累计数
input   [15:0]  dbg_vdma_frames;     // VDMA 侧帧计数
input   [15:0]  dbg_vdma_rdylow;     // 反压周期数
input   [15:0]  dbg_vdma_beatcur;    // 当前行已送出拍数
input   [15:0]  dbg_vdma_linecur;    // 当前帧已送出行数

// ============================================================
// 内部信号
// ============================================================
// pclk 域复位（高有效）
wire            pclk_rst;
assign          pclk_rst = ~aresetn_cam;

// 全局使能（pclk 域）
wire            cfg_en;
wire            cfg_sw_rst;
wire            cfg_clear_bg;
wire            cfg_roi_en;
wire            cfg_force_trig;

// 配置寄存器（pclk 域）
wire    [7:0]   cfg_thresh;
wire    [19:0]  cfg_min_area;
wire            cfg_roi_trig;
wire    [3:0]   cfg_ds_factor;
wire    [1:0]   cfg_morph_cfg;
wire    [3:0]   cfg_trig_n;
wire            cfg_mode;
wire    [3:0]   cfg_bg_alpha;
wire    [7:0]   cfg_roi_pad;
wire    [11:0]  cfg_pix_w;
wire    [11:0]  cfg_pix_h;
wire    [7:0]   cfg_row_th;
wire    [7:0]   cfg_col_th;

// 采样器输出
wire            sample_valid;
wire    [7:0]   sample_luma;
wire    [13:0]  cell_idx;
wire    [7:0]   cell_x;
wire    [6:0]   cell_y;

// 采样器打拍（第一级：匹配帧存读延迟，送 diff 模块）
reg             sample_valid_r;
reg     [7:0]   sample_luma_r;
reg     [13:0]  cell_idx_r;
reg     [7:0]   cell_x_r;
reg     [6:0]   cell_y_r;

// 采样器打拍（第二级：匹配 diff 寄存器输出延迟，送 morph 模块）
reg             sample_valid_r2;
reg     [7:0]   cell_x_r2;
reg     [6:0]   cell_y_r2;

// 帧存输出
wire    [7:0]   ref_luma;
wire            frame_toggle;

// 帧差模块输出
wire            motion_bit;
wire            motion_valid;
wire    [23:0]  energy;
wire    [19:0]  area;
wire            clr_acc;

// 形态学输出
wire            morph_bit;
wire            morph_valid;
wire    [7:0]   morph_x;
wire    [6:0]   morph_y;

// 投影模块
wire            clr_sum;
wire    [6:0]   scan_row_idx;
wire    [7:0]   scan_col_idx;
wire            scan_en;
wire    [7:0]   row_sum_out;
wire    [6:0]   col_sum_out;

// FSM 输出
wire    [11:0]  bbox_x;
wire    [11:0]  bbox_y;
wire    [11:0]  bbox_w;
wire    [11:0]  bbox_h;
wire            bbox_valid;
wire            bbox_upd;
wire            motion_trig;
wire    [3:0]   hit_cnt;
wire    [7:0]   snap_seq;
wire    [15:0]  frame_cnt;

// ROI 缩放器
wire            roi_active;
wire            roi_done;
wire    [15:0]  roi_sent_cnt;
wire    [7:0]   cam_luma;

// 状态信号（pclk → regfile）
wire            sts_busy;
wire            sts_bg_valid;
reg             bg_valid_reg;

// 帧时间（暂用 0，后续可加计数器）
wire    [15:0]  frame_ms;
assign          frame_ms = 16'd0;

// 处理延迟（暂用 0）
wire    [15:0]  proc_lat;
assign          proc_lat = 16'd0;

// ============================================================
// ROI 亮度提取（绿色通道，与采样器一致）
// ============================================================
assign cam_luma = {cam_data[10:5], 2'b00};

// ============================================================
// 采样器打拍（第一级：匹配帧存 1 周期读延迟，送 diff 模块）
// ============================================================
always @(posedge aclk_cam or posedge pclk_rst) begin
    if (pclk_rst) begin
        sample_valid_r <= 1'b0;
        sample_luma_r  <= 8'd0;
        cell_idx_r     <= 14'd0;
        cell_x_r       <= 8'd0;
        cell_y_r       <= 7'd0;
    end else begin
        sample_valid_r <= sample_valid;
        if (sample_valid) begin
            sample_luma_r <= sample_luma;
            cell_idx_r    <= cell_idx;
            cell_x_r      <= cell_x;
            cell_y_r      <= cell_y;
        end
    end
end

// 采样器打拍（第二级：匹配 diff 模块 1 周期输出延迟，送 morph 模块）
always @(posedge aclk_cam or posedge pclk_rst) begin
    if (pclk_rst) begin
        sample_valid_r2 <= 1'b0;
        cell_x_r2       <= 8'd0;
        cell_y_r2       <= 7'd0;
    end else begin
        sample_valid_r2 <= sample_valid_r;
        if (sample_valid_r) begin
            cell_x_r2 <= cell_x_r;
            cell_y_r2 <= cell_y_r;
        end
    end
end

// ============================================================
// 背景有效标志（帧计数 > 1 时有效）
// ============================================================
always @(posedge aclk_cam or posedge pclk_rst) begin
    if (pclk_rst)
        bg_valid_reg <= 1'b0;
    else if (frame_cnt >= 16'd1)
        bg_valid_reg <= 1'b1;
end
assign sts_bg_valid = bg_valid_reg;

// 忙信号：FSM 非空闲
assign sts_busy = 1'b0;  // 简化：始终不忙

// ============================================================
// 1. 降采样器
// ============================================================
motion_sampler u_motion_sampler(
    .clk            ( aclk_cam      ),
    .rst            ( pclk_rst      ),
    .cam_data       ( cam_data      ),
    .cam_valid      ( cam_valid & cfg_en ),
    .cam_xaddr      ( cam_xaddr     ),
    .cam_yaddr      ( cam_yaddr     ),
    .ds_factor      ( cfg_ds_factor ),
    .sample_valid   ( sample_valid  ),
    .sample_luma    ( sample_luma   ),
    .cell_idx       ( cell_idx      ),
    .cell_x         ( cell_x        ),
    .cell_y         ( cell_y        )
);

// ============================================================
// 2. 帧存储器（双端口 BRAM 双缓冲）
// ============================================================
motion_framestore u_motion_framestore(
    .clk            ( aclk_cam      ),
    .rst            ( pclk_rst      ),
    .we             ( sample_valid & cfg_en ),
    .addr_a         ( cell_idx      ),
    .addr_b         ( cell_idx      ),
    .din            ( sample_luma   ),
    .dout_a         ( ref_luma      ),
    .frame_toggle   ( frame_toggle  ),
    .mode           ( cfg_mode      ),
    .bg_alpha       ( cfg_bg_alpha  ),
    .clear_bg       ( cfg_clear_bg  )
);

// ============================================================
// 3. 帧差计算
// ============================================================
motion_diff u_motion_diff(
    .clk            ( aclk_cam      ),
    .rst            ( pclk_rst      ),
    .valid          ( sample_valid_r ),
    .luma           ( sample_luma_r ),
    .ref_luma       ( ref_luma      ),
    .thresh         ( cfg_thresh    ),
    .motion_bit     ( motion_bit    ),
    .motion_valid   ( motion_valid  ),
    .energy         ( energy        ),
    .area           ( area          ),
    .frame_done     ( 1'b0          ),
    .clr_acc        ( clr_acc       )
);

// ============================================================
// 4. 形态学处理（开运算）
// ============================================================
motion_morph u_motion_morph(
    .clk            ( aclk_cam      ),
    .rst            ( pclk_rst      ),
    .din_valid      ( motion_valid  ),
    .din_bit        ( motion_bit    ),
    .cell_x         ( cell_x_r2     ),
    .cell_y         ( cell_y_r2     ),
    .dout_valid     ( morph_valid   ),
    .dout_bit       ( morph_bit     ),
    .dout_x         ( morph_x       ),
    .dout_y         ( morph_y       ),
    .morph_cfg      ( cfg_morph_cfg )
);

// ============================================================
// 5. 行列投影
// ============================================================
motion_projection u_motion_projection(
    .clk            ( aclk_cam      ),
    .rst            ( pclk_rst      ),
    .valid          ( morph_valid   ),
    .motion_bit     ( morph_bit     ),
    .row_idx        ( morph_y       ),
    .col_idx        ( morph_x       ),
    .scan_en        ( scan_en       ),
    .scan_row_idx   ( scan_row_idx  ),
    .scan_col_idx   ( scan_col_idx  ),
    .row_sum_out    ( row_sum_out   ),
    .col_sum_out    ( col_sum_out   ),
    .clr_sum        ( clr_sum       ),
    .row_th         ( cfg_row_th    ),
    .col_th         ( cfg_col_th    )
);

// ============================================================
// 6. 帧末状态机
// ============================================================
motion_fsm u_motion_fsm(
    .clk            ( aclk_cam      ),
    .rst            ( pclk_rst      ),
    .cam_vs         ( cam_vs        ),
    .area           ( area          ),
    .energy         ( energy        ),
    .min_area       ( cfg_min_area  ),
    .trig_n         ( cfg_trig_n    ),
    .ds_factor      ( cfg_ds_factor ),
    .roi_pad        ( cfg_roi_pad   ),
    .pix_w          ( cfg_pix_w     ),
    .pix_h          ( cfg_pix_h     ),
    .row_th         ( cfg_row_th    ),
    .col_th         ( cfg_col_th    ),
    .scan_row_idx   ( scan_row_idx  ),
    .scan_col_idx   ( scan_col_idx  ),
    .scan_en        ( scan_en       ),
    .row_sum_in     ( row_sum_out   ),
    .col_sum_in     ( col_sum_out   ),
    .frame_toggle   ( frame_toggle  ),
    .clr_acc        ( clr_acc       ),
    .clr_sum        ( clr_sum       ),
    .bbox_x         ( bbox_x        ),
    .bbox_y         ( bbox_y        ),
    .bbox_w         ( bbox_w        ),
    .bbox_h         ( bbox_h        ),
    .bbox_valid     ( bbox_valid    ),
    .bbox_upd       ( bbox_upd      ),
    .motion_trig    ( motion_trig   ),
    .hit_cnt        ( hit_cnt       ),
    .snap_seq       ( snap_seq      ),
    .frame_cnt      ( frame_cnt     ),
    .roi_en         ( cfg_roi_en    ),
    .force_trig     ( cfg_force_trig )
);

// ============================================================
// 7. ROI 裁剪与缩放
// ============================================================
roi_scaler u_roi_scaler(
    .clk            ( aclk_cam      ),
    .rst            ( pclk_rst      ),
    .bbox_valid     ( bbox_valid & motion_trig ),
    .bbox_x         ( bbox_x        ),
    .bbox_y         ( bbox_y        ),
    .bbox_w         ( bbox_w        ),
    .bbox_h         ( bbox_h        ),
    .cam_valid      ( cam_valid     ),
    .cam_luma       ( cam_luma      ),
    .cam_xaddr      ( cam_xaddr     ),
    .cam_yaddr      ( cam_yaddr     ),
    .cam_vs         ( cam_vs        ),
    .roi_en         ( cfg_roi_en    ),
    .trig           ( motion_trig & cfg_roi_trig ),
    .m_axis_tdata   ( m_axis_roi_tdata  ),
    .m_axis_tvalid  ( m_axis_roi_tvalid ),
    .m_axis_tuser   ( m_axis_roi_tuser  ),
    .m_axis_tlast   ( m_axis_roi_tlast  ),
    .m_axis_tready  ( m_axis_roi_tready ),
    .roi_active     ( roi_active    ),
    .roi_done       ( roi_done      ),
    .roi_sent_cnt   ( roi_sent_cnt  )
);

// ============================================================
// 8. AXI-Lite 寄存器文件
// ============================================================
motion_regfile u_motion_regfile(
    .s_axi_aclk     ( s_axi_aclk    ),
    .s_axi_aresetn  ( s_axi_aresetn ),
    .home_ctrl      ( home_ctrl ),
    .home_feed      ( home_feed ),
    .home_status    ( home_status ),
    .recognition_mode_sw (recognition_mode_sw),
    .image_control(image_control),.image_frames(image_frames),
    .env_status (env_status),
    .env_payload (env_payload),
    .env_sample_seq (env_sample_seq),
    .env_age_ms (env_age_ms),
    .env_errors (env_errors),
    .env_attempts (env_attempts),

    .s_axi_awaddr   ( s_axi_awaddr  ),
    .s_axi_awvalid  ( s_axi_awvalid ),
    .s_axi_awready  ( s_axi_awready ),
    .s_axi_wdata    ( s_axi_wdata   ),
    .s_axi_wstrb    ( s_axi_wstrb   ),
    .s_axi_wvalid   ( s_axi_wvalid  ),
    .s_axi_wready   ( s_axi_wready  ),
    .s_axi_bresp    ( s_axi_bresp   ),
    .s_axi_bvalid   ( s_axi_bvalid  ),
    .s_axi_bready   ( s_axi_bready  ),
    .s_axi_araddr   ( s_axi_araddr  ),
    .s_axi_arvalid  ( s_axi_arvalid ),
    .s_axi_arready  ( s_axi_arready ),
    .s_axi_rdata    ( s_axi_rdata   ),
    .s_axi_rresp    ( s_axi_rresp   ),
    .s_axi_rvalid   ( s_axi_rvalid  ),
    .s_axi_rready   ( s_axi_rready  ),
    .pclk           ( aclk_cam      ),
    .prst           ( pclk_rst      ),
    // 配置输出（AXI → pclk）
    .cfg_en         ( cfg_en        ),
    .cfg_sw_rst     ( cfg_sw_rst    ),
    .cfg_clear_bg   ( cfg_clear_bg  ),
    .cfg_roi_en     ( cfg_roi_en    ),
    .cfg_force_trig ( cfg_force_trig ),
    .cfg_thresh     ( cfg_thresh    ),
    .cfg_min_area   ( cfg_min_area  ),
    .cfg_roi_trig   ( cfg_roi_trig  ),
    .cfg_ds_factor  ( cfg_ds_factor ),
    .cfg_morph_cfg  ( cfg_morph_cfg ),
    .cfg_trig_n     ( cfg_trig_n    ),
    .cfg_mode       ( cfg_mode      ),
    .cfg_bg_alpha   ( cfg_bg_alpha  ),
    .cfg_roi_pad    ( cfg_roi_pad   ),
    .cfg_pix_w      ( cfg_pix_w     ),
    .cfg_pix_h      ( cfg_pix_h     ),
    .cfg_row_th     ( cfg_row_th    ),
    .cfg_col_th     ( cfg_col_th    ),
    // 状态输入（pclk → AXI）
    .sts_busy       ( sts_busy      ),
    .sts_bg_valid   ( sts_bg_valid  ),
    .sts_roi_active ( roi_active    ),
    .sts_roi_done   ( roi_done      ),
    .sts_motion     ( motion_trig   ),
    .sts_bbox_upd   ( bbox_upd      ),
    .sts_bbox_x     ( bbox_x        ),
    .sts_bbox_y     ( bbox_y        ),
    .sts_bbox_w     ( bbox_w        ),
    .sts_bbox_h     ( bbox_h        ),
    .sts_area       ( area          ),
    .sts_energy     ( energy        ),
    .sts_frame_cnt  ( frame_cnt     ),
    .sts_proc_lat   ( proc_lat      ),
    .sts_roi_sent   ( roi_sent_cnt  ),
    .sts_snap_seq   ( snap_seq      ),
    .sts_frame_ms   ( frame_ms      ),
    // 采集链路调试计数器
    .sts_dbg_max_xaddr   ( dbg_max_xaddr   ),
    .sts_dbg_lines_last  ( dbg_lines_last  ),
    .sts_dbg_sof_cnt     ( dbg_sof_cnt     ),
    .sts_dbg_pix_lost    ( dbg_pix_lost    ),
    .sts_dbg_fifo_ovf    ( dbg_fifo_ovf    ),
    .sts_dbg_axis_frames ( dbg_axis_frames ),
    .sts_dbg_vdma_beats    ( dbg_vdma_beats    ),
    .sts_dbg_vdma_maxbeat  ( dbg_vdma_maxbeat  ),
    .sts_dbg_vdma_lines    ( dbg_vdma_lines    ),
    .sts_dbg_vdma_sof      ( dbg_vdma_sof      ),
    .sts_dbg_vdma_frames   ( dbg_vdma_frames   ),
    .sts_dbg_vdma_rdylow   ( dbg_vdma_rdylow   ),
    .sts_dbg_vdma_beatcur  ( dbg_vdma_beatcur  ),
    .sts_dbg_vdma_linecur  ( dbg_vdma_linecur  )
);

endmodule
