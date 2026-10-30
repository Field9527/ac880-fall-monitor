/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_fsm
// Description   : 运动检测帧末状态机
//                 - vsync 上升沿触发帧末处理
//                 - 扫描行/列投影，计算 bbox
//                 - 触发判定（面积 + 连续帧数）
//                 - 切换帧存页、清零累加器、更新快照
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_fsm(
    clk,
    rst,
    // 帧同步
    cam_vs,
    // 来自 diff 模块
    area,
    energy,
    // 配置参数（pclk 域）
    min_area,
    trig_n,
    ds_factor,
    roi_pad,
    pix_w,
    pix_h,
    row_th,
    col_th,
    // 投影扫描接口
    scan_row_idx,
    scan_col_idx,
    scan_en,
    row_sum_in,
    col_sum_in,
    // 控制输出
    frame_toggle,       // 帧存页切换
    clr_acc,            // 清零能量/面积累加器
    clr_sum,            // 清零投影累加
    // BBOX 输出（cell 坐标，已转像素坐标）
    bbox_x,
    bbox_y,
    bbox_w,
    bbox_h,
    bbox_valid,
    bbox_upd,           // bbox 更新脉冲
    // 触发输出
    motion_trig,        // 运动触发标志
    hit_cnt,            // 连续命中帧数
    // 快照
    snap_seq,
    frame_cnt,
    // ROI 相关
    roi_en,
    force_trig
);

input               clk;
input               rst;
input               cam_vs;             // 场同步（pclk 域）
input   [19:0]      area;               // 运动面积（cell 数）
input   [23:0]      energy;             // 运动能量
input   [19:0]      min_area;           // 最小触发面积
input   [3:0]       trig_n;             // 连续触发帧数阈值
input   [3:0]       ds_factor;          // 降采样因子
input   [7:0]       roi_pad;            // ROI 外扩像素
input   [11:0]      pix_w;              // 图像宽度（像素）
input   [11:0]      pix_h;              // 图像高度（像素）
input   [7:0]       row_th;             // 行有效阈值
input   [7:0]       col_th;             // 列有效阈值

output reg [6:0]    scan_row_idx;       // 行扫描地址
output reg [7:0]    scan_col_idx;       // 列扫描地址
output reg          scan_en;            // 扫描读使能
input   [7:0]       row_sum_in;         // 行投影值
input   [6:0]       col_sum_in;         // 列投影值

output reg          frame_toggle;
output reg          clr_acc;
output reg          clr_sum;

output reg [11:0]   bbox_x;
output reg [11:0]   bbox_y;
output reg [11:0]   bbox_w;
output reg [11:0]   bbox_h;
output reg          bbox_valid;
output reg          bbox_upd;

output reg          motion_trig;
output reg [3:0]    hit_cnt;

output reg [7:0]    snap_seq;
output reg [15:0]   frame_cnt;

input               roi_en;
input               force_trig;

// ========== 状态定义 ==========
localparam ST_IDLE       = 4'd0;
localparam ST_SCAN_ROWS  = 4'd1;
localparam ST_SCAN_COLS  = 4'd2;
localparam ST_CALC_BBOX  = 4'd3;
localparam ST_TRIG_CHECK = 4'd4;
localparam ST_UPDATE_SNAP= 4'd5;
localparam ST_DONE       = 4'd6;

reg [3:0] state;
reg [3:0] state_next;

// vsync 边沿检测
reg vs_r, vs_rr;
wire vs_posedge;
always @(posedge clk or posedge rst) begin
    if (rst) begin
        vs_r  <= 1'b0;
        vs_rr <= 1'b0;
    end else begin
        vs_r  <= cam_vs;
        vs_rr <= vs_r;
    end
end
assign vs_posedge = vs_r && ~vs_rr;

// 扫描计数器
reg [6:0] row_cnt;   // 0~89
reg [7:0] col_cnt;   // 0~159

// 行 min/max 寄存器
reg [6:0] row_min_reg;
reg [6:0] row_max_reg;
reg       row_found_first;

// 列 min/max 寄存器
reg [7:0] col_min_reg;
reg [7:0] col_max_reg;
reg       col_found_first;

// BBOX 计算中间值
reg [11:0] bbox_x0, bbox_y0, bbox_x1, bbox_y1;
reg [11:0] bbox_x0_clip, bbox_y0_clip, bbox_x1_clip, bbox_y1_clip;

// ========== 状态转移 ==========
always @(posedge clk or posedge rst) begin
    if (rst)
        state <= ST_IDLE;
    else
        state <= state_next;
end

always @(*) begin
    state_next = state;
    case (state)
        ST_IDLE: begin
            if (vs_posedge)
                state_next = ST_SCAN_ROWS;
        end
        ST_SCAN_ROWS: begin
            if (row_cnt >= 7'd89)
                state_next = ST_SCAN_COLS;
        end
        ST_SCAN_COLS: begin
            if (col_cnt >= 8'd159)
                state_next = ST_CALC_BBOX;
        end
        ST_CALC_BBOX: begin
            state_next = ST_TRIG_CHECK;
        end
        ST_TRIG_CHECK: begin
            state_next = ST_UPDATE_SNAP;
        end
        ST_UPDATE_SNAP: begin
            state_next = ST_DONE;
        end
        ST_DONE: begin
            state_next = ST_IDLE;
        end
        default: state_next = ST_IDLE;
    endcase
end

// ========== 行扫描 ==========
always @(posedge clk or posedge rst) begin
    if (rst) begin
        row_cnt          <= 7'd0;
        row_min_reg      <= 7'd0;
        row_max_reg      <= 7'd0;
        row_found_first  <= 1'b0;
    end else if (state == ST_SCAN_ROWS) begin
        if (row_cnt == 7'd0) begin
            // 第一拍：准备读地址 0
            row_cnt         <= 7'd1;
            row_found_first <= 1'b0;
        end else begin
            // 从第二拍开始，row_sum_in 有效（对应 row_cnt-1）
            if (row_sum_in > row_th) begin
                if (!row_found_first) begin
                    row_min_reg     <= row_cnt - 7'd1;
                    row_max_reg     <= row_cnt - 7'd1;
                    row_found_first <= 1'b1;
                end else begin
                    row_max_reg     <= row_cnt - 7'd1;
                end
            end
            row_cnt <= row_cnt + 7'd1;
        end
    end else if (state == ST_IDLE) begin
        row_cnt <= 7'd0;
    end
end

// ========== 列扫描 ==========
always @(posedge clk or posedge rst) begin
    if (rst) begin
        col_cnt          <= 8'd0;
        col_min_reg      <= 8'd0;
        col_max_reg      <= 8'd0;
        col_found_first  <= 1'b0;
    end else if (state == ST_SCAN_COLS) begin
        if (col_cnt == 8'd0) begin
            col_cnt         <= 8'd1;
            col_found_first <= 1'b0;
        end else begin
            if (col_sum_in > col_th) begin
                if (!col_found_first) begin
                    col_min_reg     <= col_cnt - 8'd1;
                    col_max_reg     <= col_cnt - 8'd1;
                    col_found_first <= 1'b1;
                end else begin
                    col_max_reg     <= col_cnt - 8'd1;
                end
            end
            col_cnt <= col_cnt + 8'd1;
        end
    end else if (state == ST_IDLE) begin
        col_cnt <= 8'd0;
    end
end

// 扫描地址输出
always @(*) begin
    scan_en = 1'b0;
    scan_row_idx = 7'd0;
    scan_col_idx = 8'd0;
    if (state == ST_SCAN_ROWS) begin
        scan_en = 1'b1;
        scan_row_idx = row_cnt;
    end else if (state == ST_SCAN_COLS) begin
        scan_en = 1'b1;
        scan_col_idx = col_cnt;
    end
end

// ========== BBOX 计算 ==========
// cell 坐标转像素坐标：
//   x0 = col_min * DS - PAD
//   y0 = row_min * DS - PAD
//   x1 = (col_max + 1) * DS - 1 + PAD
//   y1 = (row_max + 1) * DS - 1 + PAD
// 然后裁剪到图像范围内 [0, pix_w-1] / [0, pix_h-1]
always @(posedge clk or posedge rst) begin
    if (rst) begin
        bbox_x0 <= 12'd0;
        bbox_y0 <= 12'd0;
        bbox_x1 <= 12'd0;
        bbox_y1 <= 12'd0;
    end else if (state == ST_CALC_BBOX) begin
        if (row_found_first && col_found_first) begin
            // 原始 bbox（cell 转像素，加 pad）
            bbox_x0 <= col_min_reg * ds_factor - roi_pad;
            bbox_y0 <= row_min_reg * ds_factor - roi_pad;
            bbox_x1 <= (col_max_reg + 8'd1) * ds_factor - 12'd1 + roi_pad;
            bbox_y1 <= (row_max_reg + 7'd1) * ds_factor - 12'd1 + roi_pad;
        end else begin
            // 没有检测到运动，给默认 bbox
            bbox_x0 <= 12'd0;
            bbox_y0 <= 12'd0;
            bbox_x1 <= pix_w - 12'd1;
            bbox_y1 <= pix_h - 12'd1;
        end
    end
end

// 裁剪到图像范围 + 计算宽高
always @(*) begin
    bbox_x0_clip = (bbox_x0[11]) ? 12'd0 : bbox_x0;  // 负数处理
    bbox_y0_clip = (bbox_y0[11]) ? 12'd0 : bbox_y0;
    bbox_x1_clip = (bbox_x1 >= pix_w) ? (pix_w - 12'd1) : bbox_x1;
    bbox_y1_clip = (bbox_y1 >= pix_h) ? (pix_h - 12'd1) : bbox_y1;
end

// ========== 触发判定 ==========
always @(posedge clk or posedge rst) begin
    if (rst) begin
        hit_cnt     <= 4'd0;
        motion_trig <= 1'b0;
    end else if (state == ST_TRIG_CHECK) begin
        if (force_trig) begin
            // 强制触发
            hit_cnt     <= trig_n;
            motion_trig <= 1'b1;
        end else if (area >= min_area) begin
            // 面积达标，hit_cnt++
            if (hit_cnt < trig_n)
                hit_cnt <= hit_cnt + 4'd1;
            if (hit_cnt + 4'd1 >= trig_n)
                motion_trig <= 1'b1;
        end else begin
            // 面积不达标，清零
            hit_cnt     <= 4'd0;
            motion_trig <= 1'b0;
        end
    end
end

// ========== BBOX 输出寄存器 ==========
always @(posedge clk or posedge rst) begin
    if (rst) begin
        bbox_x     <= 12'd0;
        bbox_y     <= 12'd0;
        bbox_w     <= 12'd0;
        bbox_h     <= 12'd0;
        bbox_valid <= 1'b0;
        bbox_upd   <= 1'b0;
    end else if (state == ST_UPDATE_SNAP) begin
        bbox_x   <= bbox_x0_clip;
        bbox_y   <= bbox_y0_clip;
        bbox_w   <= bbox_x1_clip - bbox_x0_clip + 12'd1;
        bbox_h   <= bbox_y1_clip - bbox_y0_clip + 12'd1;
        bbox_valid <= row_found_first && col_found_first;
        bbox_upd   <= 1'b1;
    end else begin
        bbox_upd <= 1'b0;
    end
end

// ========== 控制信号 ==========
always @(posedge clk or posedge rst) begin
    if (rst) begin
        frame_toggle <= 1'b0;
        clr_acc      <= 1'b0;
        clr_sum      <= 1'b0;
    end else begin
        // frame_toggle：帧末切换 ping-pong 页
        frame_toggle <= (state == ST_DONE);
        // 清零累加器：帧末（下一帧开始前）
        clr_acc      <= (state == ST_DONE);
        clr_sum      <= (state == ST_DONE);
    end
end

// ========== 帧计数 & 快照序号 ==========
always @(posedge clk or posedge rst) begin
    if (rst) begin
        frame_cnt <= 16'd0;
        snap_seq  <= 8'd0;
    end else if (state == ST_UPDATE_SNAP) begin
        frame_cnt <= frame_cnt + 16'd1;
        snap_seq  <= snap_seq + 8'd1;
    end
end

endmodule
