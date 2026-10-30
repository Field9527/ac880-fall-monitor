/////////////////////////////////////////////////////////////////////////////////
// Module Name   : roi_scaler
// Description   : ROI 裁剪与缩放模块
//                 - 将输入图像中的 ROI 区域缩放到固定 96x96 灰度输出
//                 - 消隐期用 Bresenham 算法生成 96 项行/列地址 LUT
//                 - 像素流过时用指针比较法命中 LUT，输出 AXI-Stream
//                 - 输出：96x96 灰度，tuser=SOF, tlast=EOL
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module roi_scaler(
    clk,
    rst,
    // BBOX 输入（pclk 域，来自 FSM）
    bbox_valid,
    bbox_x,
    bbox_y,
    bbox_w,
    bbox_h,
    // 相机像素输入（pclk 域）
    cam_valid,
    cam_luma,
    cam_xaddr,
    cam_yaddr,
    cam_vs,
    // 触发使能
    roi_en,
    trig,
    // AXI-Stream 输出
    m_axis_tdata,
    m_axis_tvalid,
    m_axis_tuser,   // SOF：第一行第一个像素
    m_axis_tlast,   // EOL：每行最后一个像素
    m_axis_tready,
    // 状态
    roi_active,
    roi_done,
    roi_sent_cnt
);

input               clk;
input               rst;

input               bbox_valid;
input   [11:0]      bbox_x;
input   [11:0]      bbox_y;
input   [11:0]      bbox_w;
input   [11:0]      bbox_h;

input               cam_valid;
input   [7:0]       cam_luma;
input   [11:0]      cam_xaddr;
input   [11:0]      cam_yaddr;
input               cam_vs;

input               roi_en;
input               trig;

output  [7:0]       m_axis_tdata;
output reg          m_axis_tvalid;
output reg          m_axis_tuser;
output reg          m_axis_tlast;
input               m_axis_tready;

output reg          roi_active;
output reg          roi_done;
output reg [15:0]   roi_sent_cnt;

// 参数
localparam OUT_SIZE = 96;                // 输出尺寸 96x96
localparam OUT_IDX_W = 7;                // 0~95 需要 7bit

// ========== LUT 存储 ==========
reg [11:0] col_lut [0:OUT_SIZE-1];       // 列地址 LUT（绝对像素地址）
reg [11:0] row_lut [0:OUT_SIZE-1];       // 行地址 LUT（绝对像素地址）

// ========== LUT 生成状态机 ==========
localparam LUT_IDLE       = 4'd0;
localparam LUT_DIV_COL    = 4'd1;
localparam LUT_GEN_COL    = 4'd2;
localparam LUT_DIV_ROW    = 4'd3;
localparam LUT_GEN_ROW    = 4'd4;
localparam LUT_DONE       = 4'd5;

reg [3:0]  lut_state;
reg [11:0] div_count;                    // 除法迭代计数
reg [12:0] div_quotient;                 // 商（step）
reg [6:0]  div_remainder;                // 余数（rem，96以内，7bit）
reg [11:0] div_dividend;                 // 被除数
reg [11:0] div_acc;                      // 除法累加器

reg [6:0]  lut_idx;                      // LUT 索引（0~95）
reg [11:0] lut_x;                        // 当前源 x 坐标
reg [11:0] lut_y;                        // 当前源 y 坐标
reg [6:0]  lut_err;                      // 余数累加器

reg        lut_ready;                    // LUT 生成完成标志

// 迭代除法：dividend / 96 → quotient + remainder
// 12 次迭代，每次左移一位，减 96 试商
always @(posedge clk or posedge rst) begin
    if (rst) begin
        lut_state    <= LUT_IDLE;
        div_count    <= 12'd0;
        div_quotient <= 13'd0;
        div_remainder<= 7'd0;
        div_dividend <= 12'd0;
        div_acc      <= 12'd0;
        lut_idx      <= 7'd0;
        lut_x        <= 12'd0;
        lut_y        <= 12'd0;
        lut_err      <= 7'd0;
        lut_ready    <= 1'b0;
    end else begin
        case (lut_state)
            LUT_IDLE: begin
                lut_ready <= 1'b0;
                if (bbox_valid && roi_en) begin
                    // 开始列 LUT 生成：先做 bbox_w / 96
                    div_dividend <= bbox_w;
                    div_acc      <= 12'd0;
                    div_quotient <= 13'd0;
                    div_count    <= 12'd0;
                    lut_state    <= LUT_DIV_COL;
                end
            end

            LUT_DIV_COL: begin
                // 迭代除法（12 位）
                if (div_count < 12) begin
                    div_acc <= {div_acc[10:0], div_dividend[11 - div_count]};
                    if ({div_acc[10:0], div_dividend[11 - div_count]} >= 12'd96) begin
                        div_acc      <= {div_acc[10:0], div_dividend[11 - div_count]} - 12'd96;
                        div_quotient <= {div_quotient[11:0], 1'b1};
                    end else begin
                        div_acc      <= {div_acc[10:0], div_dividend[11 - div_count]};
                        div_quotient <= {div_quotient[11:0], 1'b0};
                    end
                    div_count <= div_count + 12'd1;
                end else begin
                    // 除法完成，开始生成列 LUT
                    lut_idx   <= 7'd0;
                    lut_x     <= bbox_x;
                    lut_err   <= 7'd0;
                    lut_state <= LUT_GEN_COL;
                end
            end

            LUT_GEN_COL: begin
                // 写入 col_lut[j] = lut_x
                col_lut[lut_idx] <= lut_x;
                // 下一步
                lut_x <= lut_x + div_quotient[11:0];  // + step
                lut_err <= lut_err + div_acc[6:0];    // + remainder（余数分布）
                if (lut_err + div_acc[6:0] >= 7'd96) begin
                    lut_x   <= lut_x + div_quotient[11:0] + 12'd1;
                    lut_err <= lut_err + div_acc[6:0] - 7'd96;
                end else begin
                    lut_x   <= lut_x + div_quotient[11:0];
                    lut_err <= lut_err + div_acc[6:0];
                end

                if (lut_idx >= OUT_SIZE - 1) begin
                    // 列 LUT 完成，开始行 LUT
                    div_dividend <= bbox_h;
                    div_acc      <= 12'd0;
                    div_quotient <= 13'd0;
                    div_count    <= 12'd0;
                    lut_state    <= LUT_DIV_ROW;
                end else begin
                    lut_idx <= lut_idx + 7'd1;
                end
            end

            LUT_DIV_ROW: begin
                // 迭代除法（bbox_h / 96）
                if (div_count < 12) begin
                    if ({div_acc[10:0], div_dividend[11 - div_count]} >= 12'd96) begin
                        div_acc      <= {div_acc[10:0], div_dividend[11 - div_count]} - 12'd96;
                        div_quotient <= {div_quotient[11:0], 1'b1};
                    end else begin
                        div_acc      <= {div_acc[10:0], div_dividend[11 - div_count]};
                        div_quotient <= {div_quotient[11:0], 1'b0};
                    end
                    div_count <= div_count + 12'd1;
                end else begin
                    lut_idx   <= 7'd0;
                    lut_y     <= bbox_y;
                    lut_err   <= 7'd0;
                    lut_state <= LUT_GEN_ROW;
                end
            end

            LUT_GEN_ROW: begin
                row_lut[lut_idx] <= lut_y;
                if (lut_err + div_acc[6:0] >= 7'd96) begin
                    lut_y   <= lut_y + div_quotient[11:0] + 12'd1;
                    lut_err <= lut_err + div_acc[6:0] - 7'd96;
                end else begin
                    lut_y   <= lut_y + div_quotient[11:0];
                    lut_err <= lut_err + div_acc[6:0];
                end

                if (lut_idx >= OUT_SIZE - 1) begin
                    lut_ready <= 1'b1;
                    lut_state <= LUT_DONE;
                end else begin
                    lut_idx <= lut_idx + 7'd1;
                end
            end

            LUT_DONE: begin
                // 等待下一帧或新 bbox
                if (bbox_valid) begin
                    // 新 bbox 到来，重新生成
                    div_dividend <= bbox_w;
                    div_acc      <= 12'd0;
                    div_quotient <= 13'd0;
                    div_count    <= 12'd0;
                    lut_ready    <= 1'b0;
                    lut_state    <= LUT_DIV_COL;
                end
            end

            default: lut_state <= LUT_IDLE;
        endcase
    end
end

// ========== 输出流水（指针比较法） ==========
// i: 输出行索引（0~95），对应源行 = row_lut[i]
// j: 输出列索引（0~95），对应源列 = col_lut[j]
// 当输入像素命中 LUT 时输出

reg [6:0]  out_i;        // 输出行指针
reg [6:0]  out_j;        // 输出列指针
reg        out_row_active; // 当前行是输出行

// vsync 检测：帧开始时重置指针
reg vs_r;
wire vs_negedge;
always @(posedge clk or posedge rst) begin
    if (rst)
        vs_r <= 1'b0;
    else
        vs_r <= cam_vs;
end
assign vs_negedge = vs_r && ~cam_vs;  // vs 下降沿 = 帧开始（vs 低有效期间为消隐）

// 触发帧捕获
reg        capture_en;    // 捕获使能
reg        sof_sent;      // SOF 是否已发送

always @(posedge clk or posedge rst) begin
    if (rst) begin
        capture_en <= 1'b0;
        roi_sent_cnt <= 16'd0;
    end else begin
        if (trig && roi_en && lut_ready) begin
            // 触发：开始捕获下一帧
            capture_en <= 1'b1;
        end else if (roi_done) begin
            // 捕获完成
            capture_en <= 1'b0;
            roi_sent_cnt <= roi_sent_cnt + 16'd1;
        end
    end
end

// 行/列指针与输出控制
always @(posedge clk or posedge rst) begin
    if (rst) begin
        out_i        <= 7'd0;
        out_j        <= 7'd0;
        out_row_active <= 1'b0;
        m_axis_tvalid <= 1'b0;
        m_axis_tuser  <= 1'b0;
        m_axis_tlast  <= 1'b0;
        roi_active   <= 1'b0;
        roi_done     <= 1'b0;
        sof_sent     <= 1'b0;
    end else if (vs_negedge && capture_en) begin
        // 帧开始，重置指针
        out_i        <= 7'd0;
        out_j        <= 7'd0;
        out_row_active <= 1'b0;
        roi_active   <= 1'b1;
        roi_done     <= 1'b0;
        sof_sent     <= 1'b0;
    end else if (cam_valid && roi_active && capture_en) begin
        // 检查当前行是否是输出行
        if (cam_yaddr == row_lut[out_i]) begin
            out_row_active <= 1'b1;
        end else if (cam_yaddr > row_lut[out_i]) begin
            // 已经过了当前输出行（理论上不应该发生，每行只有一个 yaddr 值）
            out_row_active <= 1'b0;
        end

        if (out_row_active || cam_yaddr == row_lut[out_i]) begin
            // 在输出行内，检查列是否命中
            if (out_j < OUT_SIZE) begin
                if (cam_xaddr >= col_lut[out_j]) begin
                    // 命中：输出该像素
                    m_axis_tvalid <= 1'b1;
                    m_axis_tuser  <= (out_i == 7'd0 && out_j == 7'd0 && ~sof_sent);
                    m_axis_tlast  <= (out_j == OUT_SIZE - 7'd1);
                    sof_sent      <= 1'b1;

                    out_j <= out_j + 7'd1;

                    if (out_j == OUT_SIZE - 7'd1) begin
                        // 行结束
                        out_i <= out_i + 7'd1;
                        out_j <= 7'd0;
                        out_row_active <= 1'b0;
                        if (out_i == OUT_SIZE - 7'd1) begin
                            // 帧结束
                            roi_done   <= 1'b1;
                            roi_active <= 1'b0;
                        end
                    end
                end else begin
                    m_axis_tvalid <= 1'b0;
                    m_axis_tuser  <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                end
            end else begin
                m_axis_tvalid <= 1'b0;
            end
        end else begin
            m_axis_tvalid <= 1'b0;
            m_axis_tuser  <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end
    end else begin
        m_axis_tvalid <= 1'b0;
        m_axis_tuser  <= 1'b0;
        m_axis_tlast  <= 1'b0;
        if (roi_done)
            roi_done <= 1'b0;  // 单周期脉冲
    end
end

// 输出数据
assign m_axis_tdata = cam_luma;

endmodule
