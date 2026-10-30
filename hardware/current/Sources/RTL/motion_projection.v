/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_projection
// Description   : 运动检测行列投影模块
//                 - 逐像素累加行/列投影直方图
//                 - row_sum[90]：每行的运动像素数（8bit，最大160）
//                 - col_sum[160]：每列的运动像素数（7bit，最大90）
//                 - 提供扫描读端口，供 FSM 在帧末计算 min/max
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_projection(
    clk,
    rst,
    valid,
    motion_bit,
    row_idx,
    col_idx,
    // 扫描读端口（FSM 用）
    scan_en,
    scan_row_idx,
    scan_col_idx,
    row_sum_out,
    col_sum_out,
    // 控制
    clr_sum,
    // 行列阈值（用于判定有效行/列）
    row_th,
    col_th
);

input               clk;                // 像素时钟 pclk
input               rst;                // 高有效复位
input               valid;              // 输入有效
input               motion_bit;         // 运动标记
input   [6:0]       row_idx;            // 行索引（0~89）
input   [7:0]       col_idx;            // 列索引（0~159）
input               scan_en;            // 扫描读使能
input   [6:0]       scan_row_idx;       // 扫描行地址
input   [7:0]       scan_col_idx;       // 扫描列地址
output  [7:0]       row_sum_out;        // 行投影读出值
output  [6:0]       col_sum_out;        // 列投影读出值
input               clr_sum;            // 清零累加器
input   [7:0]       row_th;             // 行有效阈值
input   [7:0]       col_th;             // 列有效阈值

// 参数
localparam ROW_NUM = 90;                 // 行数
localparam COL_NUM = 160;                // 列数

// 行投影寄存器数组（90 行 × 8bit，每行最大 160 个运动像素）
reg [7:0] row_sum [0:ROW_NUM-1];
// 列投影寄存器数组（160 列 × 7bit，每列最大 90 个运动像素）
reg [6:0] col_sum [0:COL_NUM-1];

integer i;

// 行投影累加
always @(posedge clk or posedge rst) begin
    if (rst) begin
        for (i = 0; i < ROW_NUM; i = i + 1)
            row_sum[i] <= 8'd0;
    end else if (clr_sum) begin
        for (i = 0; i < ROW_NUM; i = i + 1)
            row_sum[i] <= 8'd0;
    end else if (valid && motion_bit) begin
        if (row_sum[row_idx] < 8'd160)  // 饱和保护
            row_sum[row_idx] <= row_sum[row_idx] + 8'd1;
    end
end

// 列投影累加
always @(posedge clk or posedge rst) begin
    if (rst) begin
        for (i = 0; i < COL_NUM; i = i + 1)
            col_sum[i] <= 7'd0;
    end else if (clr_sum) begin
        for (i = 0; i < COL_NUM; i = i + 1)
            col_sum[i] <= 7'd0;
    end else if (valid && motion_bit) begin
        if (col_sum[col_idx] < 7'd90)   // 饱和保护
            col_sum[col_idx] <= col_sum[col_idx] + 7'd1;
    end
end

// 扫描读端口（组合逻辑读出）
assign row_sum_out = (scan_en && (scan_row_idx < ROW_NUM)) ? row_sum[scan_row_idx] : 8'd0;
assign col_sum_out = (scan_en && (scan_col_idx < COL_NUM)) ? col_sum[scan_col_idx] : 7'd0;

endmodule
