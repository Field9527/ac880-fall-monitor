/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_diff
// Description   : 运动检测帧差计算模块
//                 - 计算当前亮度与参考亮度的绝对差值
//                 - 阈值比较，输出 motion_bit
//                 - 能量累加（24bit 饱和）
//                 - 面积计数（20bit）
//                 - 帧末清零累加器
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_diff(
    clk,
    rst,
    valid,
    luma,
    ref_luma,
    thresh,
    motion_bit,
    motion_valid,
    energy,
    area,
    frame_done,
    clr_acc
);

input               clk;                // 像素时钟 pclk
input               rst;                // 高有效复位
input               valid;              // 输入数据有效
input   [7:0]       luma;               // 当前亮度
input   [7:0]       ref_luma;           // 参考亮度
input   [7:0]       thresh;             // 运动阈值
output reg          motion_bit;         // 运动像素标记（1=运动）
output reg          motion_valid;       // motion_bit 有效
output reg [23:0]   energy;             // 能量累加（饱和 24bit）
output reg [19:0]   area;               // 面积计数（20bit，最大 14400）
input               frame_done;         // 帧结束脉冲（读累加值用）
input               clr_acc;            // 清零累加器

// 内部信号
reg  [7:0]          diff_abs;           // 绝对差值
wire                motion_bit_comb;    // 组合逻辑运动标记

// 绝对差值计算（组合逻辑）
always @(*) begin
    if (luma >= ref_luma)
        diff_abs = luma - ref_luma;
    else
        diff_abs = ref_luma - luma;
end

// 阈值比较
assign motion_bit_comb = (diff_abs > thresh) ? 1'b1 : 1'b0;

// 打拍输出
always @(posedge clk or posedge rst) begin
    if (rst) begin
        motion_bit   <= 1'b0;
        motion_valid <= 1'b0;
    end else begin
        motion_valid <= valid;
        if (valid) begin
            motion_bit <= motion_bit_comb;
        end
    end
end

// 能量累加（24bit 饱和累加）
// 最大可能值：14400 * 255 = 3,672,000 → 22bit 足够，24bit 留有余量
always @(posedge clk or posedge rst) begin
    if (rst) begin
        energy <= 24'd0;
    end else if (clr_acc) begin
        energy <= 24'd0;
    end else if (valid) begin
        if (&energy == 1'b0) begin  // 未饱和时累加
            energy <= energy + {16'd0, diff_abs};
        end
    end
end

// 面积计数（统计 motion_bit 为 1 的 cell 数量）
// 最大 14400 → 14bit 足够，给 20bit 留余量
always @(posedge clk or posedge rst) begin
    if (rst) begin
        area <= 20'd0;
    end else if (clr_acc) begin
        area <= 20'd0;
    end else if (valid && motion_bit_comb) begin
        area <= area + 20'd1;
    end
end

endmodule
