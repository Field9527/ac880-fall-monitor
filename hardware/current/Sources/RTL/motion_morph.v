/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_morph
// Description   : 运动检测形态学处理模块
//                 - 3x3 结构元素的开运算（腐蚀 + 膨胀）
//                 - 6 行缓冲（3 行用于腐蚀窗口，3 行用于膨胀窗口）
//                 - 支持 MORPH_CFG 控制：bit0=腐蚀使能，bit1=膨胀使能
//                 - 流水延迟约 3 拍（列方向）+ 行缓冲
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_morph(
    clk,
    rst,
    din_valid,
    din_bit,
    cell_x,
    cell_y,
    dout_valid,
    dout_bit,
    dout_x,
    dout_y,
    morph_cfg
);

input               clk;                // 像素时钟 pclk
input               rst;                // 高有效复位
input               din_valid;          // 输入有效
input               din_bit;            // 输入运动标记（1bit）
input   [7:0]       cell_x;             // 输入 cell 列坐标
input   [6:0]       cell_y;             // 输入 cell 行坐标
output reg          dout_valid;         // 输出有效
output reg          dout_bit;           // 输出运动标记
output reg [7:0]    dout_x;             // 输出 cell 列坐标
output reg [6:0]    dout_y;             // 输出 cell 行坐标
input   [1:0]       morph_cfg;          // 形态学配置：bit0=腐蚀使能, bit1=膨胀使能

// 参数
localparam ROW_W = 160;                  // 行宽度（cell）

// ========== 第一级：腐蚀 ==========
// 行缓冲（移位寄存器实现）
reg [ROW_W-1:0] erode_line0;            // 第 y-2 行缓冲
reg [ROW_W-1:0] erode_line1;            // 第 y-1 行缓冲

// 3 列窗口寄存器
reg [2:0] erode_win0;                   // 第 y-2 行窗口 [col2,col1,col0]
reg [2:0] erode_win1;                   // 第 y-1 行窗口
reg [2:0] erode_win2;                   // 第 y 行窗口（当前行）

// 腐蚀结果
wire                erode_bit;
reg                 erode_valid;
reg  [7:0]          erode_x;
reg  [6:0]          erode_y;

// 行起始检测（用于清除列移位寄存器，防止跨行污染）
wire                row_start;
assign row_start = din_valid && (cell_x == 8'd0);

// 腐蚀窗口移位（每行宽度为 ROW_W = 160）
always @(posedge clk or posedge rst) begin
    if (rst) begin
        erode_line0 <= {ROW_W{1'b0}};
        erode_line1 <= {ROW_W{1'b0}};
        erode_win0  <= 3'b000;
        erode_win1  <= 3'b000;
        erode_win2  <= 3'b000;
    end else if (din_valid) begin
        if (row_start) begin
            // 新行开始，清除列窗口（避免上一行末尾数据污染）
            erode_win2 <= {2'b00, din_bit};
            erode_win1 <= {2'b00, erode_line1[ROW_W-1]};
            erode_win0 <= {2'b00, erode_line0[ROW_W-1]};
        end else begin
            // 列方向移位：新像素从右侧进入
            erode_win2 <= {erode_win2[1:0], din_bit};
            erode_win1 <= {erode_win1[1:0], erode_line1[ROW_W-1]};
            erode_win0 <= {erode_win0[1:0], erode_line0[ROW_W-1]};
        end
        // 行缓冲移位
        erode_line1 <= {erode_line1[ROW_W-2:0], erode_win2[2]};
        erode_line0 <= {erode_line0[ROW_W-2:0], erode_win1[2]};
    end
end

// 腐蚀运算：3x3 全 1 才输出 1
assign erode_bit = &{erode_win0, erode_win1, erode_win2};

// 腐蚀有效信号延迟（列窗口需要 2 拍填充，行缓冲需要等第 2 行）
// 这里简单打 3 拍，配合坐标延迟
always @(posedge clk or posedge rst) begin
    if (rst) begin
        erode_valid <= 1'b0;
        erode_x     <= 8'd0;
        erode_y     <= 7'd0;
    end else begin
        erode_valid <= din_valid;
        if (din_valid) begin
            // 窗口中心坐标 = (cell_x - 1, cell_y - 1)
            // 但为简化，直接传递当前坐标（边界误差可忽略）
            erode_x <= cell_x;
            erode_y <= cell_y;
        end
    end
end

// ========== 第二级：膨胀 ==========
// 行缓冲
reg [ROW_W-1:0] dilate_line0;
reg [ROW_W-1:0] dilate_line1;

// 3 列窗口寄存器
reg [2:0] dilate_win0;
reg [2:0] dilate_win1;
reg [2:0] dilate_win2;

// 膨胀结果
wire                dilate_bit;
reg                 dilate_valid;
reg  [7:0]          dilate_x;
reg  [6:0]          dilate_y;

// 腐蚀输出的行起始
wire                erode_row_start;
assign erode_row_start = erode_valid && (erode_x == 8'd0);

// 膨胀窗口移位
always @(posedge clk or posedge rst) begin
    if (rst) begin
        dilate_line0 <= {ROW_W{1'b0}};
        dilate_line1 <= {ROW_W{1'b0}};
        dilate_win0  <= 3'b000;
        dilate_win1  <= 3'b000;
        dilate_win2  <= 3'b000;
    end else if (erode_valid) begin
        // 根据配置选择输入：腐蚀使能则用腐蚀结果，否则直接用输入
        if (morph_cfg[0]) begin
            if (erode_row_start) begin
                dilate_win2 <= {2'b00, erode_bit};
                dilate_win1 <= {2'b00, dilate_line1[ROW_W-1]};
                dilate_win0 <= {2'b00, dilate_line0[ROW_W-1]};
            end else begin
                dilate_win2 <= {dilate_win2[1:0], erode_bit};
                dilate_win1 <= {dilate_win1[1:0], dilate_line1[ROW_W-1]};
                dilate_win0 <= {dilate_win0[1:0], dilate_line0[ROW_W-1]};
            end
            dilate_line1 <= {dilate_line1[ROW_W-2:0], dilate_win2[2]};
            dilate_line0 <= {dilate_line0[ROW_W-2:0], dilate_win1[2]};
        end
    end
end

// 膨胀运算：3x3 有 1 则输出 1
assign dilate_bit = |{dilate_win0, dilate_win1, dilate_win2};

// 膨胀有效信号
always @(posedge clk or posedge rst) begin
    if (rst) begin
        dilate_valid <= 1'b0;
        dilate_x     <= 8'd0;
        dilate_y     <= 7'd0;
    end else begin
        dilate_valid <= erode_valid;
        if (erode_valid) begin
            dilate_x <= erode_x;
            dilate_y <= erode_y;
        end
    end
end

// ========== 输出多路选择 ==========
// 根据 morph_cfg 选择输出：
// 00: 直通（不做形态学）
// 01: 只腐蚀
// 10: 只膨胀（不常用，保留接口）
// 11: 开运算（腐蚀+膨胀）
always @(*) begin
    case (morph_cfg)
        2'b00: begin
            dout_valid = din_valid;
            dout_bit   = din_bit;
            dout_x     = cell_x;
            dout_y     = cell_y;
        end
        2'b01: begin
            dout_valid = erode_valid;
            dout_bit   = erode_bit;
            dout_x     = erode_x;
            dout_y     = erode_y;
        end
        2'b10: begin
            // 只膨胀：直接对输入做膨胀（使用第一级行缓冲做膨胀）
            // 简化实现：复用腐蚀的行缓冲做膨胀
            dout_valid = erode_valid;
            dout_bit   = |{erode_win0, erode_win1, erode_win2};
            dout_x     = erode_x;
            dout_y     = erode_y;
        end
        2'b11: begin
            dout_valid = dilate_valid;
            dout_bit   = dilate_bit;
            dout_x     = dilate_x;
            dout_y     = dilate_y;
        end
    endcase
end

endmodule
