/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_sampler
// Description   : 运动检测降采样器
//                 - 将 RGB565 像素降采样到 160x90 的 cell 网格
//                 - 提取绿色通道作为亮度值（6位扩展为8位）
//                 - 输出 cell 索引（row * 160 + col）
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_sampler(
    clk,
    rst,
    cam_data,
    cam_valid,
    cam_xaddr,
    cam_yaddr,
    ds_factor,
    sample_valid,
    sample_luma,
    cell_idx,
    cell_x,
    cell_y
);

input               clk;                // 像素时钟 pclk
input               rst;                // 高有效复位，pclk 域
input   [15:0]      cam_data;           // RGB565 输入
input               cam_valid;          // 像素有效
input   [11:0]      cam_xaddr;          // 像素 X 坐标
input   [11:0]      cam_yaddr;          // 像素 Y 坐标
input   [3:0]       ds_factor;          // 降采样因子：4/8/16
output reg          sample_valid;       // 采样有效
output reg [7:0]    sample_luma;        // 采样亮度（8位灰度）
output reg [13:0]   cell_idx;           // cell 索引 = y*160 + x
output reg [7:0]    cell_x;             // cell 列坐标（0~159）
output reg [6:0]    cell_y;             // cell 行坐标（0~89）

// 内部信号
reg                 sample_hit;         // 当前像素是否命中采样点
reg [7:0]           luma_comb;          // 组合逻辑亮度值
reg [13:0]          idx_comb;           // 组合逻辑 cell 索引
reg [7:0]           cx_comb;            // 组合逻辑 cell_x
reg [6:0]           cy_comb;            // 组合逻辑 cell_y

// 采样命中判断：根据降采样因子，只有 (xaddr % ds_factor == 0) && (yaddr % ds_factor == 0) 时采样
// ds_factor=4  → xaddr[1:0]==0, yaddr[1:0]==0
// ds_factor=8  → xaddr[2:0]==0, yaddr[2:0]==0
// ds_factor=16 → xaddr[3:0]==0, yaddr[3:0]==0
always @(*) begin
    case (ds_factor)
        4'd4: begin
            sample_hit = cam_valid && (cam_xaddr[1:0] == 2'b00) && (cam_yaddr[1:0] == 2'b00);
            cx_comb    = cam_xaddr[11:2];  // /4 → 320列 → 取低8位足够（1280/4=320）
            cy_comb    = cam_yaddr[10:2];  // /4 → 180行 → 取低7位：128足够？180需要8位
            // 160x90 是 ds=8 的网格，ds=4 时是 320x180
            // 但本设计固定以 160x90 为基准，ds_factor 只影响采样密度
            // 实际 cell 网格固定 160x90，ds_factor=4/8/16 对应步长不同
            // 为统一接口，cell_x/cell_y 始终按 160x90 网格编址
            // ds=4:  xaddr/4 → 但只取 0,4,8... → 即每4个cell采1个？不对
            // 重新理解：cell 网格大小 = (PIX_W/DS_FACTOR) x (PIX_H/DS_FACTOR)
            // 默认 DS=8, 1280/8=160, 720/8=90
            // 所以 cell_x = cam_xaddr / DS_FACTOR
            // cell_y = cam_yaddr / DS_FACTOR
            // 采样条件：cam_xaddr % DS_FACTOR == 0 && cam_yaddr % DS_FACTOR == 0
        end
        4'd0: begin // Four-bit register encodes factor16 as0 (same as prior truncated literal).
            sample_hit = cam_valid && (cam_xaddr[3:0] == 4'b0000) && (cam_yaddr[3:0] == 4'b0000);
            cx_comb    = cam_xaddr[11:4];
            cy_comb    = cam_yaddr[10:4];
        end
        default: begin // 默认 ds_factor=8
            sample_hit = cam_valid && (cam_xaddr[2:0] == 3'b000) && (cam_yaddr[2:0] == 3'b000);
            cx_comb    = cam_xaddr[11:3];
            cy_comb    = cam_yaddr[10:3];
        end
    endcase
end

// cell_idx = cell_y * 160 + cell_x
// 160 = 128 + 32 = 2^7 + 2^5
// cell_y * 160 = cell_y * 128 + cell_y * 32 = {cell_y, 7'b0} + {cell_y, 5'b0}
// 但 cell 网格大小随 ds_factor 变化：
// ds=4:  320 x 180 → idx = y*320 + x
// ds=8:  160 x 90  → idx = y*160 + x
// ds=16: 80 x 45   → idx = y*80 + x
// 为简化，固定使用最大网格 320x180 = 57600，需要 16bit 地址
// 但 BRAM 固定 160x90=14400，所以 ds_factor != 8 时需要额外处理
// 简化方案：cell_idx 始终按 160 列宽计算
// ds=4 时只用到其中的偶地址（步长=2）
// ds=16 时密度更高（步长=0.5，实际只用一半空间）
// 更合理的设计：固定 160x90 网格，DS_FACTOR 只影响从原图取哪些像素
// 即 cell_x[7:0] 范围 0~159，cell_y[6:0] 范围 0~89
// cam_xaddr / DS_FACTOR 映射到 cell_x
// 为了 cell_idx 计算统一，固定 idx = cell_y * 160 + cell_x
always @(*) begin
    // 使用统一的 160 列宽计算 idx
    // cell_y 最多 90 行 → 7位足够
    idx_comb = cy_comb * 8'd160 + cx_comb;
end

// 灰度提取：取 RGB565 的绿色通道 [10:5]（6位），低位补 2 个 0 扩展为 8 位
// RGB565: R[15:11], G[10:5], B[4:0]
always @(*) begin
    luma_comb = {cam_data[10:5], 2'b00};
end

// 输出寄存器打拍
always @(posedge clk or posedge rst) begin
    if (rst) begin
        sample_valid <= 1'b0;
        sample_luma  <= 8'd0;
        cell_idx     <= 14'd0;
        cell_x       <= 8'd0;
        cell_y       <= 7'd0;
    end else begin
        sample_valid <= sample_hit;
        if (sample_hit) begin
            sample_luma <= luma_comb;
            cell_idx    <= idx_comb[13:0];
            cell_x      <= cx_comb;
            cell_y      <= cy_comb[6:0];
        end
    end
end

endmodule
