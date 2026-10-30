/////////////////////////////////////////////////////////////////////////////////
// Module Name   : motion_framestore
// Description   : 运动检测帧存储器
//                 - 双端口 BRAM，ping-pong 双缓冲
//                 - 每页 160x90 = 14400 个 8bit 单元
//                 - 端口A：读参考帧（ref_luma）
//                 - 端口B：写当前帧 / 更新背景模型
//                 - 支持 MODE=0 帧差模式 和 MODE=1 背景减除模式
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

`timescale 1ns / 1ps

module motion_framestore(
    clk,
    rst,
    we,
    addr_a,
    addr_b,
    din,
    dout_a,
    frame_toggle,
    mode,
    bg_alpha,
    clear_bg
);

input               clk;                // 像素时钟 pclk
input               rst;                // 高有效复位
input               we;                 // 写使能（端口B）
input   [13:0]      addr_a;             // 读地址（端口A，参考帧地址）
input   [13:0]      addr_b;             // 写地址（端口B，当前帧地址）
input   [7:0]       din;                // 写入数据（当前亮度）
output  [7:0]       dout_a;             // 读数据（参考亮度）
input               frame_toggle;       // 帧切换脉冲（切换 ping-pong 页）
input               mode;               // 0=帧差模式, 1=背景减除模式
input   [3:0]       bg_alpha;           // 背景更新 alpha 系数（移位位数）
input               clear_bg;           // 清零背景页（用当前帧填充）

// 参数
localparam DEPTH = 14400;               // 每页深度：160*90=14400
localparam DEPTH_BIT = 14;              // 14bit 地址

// 页选择寄存器
reg                 page_sel;           // 0: page0=ref, page1=cur; 1: page1=ref, page0=cur

// 输入打拍寄存器
reg                 we_r;
reg  [13:0]         addr_a_r;
reg  [13:0]         addr_b_r;
reg  [7:0]          din_r;

// BRAM 端口地址（含页偏移）
reg  [14:0]         bram_addr_a;        // 15bit: 页(1bit) + 地址(14bit)
reg  [14:0]         bram_addr_b;
reg  [7:0]          bram_din_b;
reg                 bram_we_b;

// 背景模式计算信号
wire [7:0]          ref_luma;           // 参考亮度（来自 BRAM 端口A）
wire [7:0]          cur_luma;           // 当前亮度（打拍后的 din）
reg  [7:0]          new_bg;             // 计算后的新背景值
reg  [8:0]          diff;               // cur - ref（9位有符号运算用）

// 帧切换：帧差模式下切换 ping-pong 页
always @(posedge clk or posedge rst) begin
    if (rst) begin
        page_sel <= 1'b0;
    end else if (frame_toggle && (mode == 1'b0)) begin
        page_sel <= ~page_sel;
    end
end

// 输入打拍（匹配 BRAM 读延迟）
always @(posedge clk or posedge rst) begin
    if (rst) begin
        we_r     <= 1'b0;
        addr_a_r <= 14'd0;
        addr_b_r <= 14'd0;
        din_r    <= 8'd0;
    end else begin
        we_r     <= we;
        addr_a_r <= addr_a;
        addr_b_r <= addr_b;
        din_r    <= din;
    end
end

// 端口A 读地址：读参考页
// 页偏移：page_sel==0 → ref=page0(0), page_sel==1 → ref=page1(DEPTH)
always @(*) begin
    if (page_sel == 1'b0)
        bram_addr_a = {1'b0, addr_a};       // page0 是参考页
    else
        bram_addr_a = {1'b1, addr_a};       // page1 是参考页
end

// 背景更新计算：new_bg = bg + (cur - bg) >> alpha
// 使用组合逻辑计算，然后打拍写入
assign ref_luma = dout_a;
assign cur_luma = din_r;

always @(*) begin
    if (cur_luma >= ref_luma) begin
        // cur > ref: new_bg = ref + (cur - ref) >> alpha
        diff = cur_luma - ref_luma;
        new_bg = ref_luma + (diff >> bg_alpha);
    end else begin
        // cur < ref: new_bg = ref - (ref - cur) >> alpha
        diff = ref_luma - cur_luma;
        new_bg = ref_luma - (diff >> bg_alpha);
    end
end

// 端口B 写控制
// MODE=0（帧差）：写当前页，数据=din_r，写使能=we_r
// MODE=1（背景）：写参考页（更新背景），数据=new_bg，写使能=we_r（延迟1拍，与读对齐）
// 注意：背景模式下 we_r 已经是打拍后的信号，与 ref_luma 对齐
always @(*) begin
    if (mode == 1'b0) begin
        // 帧差模式：写入当前页（与参考页相反）
        if (page_sel == 1'b0)
            bram_addr_b = {1'b1, addr_b};  // page1 是当前页
        else
            bram_addr_b = {1'b0, addr_b};  // page0 是当前页
        bram_din_b = din;                   // 直接用输入 din，不打拍（写在同一周期）
        bram_we_b  = we;
    end else begin
        // 背景减除模式：写回参考页（更新背景模型）
        if (page_sel == 1'b0)
            bram_addr_b = {1'b0, addr_b_r}; // 写回 page0（参考页）
        else
            bram_addr_b = {1'b1, addr_b_r}; // 写回 page1（参考页）
        bram_din_b = new_bg;                // 更新后的背景值
        bram_we_b  = we_r;                  // 延迟1拍的写使能
    end
end

// ========== 推断双端口 BRAM (2 x 14400 x 8bit) ==========
// 使用寄存器数组推断，Xilinx 综合器会识别为 BRAM
reg [7:0] mem [0:2*DEPTH-1];

// 初始化（仿真用）
integer i;
initial begin
    for (i = 0; i < 2*DEPTH; i = i + 1)
        mem[i] = 8'd0;
end

// 端口A：读
reg [7:0] dout_a_reg;
always @(posedge clk) begin
    dout_a_reg <= mem[bram_addr_a];
end
assign dout_a = dout_a_reg;

// 端口B：写
always @(posedge clk) begin
    if (bram_we_b) begin
        mem[bram_addr_b] <= bram_din_b;
    end
end

endmodule
