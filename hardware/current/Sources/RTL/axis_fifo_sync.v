`timescale 1ns / 1ps
/////////////////////////////////////////////////////////////////////////////////
// Module Name   : axis_fifo_sync
// Description   : 同步(单时钟) FWFT FIFO, 用于吸收 AXI VDMA S2MM 的周期性反压。
//
//                 背景: VDMA 写 DDR 时会周期性拉低 s_axis_s2mm_tready。
//                 摄像头 DVP 侧没有任何反压机制(像素流不会等待), 因此必须在
//                 数据包与 VDMA 之间插入弹性缓冲, 否则反压期间像素被丢弃,
//                 行长度不一致 → VDMA 报 SOFLate/EOLLate + ErrInternal,
//                 帧永不完成, DDR 帧缓冲全零。
//
//                 采用 BRAM(ram_style=block) + 1 拍读延迟的 FWFT 结构,
//                 支持背靠背连续读出(无气泡)。
//
// Dependencies  :
/////////////////////////////////////////////////////////////////////////////////

module axis_fifo_sync #(
  parameter DATA_WIDTH = 34,
  parameter ADDR_WIDTH = 10                 // 深度 = 2^ADDR_WIDTH
)(
  input  wire                      clk,
  input  wire                      rst,      // 同步高有效复位

  // 写入侧
  input  wire [DATA_WIDTH-1:0]     din,
  input  wire                      wr_en,
  output wire                      full,

  // 读出侧 (FWFT: valid 与数据同时有效)
  output wire [DATA_WIDTH-1:0]     dout,
  output wire                      valid,
  input  wire                      rd_en
);

  localparam DEPTH = 1 << ADDR_WIDTH;

  (* ram_style = "block" *) reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];

  reg [ADDR_WIDTH:0] wr_ptr;    // 写指针(多一位用于满判断)
  reg [ADDR_WIDTH:0] rd_ptr;    // 读指针: 指向下一个待取出的字
  reg [DATA_WIDTH-1:0] dout_reg;
  reg                  dout_valid;

  // ---------------- 状态 ----------------
  // full: 写指针比读指针多一圈且低位地址相同
  assign full = (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]) &&
                (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);

  // empty: 存储器中没有待取出的字 (注意: dout_reg 中的数据不算在内)
  wire [ADDR_WIDTH:0] cnt = wr_ptr - rd_ptr;
  wire empty = (cnt == {ADDR_WIDTH+1{1'b0}});

  // 取出下一个字的条件: 有数据 且 (输出寄存器空 或 本拍正被读走)
  wire rd_advance = ~empty & (~dout_valid | rd_en);

  // ---------------- 写 ----------------
  always @(posedge clk) begin
    if (rst) begin
      wr_ptr <= 0;
    end else if (wr_en && !full) begin
      mem[wr_ptr[ADDR_WIDTH-1:0]] <= din;
      wr_ptr <= wr_ptr + 1'b1;
    end
  end

  // ---------------- 读 ----------------
  always @(posedge clk) begin
    if (rst) begin
      rd_ptr      <= 0;
      dout_valid  <= 1'b0;
      dout_reg    <= {DATA_WIDTH{1'b0}};
    end else begin
      if (rd_advance) begin
        dout_reg   <= mem[rd_ptr[ADDR_WIDTH-1:0]];   // BRAM 输出寄存, 1 拍延迟
        dout_valid <= 1'b1;
        rd_ptr     <= rd_ptr + 1'b1;
      end else if (rd_en) begin
        dout_valid <= 1'b0;
      end
    end
  end

  assign dout  = dout_reg;
  assign valid = dout_valid;

endmodule
