`timescale 1ns / 1ps
/////////////////////////////////////////////////////////////////////////////////
// Module Name   : axis_broadcast
// Description   : AXI-Stream 广播器（1进2出）
//                 把相机流同时送给 VDMA S2MM 和 motion_detect（M3用）
//                 简化版：m_axis_0_tready 是主控，m_axis_1 旁路（接1'b1也不影响）
//                 两个从端口的 tready 都满足时才向前握手
//                 用一个深度为1的寄存器缓存实现
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

module axis_broadcast #(
  parameter DATA_WIDTH = 32
)(
  input                     aclk,
  input                     aresetn,

  // Slave AXI-Stream: 输入
  input  [DATA_WIDTH-1:0]   s_axis_tdata,
  input                     s_axis_tvalid,
  input                     s_axis_tuser,
  input                     s_axis_tlast,
  output reg                s_axis_tready,

  // Master AXI-Stream 0: 给 VDMA S2MM
  output [DATA_WIDTH-1:0]   m_axis_0_tdata,
  output reg                m_axis_0_tvalid,
  output                    m_axis_0_tuser,
  output                    m_axis_0_tlast,
  input                     m_axis_0_tready,

  // Master AXI-Stream 1: 给 motion_detect 检测引擎（M1先悬空/接1'b1）
  output [DATA_WIDTH-1:0]   m_axis_1_tdata,
  output reg                m_axis_1_tvalid,
  output                    m_axis_1_tuser,
  output                    m_axis_1_tlast,
  input                     m_axis_1_tready
);

  // 缓存寄存器：当两个输出端口步调不一致时暂存一拍
  reg [DATA_WIDTH-1:0]  buf_tdata;
  reg                   buf_tuser;
  reg                   buf_tlast;
  reg                   buf_valid;

  // 两个输出端口都准备好的标志
  wire both_ready;
  assign both_ready = m_axis_0_tready & m_axis_1_tready;

  // 数据通路：缓存有效时走缓存，否则直通
  assign m_axis_0_tdata = buf_valid ? buf_tdata : s_axis_tdata;
  assign m_axis_0_tuser = buf_valid ? buf_tuser : s_axis_tuser;
  assign m_axis_0_tlast = buf_valid ? buf_tlast : s_axis_tlast;

  assign m_axis_1_tdata = buf_valid ? buf_tdata : s_axis_tdata;
  assign m_axis_1_tuser = buf_valid ? buf_tuser : s_axis_tuser;
  assign m_axis_1_tlast = buf_valid ? buf_tlast : s_axis_tlast;

  // 输入 tready：没有缓存数据时可以接收
  always @(posedge aclk or negedge aresetn) begin
    if (!aresetn)
      s_axis_tready <= 1'b1;
    else if (buf_valid & both_ready)
      s_axis_tready <= 1'b1;   // 缓存被读走，可以接收新数据
    else if (!buf_valid & s_axis_tvalid & ~both_ready)
      s_axis_tready <= 1'b0;   // 数据来了但下游没全准备好，需要缓存，先反压
    else
      s_axis_tready <= ~buf_valid;
  end

  // 输出 tvalid 和缓存管理
  always @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      buf_tdata    <= {DATA_WIDTH{1'b0}};
      buf_tuser    <= 1'b0;
      buf_tlast    <= 1'b0;
      buf_valid    <= 1'b0;
      m_axis_0_tvalid <= 1'b0;
      m_axis_1_tvalid <= 1'b0;
    end else begin
      if (both_ready) begin
        // 两个下游都准备好
        if (buf_valid) begin
          // 优先把缓存数据送出
          buf_valid <= 1'b0;
          m_axis_0_tvalid <= 1'b1;
          m_axis_1_tvalid <= 1'b1;
        end else if (s_axis_tvalid) begin
          // 直通
          m_axis_0_tvalid <= 1'b1;
          m_axis_1_tvalid <= 1'b1;
        end else begin
          m_axis_0_tvalid <= 1'b0;
          m_axis_1_tvalid <= 1'b0;
        end
      end else begin
        // 下游没全准备好
        if (s_axis_tvalid & s_axis_tready & !buf_valid) begin
          // 新数据到达，存入缓存
          buf_tdata <= s_axis_tdata;
          buf_tuser <= s_axis_tuser;
          buf_tlast <= s_axis_tlast;
          buf_valid <= 1'b1;
        end
        // 输出端口保持各自的状态（谁ready谁就拉低valid）
        if (m_axis_0_tready)
          m_axis_0_tvalid <= 1'b0;
        if (m_axis_1_tready)
          m_axis_1_tvalid <= 1'b0;
      end
    end
  end

endmodule
