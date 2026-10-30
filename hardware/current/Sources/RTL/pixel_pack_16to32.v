`timescale 1ns / 1ps
/////////////////////////////////////////////////////////////////////////////////
// Module Name   : pixel_pack_16to32
// Description   : 将16bit RGB565像素流打包成32bit（2像素/拍）AXI-Stream格式
//                 偶数像素存低16位，奇数像素拼高16位一起输出
//                 一行1280像素（偶数），每行刚好640拍，tlast对齐
//
//                 【v2 修复】原版本在奇像素到达时无条件执行
//                   m_axis_tdata <= {...}; m_axis_tvalid <= 1'b1;
//                 若上一拍输出尚未被下游取走 (m_axis_tready=0), 该拍数据会被
//                 直接覆盖丢失。VDMA 写 DDR 时周期性反压, 于是每行随机丢拍,
//                 行长度不一致 → VDMA 报 SOFLate/EOLLate, 帧永不完成。
//                 v2 增加 out_free 判据, 输出未取走时不再接收奇像素。
//
// Dependencies  :
/////////////////////////////////////////////////////////////////////////////////

module pixel_pack_16to32(
  input               aclk,
  input               aresetn,

  // Slave AXI-Stream: 16bit 输入 (cam_pclk 域)
  input  [15:0]       s_axis_tdata,
  input               s_axis_tvalid,
  input               s_axis_tuser,   // SOF (Start of Frame)
  input               s_axis_tlast,   // EOL (End of Line)
  output              s_axis_tready,

  // Master AXI-Stream: 32bit 输出
  output reg [31:0]   m_axis_tdata,
  output reg          m_axis_tvalid,
  output reg          m_axis_tuser,   // SOF
  output reg          m_axis_tlast,   // EOL
  input               m_axis_tready
);

  reg         pix_phase;      // 0: 偶像素（第0/2/4...个），1: 奇像素（第1/3/5...个）
  reg [15:0]  pix_lo;         // 暂存偶像素（低16位）
  reg         sof_hold;       // 暂存SOF标记，等到奇像素一起输出
  reg         eol_hold;       // 暂存EOL标记，等到奇像素一起输出

  wire        handshake;
  assign handshake = m_axis_tvalid & m_axis_tready;

  // 输出寄存器可装入新数据的条件：本拍被读走，或者本来就是空的
  wire out_free = m_axis_tready | ~m_axis_tvalid;

  // 输入 tready:
  //   偶像素只进暂存寄存器, 永远可收;
  //   奇像素会产生一拍输出, 只有输出寄存器空闲时才可收(否则会覆盖未取走的数据)
  assign s_axis_tready = ~pix_phase | out_free;

  // 像素相位计数 & 数据打包
  always @(posedge aclk or negedge aresetn) begin
    if (!aresetn) begin
      pix_phase    <= 1'b0;
      pix_lo       <= 16'd0;
      sof_hold     <= 1'b0;
      eol_hold     <= 1'b0;
      m_axis_tdata <= 32'd0;
      m_axis_tvalid<= 1'b0;
      m_axis_tuser <= 1'b0;
      m_axis_tlast <= 1'b0;
    end else begin
      // 输出握手成功，清除有效位
      if (handshake) begin
        m_axis_tvalid <= 1'b0;
        m_axis_tuser  <= 1'b0;
        m_axis_tlast  <= 1'b0;
      end

      // 输入握手：s_axis_tvalid & s_axis_tready
      if (s_axis_tvalid & s_axis_tready) begin
        if (!pix_phase) begin
          // 偶像素：暂存低16位和标记
          pix_lo   <= s_axis_tdata;
          sof_hold <= s_axis_tuser;
          eol_hold <= s_axis_tlast;
          pix_phase<= 1'b1;
        end else begin
          // 奇像素：拼上高16位，输出32bit
          // 此处 out_free 必然为 1 (tready 已保证), 保留判据以防万一
          pix_phase     <= 1'b0;
          sof_hold      <= 1'b0;
          eol_hold      <= 1'b0;
          if (out_free) begin
            m_axis_tdata  <= {s_axis_tdata, pix_lo};
            m_axis_tuser  <= sof_hold | s_axis_tuser;
            m_axis_tlast  <= eol_hold | s_axis_tlast;
            m_axis_tvalid <= 1'b1;
          end
        end
      end
    end
  end

endmodule
