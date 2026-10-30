`timescale 1ns / 1ps
/////////////////////////////////////////////////////////////////////////////////
// Module Name   : rgb565_to_rgb888
// Description   : 将 16bit RGB565 转换成 24bit RGB888
//                 R = R[4:0] << 3  (低位补0)
//                 G = G[5:0] << 2  (低位补0)
//                 B = B[4:0] << 3  (低位补0)
//                 数据通路为组合逻辑输出，以配合显示驱动时序
//
// Dependencies  :
//
/////////////////////////////////////////////////////////////////////////////////

module rgb565_to_rgb888(
  input               clk,
  input               rst_n,

  input  [15:0]       pixel_data,
  input               pixel_valid,

  output [23:0]       rgb888_data,
  output              rgb888_valid
);

  // RGB565: RRRRR GGGGGG BBBBB (bit 15:11, 10:5, 4:0)
  // RGB888: RRRRR000 GGGGGG00 BBBBB000
  assign rgb888_data[23:16] = {pixel_data[15:11], 3'b000};   // R
  assign rgb888_data[15:8]  = {pixel_data[10:5],  2'b00};    // G
  assign rgb888_data[7:0]   = {pixel_data[4:0],   3'b000};   // B

  assign rgb888_valid = pixel_valid;

endmodule

// 将 AXI 视频帧头对齐到显示器第一有效像素；失步时丢弃余帧并等待下一 SOF。
// 不得在屏幕中途直接接收 SOF，否则帧尾会绕到下一屏顶部。
module axis_video_sync(
  input wire clk,
  input wire rst,
  input wire [15:0] s_data,
  input wire s_valid,
  input wire s_user,
  input wire s_last,
  output wire s_ready,
  input wire request,
  input wire first_pixel,
  input wire last_pixel,
  output wire [15:0] pixel,
  output wire pixel_valid
);
  reg locked;
  wire sof_ok = first_pixel ? s_user : (locked && !s_user);
  assign pixel_valid = request && s_valid && sof_ok && (s_last == last_pixel);
  assign pixel = pixel_valid ? s_data : 16'd0;
  // 等待 SOF 时可快速丢弃残帧，SOF 必须保持到屏幕帧头才能握手。
  assign s_ready = !rst && (locked ? (request && (!s_user || first_pixel)) :
                           (s_valid && (!s_user || first_pixel)));
  always @(posedge clk) begin
    if (rst)
      locked <= 1'b0;
    else if (request) begin
      if (first_pixel)
        locked <= pixel_valid;
      else if (locked && !pixel_valid)
        locked <= 1'b0;
    end
  end
endmodule
