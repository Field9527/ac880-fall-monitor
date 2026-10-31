`timescale 1ns/1ps
// Local AXI-clock outputs; software heartbeat bounds buzzer activity if PS stops.
module home_io #(
    parameter integer CLOCK_HZ=100000000,
    parameter integer TONE_HZ=2500,
    parameter integer WATCHDOG_CYCLES=300000000
)(
    input wire clk, resetn,
    input wire [9:0] control,
    input wire feed, camera_ready,
    input wire switch0,
    output wire mode_switch,
    output wire [7:0] leds,
    output wire beep,
    output wire [31:0] status
);
    localparam integer HALF_PERIOD=CLOCK_HZ/(2*TONE_HZ);
    reg [31:0] watchdog;
    reg [31:0] tone_count;
    reg tone;
    // camera_ready is asynchronous; only the automatic indicator uses it.
    (* ASYNC_REG="TRUE" *) reg [1:0] camera_sync;
    // SW0 先同步到 AXI 时钟域，由软件完成消抖。
    (* ASYNC_REG="TRUE" *) reg [1:0] switch_sync;
    assign mode_switch=switch_sync[1];
    wire active=control[8] && watchdog!=0;
    assign leds=control[9] ? control[7:0] : {7'b0,camera_sync[1]};
    assign beep=active && tone;
    assign status={21'b0,beep,(control[8] && watchdog==0),active,leds};
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            watchdog<=0; tone_count<=0; tone<=0; camera_sync<=0; switch_sync<=0;
        end else begin
            camera_sync<={camera_sync[0],camera_ready};
            switch_sync<={switch_sync[0],switch0};
            if (feed) watchdog<=WATCHDOG_CYCLES;
            else if (watchdog!=0) watchdog<=watchdog-1;
            if (!active) begin tone_count<=0; tone<=0; end
            else if (tone_count==HALF_PERIOD-1) begin tone_count<=0; tone<=~tone; end
            else tone_count<=tone_count+1;
        end
    end
endmodule
