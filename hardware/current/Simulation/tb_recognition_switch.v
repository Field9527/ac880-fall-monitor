`timescale 1ns/1ps
// 验证同步、SW0 只读寄存器和 AXI 读反压。
module tb_recognition_switch;
    reg clk=0, resetn=0, switch0=0;
    always #5 clk=~clk;
    wire mode_switch;
    wire [9:0] home_ctrl;
    wire home_feed;
    wire [31:0] home_status;
    home_io io (
        .clk(clk), .resetn(resetn), .control(home_ctrl),
        .feed(home_feed), .camera_ready(1'b0), .switch0(switch0),
        .mode_switch(mode_switch), .status(home_status)
    );
    reg [11:0] address=0;
    reg arvalid=0, rready=0;
    wire arready, rvalid;
    wire [31:0] rdata;
    wire [1:0] rresp;
    motion_regfile regs (
        .s_axi_aclk(clk), .s_axi_aresetn(resetn),
        .home_ctrl(home_ctrl), .home_feed(home_feed), .home_status(home_status),
        .recognition_mode_sw(mode_switch),
        .s_axi_awaddr(12'd0), .s_axi_awvalid(1'b0),
        .s_axi_wdata(32'd0), .s_axi_wstrb(4'd0), .s_axi_wvalid(1'b0),
        .s_axi_bready(1'b1), .s_axi_araddr(address), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rvalid(rvalid),
        .s_axi_rready(rready), .s_axi_rresp(rresp), .pclk(clk), .prst(!resetn)
    );
    task read_check;
        input [11:0] addr;
        input [31:0] expected;
        begin
            @(negedge clk); address=addr; arvalid=1;
            while (!arready) @(negedge clk);
            @(negedge clk); arvalid=0;
            while (!rvalid) @(negedge clk);
            repeat (3) begin
                if (!rvalid || rresp!==0 || rdata!==expected)
                    $fatal(1,"AXI read mismatch addr=%h data=%h expected=%h",addr,rdata,expected);
                @(negedge clk);
            end
            rready=1;
            @(negedge clk); rready=0;
        end
    endtask
    initial begin
        #23; resetn=1;
        repeat (4) @(negedge clk);
        read_check(12'h110,32'h53574D31);
        read_check(12'h114,32'd0);
        #2; switch0=1;
        @(posedge clk); #1;
        if (mode_switch!==0) $fatal(1,"SW0 skipped first synchronizer stage");
        @(posedge clk); #1;
        if (mode_switch!==1) $fatal(1,"SW0 did not propagate through two stages");
        read_check(12'h114,32'd1);
        #2; switch0=0;
        repeat (3) @(negedge clk);
        read_check(12'h114,32'd0);
        read_check(12'h100,32'h484F4D31);
        read_check(12'h10c,32'd2500);
        $display("TEST_PASS recognition_switch");
        $finish;
    end
    initial begin
        #10000; $fatal(1,"Watchdog timeout");
    end
endmodule
