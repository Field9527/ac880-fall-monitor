`timescale 1ns/1ps
// DHT11 open-drain master and threshold light input, all in the AXI clock.
// No CPU bit-banging, interrupts, DMA, or video clock changes are needed.
module environment_sensors #(
    parameter integer CLOCK_HZ=100000000,
    parameter integer STARTUP_US=2000000,
    parameter integer SAMPLE_US=5000000,
    parameter integer START_US=18000,
    parameter integer LIGHT_FILTER_US=10000
)(
    input wire clk, resetn,
    inout wire dht11_data,
    input wire light_do,
    output wire [31:0] status,
    output reg [31:0] payload, sample_seq, age_ms, errors, attempts
);
    localparam integer DIV=CLOCK_HZ/1000000;
    localparam [3:0] WAIT=0, START=1, RELEASE=2, RESPONSE=3,
        RESP_LOW=4, RESP_HIGH=5, BIT_LOW=6, BIT_HIGH=7, CHECK=8;
    reg [31:0] divider, timer, light_count;
    reg [9:0] millisecond;
    reg [3:0] state;
    reg first, drive_low, has_good, last_ok, light_level;
    reg [7:0] error_code;
    reg [5:0] bit_index;
    reg [39:0] shift;
    (* ASYNC_REG="TRUE" *) reg [1:0] dht_sync, light_sync;
    wire tick=(divider==DIV-1);
    wire [7:0] checksum=shift[39:32]+shift[31:24]+shift[23:16]+shift[15:8];
    assign dht11_data=drive_low ? 1'b0 : 1'bz;
    assign status={16'd0,error_code,3'd0,dht_sync[1],light_level,last_ok,(state!=WAIT),has_good};

    task failed;
        input [7:0] reason;
        begin
            errors<=errors+1;error_code<=reason;last_ok<=0;
            drive_low<=0;state<=WAIT;timer<=0;
        end
    endtask

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin dht_sync<=2'b11;light_sync<=2'b11;end
        else begin dht_sync<={dht_sync[0],dht11_data};light_sync<={light_sync[0],light_do};end
    end
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            divider<=0;millisecond<=0;timer<=0;light_count<=0;
            state<=WAIT;first<=1;drive_low<=0;has_good<=0;last_ok<=0;
            light_level<=1;error_code<=0;bit_index<=0;shift<=0;
            payload<=0;sample_seq<=0;age_ms<=32'hffffffff;errors<=0;attempts<=0;
        end else begin
            if (tick) divider<=0;else divider<=divider+1;
            if (tick) begin
                if (millisecond==999) begin
                    millisecond<=0;
                    if (age_ms!=32'hffffffff) age_ms<=age_ms+1;
                end else millisecond<=millisecond+1;
                if (light_sync[1]==light_level) light_count<=0;
                else if (light_count>=LIGHT_FILTER_US-1) begin
                    light_level<=light_sync[1];light_count<=0;
                end else light_count<=light_count+1;
                timer<=timer+1;
                case(state)
                    WAIT: if(timer>=(first ? STARTUP_US : SAMPLE_US)-1) begin
                        first<=0;attempts<=attempts+1;drive_low<=1;
                        timer<=0;state<=START;shift<=0;bit_index<=0;
                    end
                    START: if(timer>=START_US-1) begin
                        drive_low<=0;timer<=0;state<=RELEASE;
                    end
                    // Discard the synchronized copy of our own start pulse.
                    RELEASE: if(timer>=9) begin timer<=0;state<=RESPONSE;end
                    RESPONSE: if(!dht_sync[1]) begin timer<=0;state<=RESP_LOW;end
                        else if(timer>=119) failed(1);
                    RESP_LOW: if(dht_sync[1]) begin
                        if(timer<45 || timer>110) failed(3);
                        else begin timer<=0;state<=RESP_HIGH;end
                    end else if(timer>=119) failed(2);
                    RESP_HIGH: if(!dht_sync[1]) begin
                        if(timer<50 || timer>110) failed(3);
                        else begin timer<=0;state<=BIT_LOW;end
                    end else if(timer>=119) failed(3);
                    BIT_LOW: if(dht_sync[1]) begin
                        if(timer<30 || timer>90) failed(3);
                        else begin timer<=0;state<=BIT_HIGH;end
                    end else if(timer>=99) failed(3);
                    BIT_HIGH: if(!dht_sync[1]) begin
                        if(timer<15 || timer>95) failed(3);
                        else begin
                            shift<={shift[38:0],(timer>=49)};timer<=0;
                            if(bit_index==39) state<=CHECK;
                            else begin bit_index<=bit_index+1;state<=BIT_LOW;end
                        end
                    end else if(timer>=99) failed(3);
                    CHECK: begin
                        if(checksum!=shift[7:0]) failed(4);
                        else if(shift[39:32]>100 || shift[31:24]>9 || shift[23:16]>80 ||
                                shift[14:8]>9 || (shift[39:32]==100 && shift[31:24]!=0)) failed(5);
                        else begin
                            payload<=shift[39:8];sample_seq<=sample_seq+1;age_ms<=0;
                            has_good<=1;last_ok<=1;error_code<=0;timer<=0;state<=WAIT;
                        end
                    end
                    default: failed(3);
                endcase
            end
        end
    end
endmodule
