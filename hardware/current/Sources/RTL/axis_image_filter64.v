`timescale 1ns/1ps
// One RGB565 frame per AXIS packet, four adjacent pixels per beat.
// Neutral colour mode is a lossless, one-beat-per-clock bypass. Sobel uses
// two packed grayscale row buffers; stalled outputs never advance the window.
module axis_image_filter64 #(
    parameter WIDTH=1280, HEIGHT=720
)(
    input wire clk, input wire resetn,
    input wire [31:0] control,
    input wire [63:0] s_data, input wire [7:0] s_keep,
    input wire s_valid, output wire s_ready, input wire s_last,
    output wire [63:0] m_data, output wire [7:0] m_keep,
    output wire m_valid, input wire m_ready, output wire m_last,
    output reg [31:0] frame_count
);
    localparam WORDS=WIDTH/4;
    localparam LOAD=0, CONVERT=1, EDGE=2, OUT=3, RIGHT=4, BOTTOM=5, MAGNITUDE=6, LUMA=7;
    localparam NEXT_WORD=0, NEXT_RIGHT=1, NEXT_ROW=2, NEXT_BOTTOM=3, NEXT_BLACK=4;
    reg [2:0] state, after_output;
    reg frame_active;
    reg [31:0] latched_control;
    wire [31:0] selected=frame_active ? latched_control : control;
    wire bypass=(state==LOAD && selected==32'h00800000);
    reg [15:0] word_x, row_y, bottom_x;
    reg [63:0] output_word;
    reg output_last, input_last;
    (* use_dsp="yes" *) reg signed [18:0] products[0:11];
    reg [95:0] adjusted_word;
    reg [31:0] gray_word;
    reg [87:0] gradients;
    (* ram_style="block" *) reg [31:0] row_a[0:WORDS-1];
    (* ram_style="block" *) reg [31:0] row_b[0:WORDS-1];
    reg [31:0] read_a, read_b;
    reg [31:0] prev_top, prev_mid, prev_bot;
    reg [7:0] left_top, left_mid, left_bot;
    wire [31:0] current_top=(row_y<2) ? 0 : (row_y[0] ? read_b : read_a);
    wire [31:0] current_mid=(row_y==0) ? 0 : (row_y[0] ? read_a : read_b);
    assign s_ready=resetn && (bypass ? m_ready : state==LOAD);
    assign m_valid=resetn && (bypass ? s_valid : state==OUT);
    assign m_data=bypass ? s_data : output_word;
    assign m_keep=bypass ? s_keep : 8'hff;
    assign m_last=bypass ? s_last : output_last;

    function [7:0] expand5(input [4:0] v);
        expand5={v,v[4:2]};
    endfunction
    function [7:0] expand6(input [5:0] v);
        expand6={v,v[5:4]};
    endfunction
    function signed [8:0] centered(input [7:0] v);
        centered=$signed({1'b0,v})-9'sd128;
    endfunction
    function [7:0] clamp_product(input signed [18:0] v, input signed [7:0] brightness);
        reg signed [12:0] adjusted;
        begin
            adjusted=(v >>> 7)+13'sd128+brightness;
            clamp_product=adjusted<0 ? 0 : adjusted>255 ? 255 : adjusted[7:0];
        end
    endfunction
    function [7:0] luma(input [7:0] r,g,b);
        reg [16:0] total;
        begin total=r*17'd77+g*17'd150+b*17'd29+17'd128; luma=total[15:8]; end
    endfunction
    function [15:0] pack_gray(input [7:0] g);
        pack_gray={g[7:3],g[7:2],g[7:3]};
    endfunction
    function [7:0] lane(input [31:0] v, input integer n);
        lane=(v >> (8*n));
    endfunction
    function [87:0] sobel_gradients(
        input [31:0] top,mid,bot,
        input [7:0] lt,lm,lb,rt,rm,rb,
        input zero_row,zero_left,zero_right
    );
        integer p;
        // Each derivative lies in [-1020,1020]. Bound the arithmetic rather
        // than synthesizing 32-bit integer subtract/absolute-value chains.
        reg signed [10:0] tl,tr,ml,mr,bl,br,tc,bc,gx,gy;
        begin
            sobel_gradients=0;
            for(p=0;p<4;p=p+1) begin
                tl=p==0 ? lt : lane(top,p-1); tr=p==3 ? rt : lane(top,p+1);
                ml=p==0 ? lm : lane(mid,p-1); mr=p==3 ? rm : lane(mid,p+1);
                bl=p==0 ? lb : lane(bot,p-1); br=p==3 ? rb : lane(bot,p+1);
                tc=lane(top,p); bc=lane(bot,p);
                gx=(tr+(mr<<<1)+br)-(tl+(ml<<<1)+bl);
                gy=(bl+(bc<<<1)+br)-(tl+(tc<<<1)+tr);
                if(zero_row || (zero_left && p==0) || (zero_right && p==3)) begin gx=0;gy=0;end
                sobel_gradients[22*p +:11]=gx;
                sobel_gradients[22*p+11 +:11]=gy;
            end
        end
    endfunction
    function [63:0] pack_edges(input [87:0] v);
        integer p;
        reg signed [10:0] gx,gy;
        reg [10:0] ax,ay;
        reg [11:0] sum;
        reg [7:0] value;
        begin
            for(p=0;p<4;p=p+1) begin
                gx=$signed(v[22*p +:11]);gy=$signed(v[22*p+11 +:11]);
                ax=gx<0 ? -gx : gx;ay=gy<0 ? -gy : gy;
                sum={1'b0,ax}+{1'b0,ay};
                value=sum>=12'd1024 ? 8'd255 : sum[9:2];
                pack_edges[16*p +:16]=pack_gray(value);
            end
        end
    endfunction

    // Separate synchronous RAM process keeps both row stores in BRAM.
    always @(posedge clk) begin
        if(resetn && state==LOAD && !bypass && s_valid && s_ready && selected[1:0]==2) begin
            read_a<=row_a[word_x]; read_b<=row_b[word_x];
        end
        if(resetn && state==EDGE) begin
            if(row_y[0]) row_b[word_x]<=gray_word;
            else row_a[word_x]<=gray_word;
        end
    end

    integer i;
    reg [7:0] r,g,b,y;
    always @(posedge clk) begin
        if(!resetn) begin
            state<=LOAD; frame_active<=0; latched_control<=32'h00800000;
            word_x<=0; row_y<=0; bottom_x<=0; frame_count<=0;
            output_word<=0; output_last<=0; input_last<=0; after_output<=NEXT_WORD;
            prev_top<=0;prev_mid<=0;prev_bot<=0;left_top<=0;left_mid<=0;left_bot<=0;
            gray_word<=0;gradients<=0;adjusted_word<=0;
            for(i=0;i<12;i=i+1) products[i]<=0;
        end else begin
            if(s_valid && s_ready && !frame_active) begin
                latched_control<=control; frame_active<=1;
            end
            if(m_valid && m_ready && m_last) begin
                frame_count<=frame_count+1; frame_active<=0;
                word_x<=0;row_y<=0;bottom_x<=0;
            end
            case(state)
            LOAD: if(!bypass && s_valid && s_ready) begin
                for(i=0;i<4;i=i+1) begin
                    products[3*i]<=centered(expand5(s_data[16*i+11 +:5]))*$signed({1'b0,selected[24:16]});
                    products[3*i+1]<=centered(expand6(s_data[16*i+5 +:6]))*$signed({1'b0,selected[24:16]});
                    products[3*i+2]<=centered(expand5(s_data[16*i +:5]))*$signed({1'b0,selected[24:16]});
                end
                input_last<=s_last; state<=CONVERT;
            end
            CONVERT: begin
                for(i=0;i<4;i=i+1) begin
                    r=clamp_product(products[3*i],$signed(latched_control[15:8]));
                    g=clamp_product(products[3*i+1],$signed(latched_control[15:8]));
                    b=clamp_product(products[3*i+2],$signed(latched_control[15:8]));
                    adjusted_word[24*i +:24]<={r,g,b};
                end
                state<=LUMA;
            end
            LUMA: begin
                for(i=0;i<4;i=i+1) begin
                    r=adjusted_word[24*i+16 +:8];
                    g=adjusted_word[24*i+8 +:8];
                    b=adjusted_word[24*i +:8];
                    y=luma(r,g,b);
                    gray_word[8*i +:8]<=y;
                    output_word[16*i +:16]<=latched_control[1:0]==1 ? pack_gray(y) : {r[7:3],g[7:2],b[7:3]};
                end
                if(latched_control[1:0]==2) state<=EDGE;
                else begin output_last<=input_last;after_output<=NEXT_WORD;state<=OUT;end
            end
            EDGE: begin
                prev_top<=current_top;prev_mid<=current_mid;prev_bot<=gray_word;
                left_top<=prev_top[31:24];left_mid<=prev_mid[31:24];left_bot<=prev_bot[31:24];
                if(row_y!=0 && word_x!=0) begin
                    gradients<=sobel_gradients(prev_top,prev_mid,prev_bot,left_top,left_mid,left_bot,
                        current_top[7:0],current_mid[7:0],gray_word[7:0],row_y==1,word_x==1,0);
                    output_last<=0;after_output<=word_x==WORDS-1 ? NEXT_RIGHT : NEXT_WORD;state<=MAGNITUDE;
                end else begin
                    if(word_x==WORDS-1) begin
                        word_x<=0;row_y<=row_y+1;
                        prev_top<=0;prev_mid<=0;prev_bot<=0;left_top<=0;left_mid<=0;left_bot<=0;
                    end else word_x<=word_x+1;
                    state<=LOAD;
                end
            end
            OUT: if(m_ready) begin
                if(output_last) state<=LOAD;
                else case(after_output)
                    NEXT_WORD: begin if(latched_control[1:0]==2) word_x<=word_x+1;state<=LOAD;end
                    NEXT_RIGHT:state<=RIGHT;
                    NEXT_ROW:begin
                        word_x<=0;row_y<=row_y+1;state<=LOAD;
                        prev_top<=0;prev_mid<=0;prev_bot<=0;left_top<=0;left_mid<=0;left_bot<=0;
                    end
                    NEXT_BOTTOM:begin bottom_x<=0;state<=BOTTOM;end
                    NEXT_BLACK:begin bottom_x<=bottom_x+1;state<=BOTTOM;end
                    default:state<=LOAD;
                endcase
            end
            RIGHT: begin
                gradients<=sobel_gradients(prev_top,prev_mid,prev_bot,left_top,left_mid,left_bot,
                    0,0,0,row_y==1,WORDS==1,1);
                output_last<=0;after_output<=row_y==HEIGHT-1 ? NEXT_BOTTOM : NEXT_ROW;state<=MAGNITUDE;
            end
            MAGNITUDE:begin output_word<=pack_edges(gradients);state<=OUT;end
            BOTTOM: begin
                output_word<=0;output_last<=bottom_x==WORDS-1;after_output<=NEXT_BLACK;state<=OUT;
            end
            default:state<=LOAD;
            endcase
        end
    end
endmodule
