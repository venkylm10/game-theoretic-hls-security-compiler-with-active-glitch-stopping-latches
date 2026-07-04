// Trichina (2003) register-based masked AND gadget, bitwise, width-parameterized.
// a = a0 ^ a1, b = b0 ^ b1 (2-share Boolean masking). r is fresh random mask, one
// bit per output bit, must be independent across gadget instances and cycles.
// The combinational cross terms are latched in ONE register stage before being
// exposed on z0/z1, which is the "register barrier" naive manual masking relies
// on to stop glitches recombining the shares (cf. MaskedHLS's register-only
// barrier). z0 ^ z1 == a & b (bitwise), proven separately by formal equivalence
// checking against the unmasked baseline.
module masked_and_gadget #(parameter W = 8) (
    input                  clk,
    input                  rst,
    input      [W-1:0]     a0, a1,
    input      [W-1:0]     b0, b1,
    input      [W-1:0]     r,      // fresh randomness, one bit per lane
    output reg [W-1:0]     z0, z1
);
    wire [W-1:0] u0 = (a0 & b0) ^ r;
    wire [W-1:0] u1 = (a1 & b1) ^ (r ^ (a0 & b1) ^ (a1 & b0));

    always @(posedge clk) begin
        if (rst) begin
            z0 <= {W{1'b0}};
            z1 <= {W{1'b0}};
        end else begin
            z0 <= u0;
            z1 <= u1;
        end
    end
endmodule
