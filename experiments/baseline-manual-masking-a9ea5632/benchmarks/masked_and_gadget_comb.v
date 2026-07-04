// Purely combinational form of the Trichina masked-AND gadget (no clk/register).
// Used ONLY for formal equivalence checking: SAT-based equivalence proofs
// operate on combinational netlists, and inserting/removing pipeline
// registers on a feed-forward (loop-free) datapath never changes the function
// it computes -- only when the answer appears. The clocked variant
// (masked_and_gadget.v) is used for real area/register counting; this one is
// used to prove z0^z1 == a&b holds for ALL values of the randomness r.
module masked_and_gadget_comb #(parameter W = 8) (
    input      [W-1:0] a0, a1,
    input      [W-1:0] b0, b1,
    input      [W-1:0] r,
    output     [W-1:0] z0, z1
);
    assign z0 = (a0 & b0) ^ r;
    assign z1 = (a1 & b1) ^ (r ^ (a0 & b1) ^ (a1 & b0));
endmodule
