"""
Bit-exact Python transliteration of the DOM-AND masked PRESENT S-box C source
that MaskedHLS's HLS pipeline compiled into RTL (see
`5. FunctionalCorrectness/PRESENT/present_domand.c` in the upstream
nilotpolas/MaskedHLS repo -- reproduced logic only, no bytes copied verbatim
in a build sense; every statement below maps 1:1 to a line of that C source).

Used as the formal-equivalence oracle: netlist_sim.py's simulation of the
*synthesized gate-level netlist* must match this function bit-for-bit for the
RTL to be trusted as "the compiler did not silently break the design" (the
correctness check flagged as missing by the automated methodology review).
"""


def domand(a0, a1, b0, b1, z):
    p2 = a0 & b1
    i1 = p2 ^ z
    p3 = a1 & b0
    i2 = p3 ^ z
    p1 = a0 & b0
    p4 = a1 & b1
    y0 = i1 ^ p1
    y1 = i2 ^ p4
    return y0, y1


def sbox(x0_0, x1_0, x2_0, x3_0, x0_1, x1_1, x2_1, x3_1, r):
    L0_0 = x1_0 ^ x2_0
    L1_0 = x0_0 ^ x1_0
    L8_0 = x2_0 ^ x0_0
    L5_0 = x0_0 ^ x3_0
    L0_1 = x1_1 ^ x2_1
    L1_1 = x0_1 ^ x1_1
    L8_1 = x2_1 ^ x0_1
    L5_1 = x0_1 ^ x3_1

    Q0_0 = 1 - L0_0
    Q1_0 = 1 - L1_0
    Q3_0 = 1 - x3_0
    Q4_0 = 1 - x2_0
    Q0_1 = 1 - L0_1
    Q1_1 = 1 - L1_1
    Q3_1 = 1 - x3_1
    Q4_1 = 1 - x2_1

    L2_0 = Q1_0 ^ x2_0
    L3_0 = Q0_0 ^ x3_0
    L2_1 = Q1_1 ^ x2_1
    L3_1 = Q0_1 ^ x3_1

    T0_0, T0_1 = domand(Q0_0, Q0_1, Q1_0, Q1_1, r)

    L10_0 = 1 - L2_0
    L10_1 = 1 - L2_1

    T2_0, T2_1 = domand(x1_0, x1_1, Q4_0, Q4_1, r)
    Q2_0 = T0_0 ^ L2_0
    L4_0 = T0_0 ^ T2_0
    Q7_0 = T0_0 ^ L5_0
    Q6_0 = L4_0 ^ L3_0
    Q2_1 = T0_1 ^ L2_1
    L4_1 = T0_1 ^ T2_1
    Q7_1 = T0_1 ^ L5_1
    Q6_1 = L4_1 ^ L3_1

    T1_0, T1_1 = domand(Q2_0, Q2_1, Q3_0, Q3_1, r)
    T3_0, T3_1 = domand(Q6_0, Q6_1, Q7_0, Q7_1, r)

    L7_0 = T0_0 ^ T1_0
    L11_0 = T1_0 ^ L10_0
    L7_1 = T0_1 ^ T1_1
    L11_1 = T1_1 ^ L10_1

    Y0_01 = L7_0 ^ T2_0
    Y1_01 = L8_0 ^ T3_0
    Y0_11 = L7_1 ^ T2_1
    Y1_11 = L8_1 ^ T3_1

    Y0_0 = x3_0 ^ Y0_01
    Y1_0 = L7_0 ^ Y1_01
    Y2_0 = L11_0 ^ T2_0
    Y3_0 = T2_0 ^ L5_0
    Y0_1 = x3_1 ^ Y0_11
    Y1_1 = L7_1 ^ Y1_11
    Y2_1 = L11_1 ^ T2_1
    Y3_1 = T2_1 ^ L5_1
    return Y0_0, Y1_0, Y2_0, Y3_0, Y0_1, Y1_1, Y2_1, Y3_1


def sbox_vec(x0_0, x1_0, x2_0, x3_0, x0_1, x1_1, x2_1, x3_1, r):
    """Same function, vectorized over numpy integer/bool arrays."""
    return sbox(x0_0, x1_0, x2_0, x3_0, x0_1, x1_1, x2_1, x3_1, r)
