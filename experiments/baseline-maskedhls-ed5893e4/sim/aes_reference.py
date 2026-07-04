"""
Golden oracle for the AES DOM-masked Sbox correctness check.

`fixtures/aes_sbox_reference_xor_table.txt` is copied verbatim from
`output_original.txt` in the upstream nilotpolas/MaskedHLS repo
(`5. FunctionalCorrectness/AES/`) -- it was produced by the MaskedHLS authors'
own iverilog simulation of this exact masked-Sbox netlist (fixed r0..r35 and
dec_* constants, sweeping all input-share combinations) and records, for
every possible (t0 xor t1) plaintext byte, the correct AES S-box output byte
(y0 xor y1). All 256 plaintext values are present and every duplicate row is
internally consistent (verified in aggregate_report step), which is itself
already a mask-invariance sanity signal from the original authors' own run.

We re-derive the same 256-entry table by driving our own gate-level
simulation with the SAME fixed r0..r35 / dec_* constants used in the
upstream `aes_tb.v` testbench, then diff bit-for-bit against this file.
"""
import os

_FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures",
                         "aes_sbox_reference_xor_table.txt")

# Fixed random-mask byte values from the upstream aes_tb.v testbench (used only
# for this deterministic correctness check, NOT for the TVLA security traces,
# where r0..r35 are drawn fresh per trace).
REFERENCE_R = {
    "r0": 183, "r1": 183, "r2": 30, "r3": 196, "r4": 16, "r5": 36, "r6": 85,
    "r7": 54, "r8": 18, "r9": 238, "r10": 202, "r11": 134, "r12": 244,
    "r13": 252, "r14": 204, "r15": 111, "r16": 123, "r17": 144, "r18": 29,
    "r19": 135, "r20": 219, "r21": 149, "r22": 181, "r23": 97, "r24": 250,
    "r25": 25, "r26": 11, "r27": 223, "r28": 37, "r29": 16, "r30": 106,
    "r31": 73, "r32": 137, "r33": 232, "r34": 187, "r35": 172,
}
REFERENCE_DEC = {
    "dec_0": 0, "dec_1": 1, "dec_255": 255, "dec_169": 169, "dec_129": 129,
    "dec_9": 9, "dec_72": 72, "dec_242": 242, "dec_243": 243, "dec_152": 152,
    "dec_240": 240, "dec_4": 4, "dec_15": 15, "dec_12": 12, "dec_2": 2,
    "dec_3": 3, "dec_16": 16, "dec_36": 36, "dec_220": 220, "dec_11": 11,
    "dec_158": 158, "dec_45": 45, "dec_88": 88, "dec_99": 99,
}


def load_reference_table():
    table = {}
    with open(_FIXTURE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split(",")
            a, b = int(a), int(b)
            if a in table and table[a] != b:
                raise ValueError(f"inconsistent reference table at xor={a}")
            table[a] = b
    if len(table) != 256:
        raise ValueError(f"expected 256 entries, got {len(table)}")
    return table
