"""
census_data.py -- the edge-regime design census (author-classified, cut-off Sep 2026).

Each row: (name, bibkey, year, platform, regime, phase, components, impl, flags)
  regime : 'mw' (milliwatt-class silicon, < ~1 W)
           '1w' (single-watt: operator-level units and accelerators for ~1 W integration)
           'fw' (several-watt: board- and SoC-level systems up to ~10 W)
  phase  : 'P' | 'D' | 'P+D' | 'op'  (op = operator-level unit / macro)
  comps  : subset of GEMM, ATT (attention products), KV (KV-cache handling), SM, NORM,
           ACT, ROPE, MOE
  impl   : Si (measured silicon) | SoC (measured on commercial mobile silicon) |
           FPGA | PL (post-layout) | Syn (post-synthesis) | Sim (simulation) | NR
  flags  : 't' = classification from title/abstract only (details not disclosed in
                 the material available to us);
           'r' = regime inferred from energy/token or platform class (power not reported);
           'e' = evaluated on encoder / short-sequence Transformer workloads, not LLM decode.
Excluded by rule (15 Oct 2026): operator-level units prototyped on data-center FPGA cards without a
stated edge target or with reported power above the regime (Kim et al. VCU128, Kang & Wang U55C
[17.7 W], HAAN U280); they remain cited as operator-level evidence in the non-GEMM section.
"""

CENSUS = [
    # ------------------------------------------------------------ several-watt
    ("TeLLMe v1",            "tellme",            2025, "low-power FPGA (KV260)",     "fw", "P+D", ["GEMM", "ROPE", "NORM"], "FPGA", ""),
    ("TeLLMe v2",            "tellme2",           2026, "low-power FPGA (KV260)",     "fw", "P+D", ["GEMM", "ROPE", "NORM"], "FPGA", ""),
    ("PD-Swap",              "zhang2025pdswap",   2025, "low-power FPGA",             "fw", "P+D", ["GEMM", "ATT", "KV"], "FPGA", ""),
    ("Hummingbird",          "hummingbird",       2025, "embedded FPGA (KV260/ZCU104)", "fw", "D", ["GEMM", "ATT", "SM", "NORM", "ACT", "ROPE"], "FPGA", ""),
    ("MEADOW",               "meadow",            2025, "FPGA (ZCU102)",              "fw", "P+D", ["GEMM", "KV"], "FPGA", ""),
    ("Li et al. (DATE)",     "lidate25",          2025, "embedded FPGA (KV260)",      "fw", "D",   ["GEMM", "KV", "ROPE"], "FPGA", ""),
    ("COBRA",                "cobra",             2025, "FPGA (ZCU102/KV260)",        "fw", "P",   ["GEMM", "SM"], "FPGA", "e"),
    ("TENET",                "tenet",             2025, "FPGA + ASIC",                "fw", "P+D", ["GEMM", "ATT"], "FPGA/Syn", ""),
    ("LightMamba",           "lightmamba",        2025, "FPGA (VCK190)",              "fw", "D",   ["GEMM"], "FPGA", ""),
    ("SpecMamba",            "zhong2025specmamba", 2025, "FPGA (VCK190)",             "fw", "D",   ["GEMM"], "FPGA", ""),
    ("SPECTRA",              "tombesi2026spectra", 2026, "tiled FPGA prototype",      "fw", "P+D", ["GEMM"], "FPGA", ""),
    ("Xiang et al.",         "xiang2025qwen",     2025, "embedded FPGA (KV260)",      "fw", "D",   ["GEMM"], "FPGA", ""),
    ("LlamaF",               "llamaf",            2024, "embedded FPGA (ZCU102)",     "fw", "D",   ["GEMM"], "FPGA", ""),
    ("Haris et al.",         "haris_edge",        2024, "embedded FPGA",              "fw", "D",   ["GEMM"], "FPGA", ""),
    ("Edge-MoE (ViT)",       "edgemoe_vit",       2023, "FPGA",                       "fw", "P",   ["MOE"], "FPGA", "e"),
    ("llm.npu",              "llmnpu",            2025, "mobile NPU (SoC)",           "fw", "P",   ["GEMM"], "SoC", ""),
    ("ShadowNPU",            "shadownpu",         2026, "mobile NPU (SoC)",           "fw", "P",   ["ATT"], "SoC", ""),
    ("HeteroInfer",          "heteroinfer",       2025, "mobile SoC (GPU+NPU)",       "fw", "P+D", ["GEMM"], "SoC", ""),
    ("Hao et al.",           "hao2026ttsnpu",     2026, "mobile NPU (SoC)",           "fw", "D",   ["GEMM", "SM"], "SoC", ""),
    ("Huang et al.",         "huang2026kvreuse",  2026, "mobile NPU (SoC)",           "fw", "P+D", ["KV", "ROPE"], "SoC", ""),
    ("EStream",              "zhang2026estream",  2026, "mobile NPU + flash (SoC)",   "fw", "P",   ["GEMM", "MOE"], "SoC", ""),
    ("PowerInfer-2",         "xue2024powerinfer2", 2024, "phone SoC + flash",         "fw", "D",   ["GEMM"], "SoC", ""),
    ("Lever",                "wang2026lever",     2026, "phone SoC + flash",          "fw", "D",   ["GEMM"], "SoC", ""),
    ("EdgeMoE",              "edgemoe",           2023, "mobile SoC",                 "fw", "D",   ["MOE"], "SoC", ""),
    ("FACIL",                 "facil",               2025, "SoC--PIM (on-device)",        "fw", "D",   ["GEMM"], "Sim", ""),
    ("CD-PIM",               "lin2026cdpim",      2026, "LPDDR5-PIM",                 "fw", "D",   ["GEMM", "ATT", "KV"], "Sim", ""),
    ("P3-LLM",               "chen2026p3llm",     2026, "NPU + DRAM-PIM",             "fw", "P+D", ["GEMM", "ATT", "KV"], "Sim", ""),
    ("LP-Spec",              "he2025lpspec",      2025, "LPDDR5-PIM (mobile)",        "fw", "D",   ["GEMM", "ATT"], "Sim", ""),
    ("AHASD",                "ma2026ahasd",       2026, "NPU + LPDDR5-PIM",           "fw", "D",   ["GEMM", "ATT"], "Sim", ""),
    ("CHIME",                "chen2025chime",     2025, "chiplet M3D-DRAM + RRAM",    "fw", "P+D", ["GEMM", "ATT", "KV"], "Sim", ""),
    ("H3D-Transformer",      "h3d",               2024, "3D heterogeneous ASIC",      "fw", "P+D", ["GEMM", "ATT"], "Sim", ""),
    # ------------------------------------------------------------ single-watt
    ("PDAcc",                 "wang2025pdacc",       2025, "4-chiplet ASIC (22 nm)",      "mw", "P+D", ["GEMM", "ATT", "KV"], "Si", ""),
    ("CELLA",                 "wu2025cella",         2025, "digital-CIM ASIC (28 nm)",    "mw", "P+D", ["GEMM", "KV"], "Si", "r"),
    ("RCW-CIM",               "guo2026rcwcim",       2026, "digital-CIM ASIC (22 nm)",    "mw", "P+D", ["GEMM", "SM", "NORM"], "Si", "r"),
    ("RS-LPU",                "dong2026reram",       2026, "ReRAM-on-logic ASIC (55 nm)", "1w", "D",   ["GEMM"], "Si", ""),
    ("Revolver",              "kim2026revolver",     2026, "ASIC (28 nm)",                "1w", "P+D", ["GEMM"], "Si", ""),
    ("LUT-SSM",               "yoo2026lutssm",       2026, "ASIC (28 nm FD-SOI)",         "mw", "P+D", ["GEMM", "ACT"], "Si", ""),
    ("Cao et al. (3D PNM)",   "cao2026pnm",          2026, "3D DRAM--logic PNM (28 nm)",  "fw", "D",   ["GEMM"], "Si", ""),
    ("Spectra",               "mun2026spectra",      2026, "ASIC (28 nm)",                "mw", "P+D", ["GEMM", "KV"], "Si", ""),
    ("TACC",                  "yang2026tacc",        2026, "ASIC (16 nm)",                "mw", "P+D", ["GEMM"], "Si", ""),
    ("Qin et al. (ISSCC)",    "isscc25_llm",         2025, "ASIC (28 nm)",                "mw", "P+D", ["GEMM", "SM", "ACT"], "Si", ""),
    ("HiKV",                 "fang2026hikv",      2026, "ASIC (decode attention)",    "1w", "D",   ["ATT", "KV"], "Syn", ""),
    ("EdgeXpert",             "ha2026edgexpert",     2026, "ASIC (28 nm)",                "1w", "P+D", ["GEMM", "MOE"], "Syn", ""),
    ("BitROM",               "zhang2026bitrom",   2026, "CiROM ASIC (65 nm)",         "1w", "D",   ["GEMM", "KV"], "Sim", ""),
    ("TOM",                   "guan2026tom",         2026, "ternary ROM--SRAM ASIC",      "fw", "P+D", ["GEMM"], "Syn", ""),
    ("Platinum",              "shan2025platinum",    2025, "ASIC (28 nm)",                "fw", "op",  ["GEMM"], "Syn", ""),
    ("PALUTE",               "tian2026palute",    2026, "M3D-DRAM LUT-PIM",           "1w", "P+D", ["GEMM", "ACT"], "Sim", ""),
    ("HARDSEA",              "hardsea",           2024, "hybrid ReRAM/SRAM CIM",      "1w", "P+D", ["ATT"], "NR", ""),
    ("X-Former",             "xformer",           2023, "ReRAM CIM",                  "1w", "P",   ["ATT"], "Sim", "e"),
    ("ReTransformer",        "retransformer",     2020, "ReRAM PIM",                  "1w", "P",   ["GEMM", "ATT"], "Sim", "e"),
    ("ATT (ReRAM)",          "att_reram",         2020, "ReRAM",                      "1w", "P",   ["ATT"], "Sim", "e"),
    ("SOLE",                 "sole",              2023, "ASIC unit (28 nm)",          "1w", "op",  ["SM", "NORM"], "Syn", ""),
    ("NN-LUT",               "nnlut",             2022, "ASIC unit (7 nm)",           "1w", "op",  ["ACT", "SM", "NORM"], "Syn", ""),
    ("PEANO-ViT",            "peanovit",          2024, "FPGA unit (VU9P)",           "1w", "op",  ["ACT", "SM", "NORM"], "FPGA", "e"),
    ("ConSmax",              "consmax",           2024, "ASIC unit (16 nm)",          "1w", "op",  ["SM"], "Syn", ""),
    ("CORDIC RoPE",          "rope_cordic",       2025, "ASIC unit",                  "1w", "op",  ["ROPE"], "Syn", ""),
    ("RoPIM",                "ropim",             2025, "DRAM bank-level PIM",        "1w", "op",  ["ROPE"], "Sim", ""),
    ("FusionCIM",            "xuan2026fusioncim", 2026, "CIM attention engine",       "1w", "op",  ["ATT", "SM"], "Sim", ""),
    ("MIVE",                  "alexandridis2026mive", 2026, "integer vector engine (28 nm)", "mw", "op",  ["SM", "NORM"], "PL", ""),
    ("CAMTA",                "sotoporras2026camta", 2026, "ASIC unit (65 nm) / FPGA",  "1w", "op",  ["ACT", "SM"], "Syn", ""),
    ("LightningRMS",         "sur2026lightningrms", 2026, "FPGA unit (Kintex US+)",   "1w", "op",  ["NORM"], "FPGA", ""),
    # ------------------------------------------------------------ milliwatt
    ("Slim-Llama",           "kim2025slimllama",  2025, "ASIC (28 nm)",               "mw", "P+D", ["GEMM"], "Si", ""),
    ("T-REX",                "moon2025trex",      2025, "ASIC (16 nm)",               "mw", "P+D", ["GEMM", "SM", "NORM", "ACT"], "Si", "e"),
    ("Li et al. (VLSI'26)",   "li2026reasoning",     2026, "ASIC (22 nm)",                "mw", "P+D", ["GEMM", "KV"], "Si", ""),
    ("SMoLPU",                "ha2026smolpu",        2026, "NPU--CIM ASIC (28 nm FD-SOI)", "mw", "P+D", ["GEMM", "MOE"], "Si", ""),
    ("SoulMate",              "hong2026soulmate",    2026, "mobile SoC (28 nm)",          "mw", "P+D", ["GEMM"], "Si", ""),
    ("Liu et al. (ISSCC'26)", "liu2026subspace",     2026, "ASIC (28 nm)",                "mw", "P+D", ["GEMM", "KV", "ROPE", "ACT"], "Si", ""),
    ("Tri-Oracle",            "yoo2026trioracle",    2026, "ASIC (28 nm FD-SOI, VLM)",    "1w", "P+D", ["GEMM", "ATT", "SM"], "Si", ""),
    ("Tao et al. (CICC'26)",  "tao2026slmdecoder",   2026, "ASIC (16 nm)",                "mw", "D",   ["GEMM", "ATT", "KV", "SM", "NORM"], "Si", ""),
    ("EdgeFTX",               "hao2025edgeftx",      2025, "ASIC (22 nm)",                "mw", "P+D", ["GEMM"], "Si", ""),
    ("LLM-CIM",               "wang2025llmcim",      2025, "digital-CIM macro (28 nm)",   "mw", "op",  ["GEMM", "ACT"], "Si", ""),
    ("D2CIM",                 "chen2026d2cim",       2026, "digital-CIM macro (28 nm)",   "mw", "op",  ["ATT", "SM", "KV"], "Si", ""),
    ("TOKCIM",                "li2026tokcim",        2026, "digital CIM (65 nm)",         "mw", "P+D", ["GEMM", "SM"], "Si", ""),
    ("C-Transformer",        "ctransformer",      2024, "ASIC (28 nm)",               "mw", "P+D", ["GEMM"], "Si", ""),
    ("MulTCIM",              "multcim",           2023, "digital CIM (28 nm)",        "mw", "P+D", ["ATT", "SM"], "Si", "e"),
    ("Tambe et al.",         "tambe_isscc",       2023, "ASIC (12 nm)",               "mw", "P",   ["GEMM"], "Si", "e"),
    ("VitaLLM",               "lin2026vitallmchip",  2026, "ASIC (16 nm)",                "mw", "P+D", ["GEMM", "ATT", "KV", "SM", "NORM"], "Si", ""),
    ("Chen et al. (ISLPED)", "chen2025hsa",  2025, "ASIC (28 nm)",               "mw", "P+D", ["GEMM", "ROPE", "NORM"], "PL", ""),
    ("ITA",                  "ita",               2023, "ASIC (22 nm)",               "mw", "op",  ["ATT", "SM"], "PL", "e"),
    ("Choi et al.",          "choi_edge_nongemm", 2026, "ASIC unit (28 nm)",          "mw", "op",  ["SM", "NORM"], "Syn", ""),
    ("SoftEx",               "belano2024softex",  2024, "RISC-V cluster unit (12 nm)", "mw", "op", ["SM", "ACT"], "PL", ""),
    ("VEXP",                  "wang2025vexp",        2025, "RISC-V FPU extension (12 nm)", "mw", "op",  ["SM"], "Syn", ""),
    ("CIMple",               "ahn2025cimple",     2025, "SRAM-CIM unit (28 nm)",      "mw", "op",  ["ATT", "SM"], "PL", ""),
    ("EdgeBERT",             "edgebert_chip",     2021, "ASIC (12 nm)",               "mw", "P",   ["GEMM"], "PL", "e"),
]
