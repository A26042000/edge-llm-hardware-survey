# Abstract-level decisions for the candidates marked '?' at title level (keys = index in
# cand_list.txt, the first assessment pass), plus the nine pre-2023 candidates (keyed by title).
# Codes as in codes_title.py, plus U = edge regime not established by the source.
ABSTRACT = {
 6:"V",7:"V",8:"V",15:"Y",20:"V",21:"V",23:"C",24:"Y",27:"S",28:"V",29:"V",32:"C",34:"C",35:"V",36:"S",37:"Y",39:"Y",
 40:"Y",42:"S",43:"C",45:"Y",48:"C",51:"U",55:"S",59:"Y",60:"Y",64:"Y",69:"S",70:"Y",71:"S",73:"C",74:"Y",75:"Y",77:"Y",78:"Y",
 89:"S",90:"V",91:"Y",92:"Y",97:"Y",98:"U",99:"Y",101:"U",104:"Y",108:"Y",109:"V",112:"Y",114:"V",115:"V",119:"V",
 120:"Y",121:"Y",122:"Y",124:"V",127:"Y",128:"V",129:"C",131:"Y",132:"Y",134:"V",145:"C",151:"V",152:"Y",153:"Y",154:"Y",
 155:"Y",159:"V",160:"Y",161:"S",168:"Y",170:"U",171:"Y",172:"Y",179:"S",183:"V",185:"Y",186:"Y",187:"S",188:"Y",189:"Y",
 190:"U",191:"Y",192:"Y",193:"C",195:"Y",196:"V",197:"V",198:"V",199:"Y",201:"Y",204:"Y",205:"Y",206:"U",208:"Y",209:"C",
 214:"Y",215:"V",217:"V",220:"V",221:"Y",222:"V",225:"Y",226:"C",227:"C",229:"V",233:"Y",236:"U",237:"T",238:"V",240:"V",
 241:"V",246:"Y",249:"V",250:"S",251:"V",255:"U",257:"V",258:"V",259:"V",260:"C",261:"C",263:"V",264:"V",265:"S",267:"V",
 268:"V",270:"V",274:"V",275:"V",276:"V",277:"V",281:"V",283:"V",284:"S",285:"U",286:"V",288:"U",291:"Y",295:"V",303:"U",
 305:"V",307:"V",308:"C",309:"V",310:"C",316:"V",317:"V",318:"V",
}
PRE2023 = {
 "A 17–95.6 TOPS/W Deep Learning Inference Accelerator": "V",
 "A Low-Latency Streaming On-Device Automatic Speech Recognition": "V",
 "A Resource-Saving Energy-Efficient Reconfigurable Hardware Accelerator for BERT": "V",
 "Accelerating NLP Tasks on FPGA with Compressed BERT": "V",
 "Analog-memory-based 14nm Hardware Accelerator": "C",
 "DNNFuser": "D",
 "M3ViT": "Y",
 "Accommodating Transformer onto FPGA": "V",
 "Vis-TOP": "V",
}
# Abstract-level census coding of the eligible (Y) designs: index -> (short name, platform class,
# regime by the rule of Sec. census_rule with ASIC/CIM chips lacking a power figure defaulted to mW,
# components named in the title/abstract, flags)
CODE = {
 15:("APTQ+","FPGA","fw",["GEMM","SM","ACT"],""), 24:("Fused-SD accelerator","ASIC","mw",["GEMM"],""),
 37:("RISC-V+LLM attention","ASIC","mw",["ATT"],""), 39:("Time-domain analog Softmax","ASIC op.","mw",["SM"],""),
 40:("DSLA","ASIC","mw",["GEMM"],""), 45:("EdgeCIM","CIM","mw",["GEMM"],""), 59:("HALO","ASIC","mw",["GEMM"],""),
 60:("HINT","CIM","mw",["GEMM"],""), 64:("HyPIM","3D-PIM","mw",["GEMM","ATT"],""), 70:("LogFlex","ASIC","mw",["GEMM"],""),
 74:("MVP","3D DRAM PNM","fw",["ATT","KV"],""), 75:("MicroXPU","CIM","mw",["GEMM"],""), 77:("NASiC","3D NAND CIM","fw",["GEMM","MOE"],""),
 78:("GI-FeFET CIM","CIM","mw",["GEMM","ACT"],""), 91:("SATIN","ASIP","mw",["ATT","KV"],""), 92:("SATLLM","ASIC","mw",["GEMM"],""),
 97:("SoftApprox","FPGA op.","fw",["SM"],""), 99:("T3","FPGA","fw",["SM","NORM"],"e"), 104:("Three Birds","ASIC","mw",["GEMM"],""),
 108:("TriGen","NPU","fw",["GEMM"],""), 112:("3D-CIMlet","CIM chiplets","mw",["GEMM"],""), 120:("Bit-layered NUQ","ASIC","mw",["GEMM"],""),
 121:("H-INMC","eDRAM CIM","mw",["ATT"],""), 122:("DCIM FlashAttention","CIM","mw",["ATT","SM"],""), 127:("Hybrid PIM-CIM","DRAM-PIM + CIM","fw",["ATT","KV"],""),
 131:("FS-Cap Delta-CIM","CIM","mw",["GEMM"],""), 132:("One-shot FP-CIM","CIM macro","mw",["GEMM"],""), 152:("BBAL","ASIC","mw",["GEMM"],""),
 153:("CINELL","eDRAM CIM","mw",["ATT"],""), 154:("Robotics FPGA LLM","FPGA","fw",["GEMM"],""), 155:("GWQ DRAM-PIM","DRAM-PIM","fw",["GEMM"],""),
 160:("KV260 tiled GEMM","FPGA","fw",["GEMM"],""), 168:("Decoder LM on CNN NPU","NPU","fw",["GEMM"],""),
 171:("Fully fused integer-only","FPGA","fw",["GEMM","SM","NORM"],"e"), 172:("F-BFQ","FPGA","fw",["GEMM"],""),
 185:("KV-CIM","CIM","mw",["KV","ATT"],""), 186:("Artix-7 LLM","FPGA","fw",["GEMM","ATT","NORM","KV"],""),
 188:("LLM-NPU","NPU","fw",["GEMM"],""), 189:("LLM-on-the-Palm","NAND PIM","fw",["GEMM"],""), 191:("Lembda","CPU+FPGA","fw",["GEMM"],""),
 192:("LoRA-CIM","CIM","mw",["GEMM"],""), 195:("Division-free nonlinear","NPU op.","fw",["SM","NORM","ACT"],"e"),
 199:("MSE-Softmax","FPGA op.","fw",["SM"],""), 201:("MoE-OPU","FPGA","fw",["GEMM","MOE"],""), 204:("NX-CGRA","CGRA","mw",["GEMM","SM","NORM","ACT"],""),
 205:("ORCHES","GPU-PIM","fw",["GEMM","KV"],""), 208:("PNM-X","PNM","fw",["GEMM"],""), 214:("ROMA","ROM-SRAM ASIC","mw",["GEMM"],""),
 221:("SLIM","ASIC","mw",["GEMM","ATT"],""), 225:("SparCIM","CIM","mw",["GEMM"],""), 233:("VEDA","ASIC","mw",["KV","ATT"],""),
 246:("Attention-Softmax unit","ASIC op.","mw",["SM","ATT"],""), 291:("Xiangshan LLM ISA","RISC-V","1w",["GEMM"],""),
 "M3ViT":("M3ViT","FPGA","fw",["MOE"],"e"),
}
