# Component-Aware Hardware for Edge LLM Inference: data and scripts

Companion material for the survey *Component-Aware Hardware for Edge LLM Inference: A Survey from
Memory-Bound Decoding to Non-GEMM Operations and Reasoning Workloads* (A. Aldashti, H. Saleh,
M. Al-Qutayri, G. Alsuhli; Khalifa University). Everything the paper computes can be regenerated
from this folder; nothing here is needed to *read* the paper.

## What is here

| Folder / file | What it is | Used for |
|---|---|---|
| `census_data.py` | The design census: 84 edge-regime LLM hardware designs, each with year, platform, power class, inference phase, components, implementation status and flags. One Python tuple per design. | Gap matrix (Fig. 14), timeline (Fig. 7), census table (Table C.11), all counts in the text. |
| `make_figs.py` | Regenerates the census table and the census-derived figures. | |
| `census_robustness.py` | Recomputes the gap matrix under the nine sensitivity scenarios and the 10,000 random reclassifications. | Table 3. |
| `edge_token_model.py` | The first-order per-component model of one decode token (Eq. 5): operation and byte counts from public model configurations, nominal platform parameters. | Figs. 4 and 8. |
| `frontier_model.py` | The same model applied to reasoning contexts, latent/hybrid attention, speculative and diffusion decoding, and MoE. Every assumption is a named constant. | Fig. 13. |
| `edge_bottleneck_calc.py` | Memory-bound throughput ceilings for the compared designs. | Fig. 11. |
| `validation/` | **Measured data.** llama.cpp `llama-bench` results (`cpu_*.json`, `gpu_*.json`), the eager-PyTorch operator profiles (`prof_*.json`), the bandwidth microbenchmark (`membw.py`), the profiler (`profile_ops.py`) and the analysis/figure script (`analyze.py`). Each JSON records the CPU/GPU name and software versions. | Sec. 6.4, Fig. 9. |
| `prisma/` | **The logged literature search.** `search.py` (exact OpenAlex and arXiv queries), `raw/` (all 7,340 retrieved records), `search_log.json` (run date and hit counts), `screen.py` (deduplication and rule-based screening), `assessed.csv` (the 330 hand-assessed candidates with a decision and reason each), `audit_designs.py` (the 54 additional eligible designs). | Sec. 2, Fig. 2. |
| `second_coder/` | A 30-row blank coding sheet and `kappa.py`, which scores an independent coder against the census (Cohen's kappa). | Inter-coder agreement. |
| `verify_refs.py` | Checks every cited reference against its DOI / arXiv record. | Reference integrity. |
| `edge_llm_refs.bib`, `manuscript.aux` | Bibliography and the list of cited keys. | `make_corpus.py` (Fig. 3). |

## How to regenerate

```
pip install numpy matplotlib
python edge_token_model.py; python frontier_model.py; python validation/analyze.py
python make_figs.py; python census_robustness.py; python make_corpus.py
cd prisma; python screen.py; python assess.py            # from the stored raw records
cd prisma; python search.py                              # optional: re-run the search (network, ~10 min)
```
`validation/profile_ops.py` needs PyTorch and Transformers 4.46 plus the Llama-3.2-1B-Instruct
weights; the llama.cpp commands are given in Appendix B of the paper.

## Extending the census

Add one tuple to `CENSUS` in `census_data.py` (the file header documents every field), then run
`make_figs.py` and `census_robustness.py`. Every figure, table and count that depends on the census
updates.

## Licence and citation

Scripts: MIT. Data (census, measured results, search records): CC BY 4.0. See `LICENSE`.
If you use this material, please cite the survey (citation details will be added on acceptance).
