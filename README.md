# Frugal AI — Continual Learning Benchmark Review & StreamBench Experiments

M1 coursework project (Frugal AI course, Université Paris-Saclay): a **systematic review of continual-learning benchmarks for LLMs and language agents**, plus hands-on StreamBench (Self-StreamICL) experiments.

## Highlights

- **Systematic benchmark review** (`methodology.md`, `benchmark_anatomy_analysis.md`, `continual_learning_agent_benchmarks.md`) — PRISMA-inspired review of benchmarks (StreamBench, LifelongAgentBench, SWE-Bench-CL, MemoryBench, TRACE, and more), covering RQ1–RQ5: inventory, construction, measurement, validity, comparability.
- **StreamBench self-streaming ICL experiment** (`StreamBench_Self_StreamICL_M1.ipynb`) — runs StreamBench-style self-stream in-context learning with real results.
- **Reproducible notebook generation** — `build_real_streambench_notebook.py` / `build_streambench_notebook.py` build the experiment notebook; `execute_notebook_inprocess.py` executes it in-process.
- **Results** — `results/` contains real run outputs (`results_qwen_streambench.csv`, `summary_qwen_streambench.csv`).
- **References** — `references.bib`.

## Quick start

```bash
pip install jupyter nbformat pandas
jupyter notebook StreamBench_Self_StreamICL_M1.ipynb
```

## Layout

```
├── StreamBench_Self_StreamICL_M1.ipynb   # main experiment notebook
├── build_real_streambench_notebook.py    # generate experiment notebook
├── build_streambench_notebook.py
├── execute_notebook_inprocess.py
├── methodology.md                        # systematic review methodology
├── benchmark_anatomy_analysis.md         # benchmark anatomy analysis
├── continual_learning_agent_benchmarks.md
├── results/                              # real run outputs (csv)
├── docs/
└── references.bib
```

## Notes

- Model checkpoints and raw experiment data are not included in this repo (see `.gitignore`).
- This repository is the course group project; the review follows a PRISMA-inspired reporting logic.
