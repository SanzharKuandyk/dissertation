# MLTest

Master's thesis project on **predicting LLM suitability for unit test generation** in system-level code (C, C++, Rust).

The idea is simple: not every function is worth calling an expensive LLM for.
Some are trivial enough that a template works. Some are too complex for any
generator to handle well. So before generation, MLTest screens each function
using 22 static source-level features and a Random Forest classifier. It
returns a suitability score and assigns one of three buckets:
`good_candidate`, `borderline`, `risky`.

Trained and evaluated on 158 functions across 9 benchmark files. Real
LLM generation runs are included for context (gpt-5 by default).

## Quick start

```bash
pip install -e .
python -m mltest screen benchmarks/cpp/math_utils.cpp --top 10
```

That parses the file, scores each function, and prints a ranked table — no API
key needed.

## Layout

- `mltest/` — package: parsers, feature extractor, ML model, CLI, runners
- `benchmarks/` — C / C++ / Rust source files used for training and evaluation
- `models/strategy_selector.joblib` — trained Random Forest
- `results/` — JSON outputs from screening and benchmark runs
- `graphs/` — generated charts
- `run_evaluation.py` — full benchmark pipeline (LLM + template + screening)

## Acknowledgements

Built as part of a master's thesis at Astana IT University.

This project used AI assistance during development. Claude (Anthropic) and
ChatGPT (OpenAI) were used for code review, debugging, refactoring suggestions,
and helping draft parts of the thesis manuscript. All design decisions, final
implementation choices, experiments, and conclusions are my own; the AI tools
acted as a faster pair-programming partner, not as authors.

The downstream generation pipeline itself also calls an LLM (`gpt-5` by
default) as part of the system under study — that's a separate role and is
documented in the thesis.
