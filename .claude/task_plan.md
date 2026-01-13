# Dissertation Presentation: ML-Driven Automated Unit Test Generation

## Goal
Create a working prototype and presentation-ready results for "Machine Learning-Driven Automated Unit Test Generation for System-Level Programming Languages"

## Deadline
Next week - presentation showing progress with code, graphs, and tool demo

---

## Phase 1: Research & Foundation [status: complete]
- [x] Research existing test generation approaches (EvoSuite, Pex, Randoop, KLEE)
- [x] Study ML approaches for code generation (CodeBERT, CodeT5, StarCoder)
- [x] Identify target system-level languages (C, Rust)
- [x] Define evaluation metrics (coverage, mutation score, assertion quality)

## Phase 2: Architecture Design [status: complete]
- [x] Design tool architecture
- [x] Choose ML model approach (LLM API prompting)
- [x] Define input/output format for test generation
- [x] Plan data pipeline

## Phase 3: Implementation [status: complete]
- [x] Set up project structure
- [x] Implement code parser/analyzer for C
- [x] Implement code parser/analyzer for Rust
- [x] Implement ML model integration (OpenAI/Anthropic)
- [x] Build test generation pipeline
- [x] Create CLI interface

## Phase 4: Evaluation & Benchmarks [status: complete]
- [x] Create benchmark programs (3 C files, 3 Rust files)
- [x] Run test generation on benchmarks
- [x] Measure code coverage achieved
- [x] Compare with baseline approaches
- [x] Collect metrics data

## Phase 5: Visualization & Graphs [status: complete]
- [x] Coverage comparison charts
- [x] Test quality metrics graphs
- [x] Performance benchmarks visualization
- [x] Summary dashboard

## Phase 6: Presentation Prep [status: complete]
- [x] Compile results into presentation format
- [x] Create demo script
- [x] Prepare talking points

---

## Key Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target Language | C & Rust | Most critical system-level languages |
| ML Approach | LLM API (OpenAI/Anthropic) | Quick results, high quality |
| Evaluation Metrics | Line coverage, branch coverage, pass rate | Industry standard metrics |

## Results Summary
| Metric | LLM-Generated | Template Baseline | Improvement |
|--------|---------------|-------------------|-------------|
| Line Coverage | 81.2% | 44.3% | +36.9% |
| Branch Coverage | 70.7% | 38.3% | +32.4% |
| Pass Rate | 79.2% | 66.2% | +13.0% |

## Files Created
| File | Purpose |
|------|---------|
| mltest/ | Main tool package |
| benchmarks/c/ | C benchmark programs (3 files, 40 functions) |
| benchmarks/rust/ | Rust benchmark programs (3 files, 37 functions) |
| results/results.json | Evaluation metrics data |
| graphs/*.png | 8 visualization charts |
| PRESENTATION.md | Presentation outline and slides |
| run_evaluation.py | Evaluation runner script |
