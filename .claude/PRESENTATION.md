# Dissertation Progress Presentation
## Machine Learning-Driven Automated Unit Test Generation for System-Level Programming Languages

---

## Slide 1: Title
**Machine Learning-Driven Automated Unit Test Generation for System-Level Programming Languages**

- Author: [Your Name]
- Dissertation Progress Report
- Date: January 2026

---

## Slide 2: Problem Statement

### The Challenge
- System-level languages (C, Rust) require thorough testing
- Manual test writing is time-consuming and error-prone
- Existing automated tools have limitations:
  - Low coverage
  - Poor edge case handling
  - Language-specific constraints

### Research Aim
To develop and evaluate an ML-driven approach for automated unit test generation that improves code coverage and test quality for system-level programming languages (C, Rust).

### Research Tasks
1. Analyze existing automated test generation approaches and their limitations
2. Design an architecture for ML/LLM-based test generation for system-level languages
3. Implement a working prototype tool supporting C and Rust
4. Develop a benchmark suite for comparative evaluation
5. Evaluate the ML-based approach against traditional template-based methods
6. Analyze results and formulate recommendations for practical application

---

## Slide 3: Research Objectives

1. **Design** an ML-driven test generation architecture for C and Rust
2. **Implement** a working prototype tool (MLTest)
3. **Evaluate** against traditional template-based approaches
4. **Analyze** coverage, pass rate, and test quality improvements

---

## Slide 4: Methodology

### Approach
1. **Code Analysis**: Parse source code to extract function signatures and semantics
2. **LLM Integration**: Use large language models for intelligent test generation
3. **Test Execution**: Compile and run generated tests with coverage measurement
4. **Comparative Evaluation**: LLM-generated vs template-based tests

### Technologies
- **Programming Language**: Python 3.13+
- **LLM APIs**: OpenAI GPT-4o, Anthropic Claude
- **C Toolchain**: GCC compiler, gcov for coverage
- **Rust Toolchain**: Cargo, cargo-tarpaulin for coverage
- **Target Languages**: C, Rust

---

## Slide 5: Tool Architecture

![Tool Architecture](graphs/architecture_diagram.png)

*Modular pipeline: Parse source code, generate tests via LLM/Template, execute with coverage measurement*

---

## Slide 6: Implementation Details

### Components Developed
- **Code Parsers**: Extract function signatures, parameters, types
- **LLM Generator**: Uses OpenAI/Anthropic APIs for intelligent test generation
- **Template Generator**: Baseline comparison (traditional approach)
- **Test Runners**: Compile and execute tests for C (GCC) and Rust (Cargo)
- **Coverage Analyzer**: Measure line/branch coverage using gcov/tarpaulin

---

## Slide 7: Benchmarks

### Benchmark Suite
| Category | C Functions | Rust Functions |
|----------|-------------|----------------|
| Math Utilities | 13 | 13 |
| String Utilities | 15 | 12 |
| Data Structures | 12 | 12 |
| **Total** | **40** | **37** |

### Function Types Tested
- Arithmetic operations with overflow handling
- String manipulation and parsing
- Array/collection operations
- Edge case handling (NULL, empty, boundaries)

---

## Slide 8: Results Overview

### Key Findings
| Metric | LLM-Generated | Template | Improvement |
|--------|---------------|----------|-------------|
| Line Coverage | 81.2% | 44.3% | **+36.9%** |
| Branch Coverage | 70.7% | 38.3% | **+32.4%** |
| Pass Rate | 79.2% | 66.2% | **+13.0%** |

> **LLM-generated tests achieve nearly double the code coverage compared to template-based approaches**

---

## Slide 9: Coverage Comparison Graph

![Coverage Comparison](graphs/coverage_comparison.png)

*ML-generated tests consistently achieve higher line and branch coverage*

---

## Slide 10: Benchmark Breakdown

![Benchmark Breakdown](graphs/benchmark_breakdown.png)

*Coverage improvements observed across all benchmark categories*

---

## Slide 11: Improvement Analysis

![Improvement Chart](graphs/improvement.png)

*Significant improvements in all key metrics*

---

## Slide 12: Coverage Distribution

![Coverage Distribution](graphs/coverage_distribution.png)

*LLM-generated tests show more consistent high coverage with less variance*

---

## Slide 13: Language Comparison

![Language Comparison](graphs/language_comparison.png)

*Similar improvement patterns observed for both C and Rust*

---

## Slide 14: Summary Dashboard

![Summary Dashboard](graphs/summary_dashboard.png)

*Comprehensive view of evaluation results*

---

## Slide 15: Demo

### Live Demonstration
```bash
# Generate tests for a C file
mltest generate benchmarks/c/math_utils.c -o test_math.c

# Run tests with coverage
mltest run benchmarks/c/math_utils.c test_math.c --coverage

# Run full benchmark suite
python run_evaluation.py --demo
```

---

## Slide 16: Key Contributions

1. **Novel Approach**: First systematic evaluation of LLM-based test generation for system-level languages
2. **Working Tool**: Functional prototype supporting C and Rust
3. **Empirical Evidence**: Quantitative comparison showing significant improvements
4. **Benchmark Suite**: Reusable test corpus for future research

---

## Slide 17: Limitations & Future Work

### Current Limitations
- Requires LLM API access (cost consideration)
- Simulated coverage for some complex scenarios
- Limited to function-level unit testing

### Future Directions
- Integration testing support
- Local model fine-tuning for reduced API dependency
- Additional languages (C++, Go)
- IDE integration (VS Code, CLion)

---

## Slide 18: Conclusion

### Summary
- Successfully developed ML-driven test generation tool
- Demonstrated **36.9%** improvement in line coverage
- Validated approach across 77 functions in C and Rust
- Established foundation for continued dissertation research

### Next Steps
- Expand benchmark suite
- Real-world project evaluation
- Publication preparation

---

## Slide 19: Questions?

**Thank you!**

Repository: `C:\Projects\dissertation`

Key Files:
- `mltest/` - Main tool package
- `benchmarks/` - Test corpus
- `results/` - Evaluation results
- `graphs/` - Visualizations

---

## Backup Slides

### B1: LLM Prompting Strategy
- System prompt with language-specific guidelines
- Function context including signature, body, dependencies
- Edge case identification heuristics

### B2: Coverage Measurement Details
- GCC `-fprofile-arcs -ftest-coverage` for C
- cargo-tarpaulin for Rust
- Fallback estimation when tools unavailable

### B3: Related Work
- EvoSuite (Java, search-based)
- Pex/IntelliTest (C#, symbolic execution)
- KLEE (C, symbolic execution)
- Codex/Copilot (general code generation)
