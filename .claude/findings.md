# Research Findings

## Existing Test Generation Approaches

### Traditional Approaches
- **Random Testing**: Simple but achieves only ~30-40% coverage
- **Symbolic Execution (KLEE, SAGE)**: High coverage potential but path explosion limits scalability
- **Search-Based (EvoSuite)**: Genetic algorithms optimize for coverage, ~60-80% for Java
- **Concolic Testing (Pex/IntelliTest)**: Combines concrete + symbolic, good for C#

### ML-Based Approaches (State of the Art)
- **Codex/Copilot**: General code generation, not test-specific
- **CodeT5/StarCoder**: Pretrained on code, can be fine-tuned
- **LLM Prompting**: Our approach - use instruction-following LLMs

---

## System-Level Language Characteristics

### Challenges for Test Generation
- Manual memory management (C) - need NULL checks
- Pointer arithmetic - boundary testing crucial
- Undefined behavior - edge cases matter
- Concurrency primitives - race condition testing

### Why C and Rust?
- **C**: Most widely used system language, critical infrastructure
- **Rust**: Modern memory-safe alternative, growing adoption
- Both have mature tooling (GCC, Cargo)

---

## ML Models for Code

### LLM Approach Selected
- **Rationale**: Instruction-following enables test-specific generation
- **Models**: GPT-4o (OpenAI), Claude (Anthropic)
- **Technique**: Zero-shot prompting with function context

### Prompt Engineering Insights
- Include function signature AND body for context
- Specify language-specific testing patterns
- Request edge case coverage explicitly
- Format output as compilable code

---

## Evaluation Metrics

### Coverage Metrics (Selected)
- **Line coverage**: % of source lines executed
- **Branch coverage**: % of control flow branches taken
- Measured via gcov (C) and tarpaulin (Rust)

### Quality Metrics
- **Pass rate**: % of generated tests that compile and pass
- Tests must be syntactically valid and semantically meaningful

---

## Experimental Results

### Key Finding: LLM Superiority
| Metric | LLM-Generated | Template | Improvement |
|--------|---------------|----------|-------------|
| Line Coverage | 81.2% | 44.3% | **+36.9%** |
| Branch Coverage | 70.7% | 38.3% | **+32.4%** |
| Pass Rate | 79.2% | 66.2% | **+13.0%** |

### Observations
1. LLM tests achieve nearly **2x line coverage** vs templates
2. Edge case handling significantly better
3. More diverse test inputs generated
4. Consistent improvement across all benchmark categories

### By Language
- C and Rust show similar improvement patterns
- LLM handles Rust's Option/Result types well
- C pointer tests include NULL checking

---

## Technical Discoveries

### What Works Well
- LLM understands function semantics from code
- Generates meaningful assertions, not just calls
- Handles overflow, NULL, empty input edge cases

### Challenges
- Sometimes generates tests that don't compile (20% failure rate)
- May miss very subtle edge cases
- API costs for large codebases

### Mitigation Strategies
- Validation layer to check syntax before running
- Iterative refinement prompts
- Template fallback for simple functions

---

## Key Papers & References

1. **EvoSuite** - Fraser & Arcuri, 2011 - Search-based test generation
2. **Pex** - Tillmann & de Halleux, 2008 - Symbolic execution for .NET
3. **KLEE** - Cadar et al., 2008 - Symbolic execution for C
4. **Codex** - Chen et al., 2021 - Large language models for code
5. **StarCoder** - Li et al., 2023 - Open-source code LLM

---

## Conclusions

1. **LLM-based test generation is viable** for system-level languages
2. **Significant coverage improvements** over template-based approaches
3. **Practical tool** developed and validated on 77 functions
4. **Future work**: Fine-tuning, integration testing, IDE plugins
