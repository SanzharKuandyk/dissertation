# Dissertation Video Script
## ML-Driven Automated Unit Test Generation for System-Level Languages
**Target Duration**: 1:00 - 1:30

---

## SCENE 1: INTRODUCTION (0:00 - 0:12)

### VISUAL
- Show title card: "ML-Driven Automated Unit Test Generation"
- Subtitle: "C & Rust | MLTest Tool"
- Quick transition to VS Code with a C/Rust file open

### NARRATION
> "Writing unit tests for system-level code is time-consuming and error-prone. What if AI could do it for you? Meet MLTest - an ML-powered tool that automatically generates comprehensive unit tests for C and Rust."

### TECHNICAL NOTES
- Use a clean, simple C function as the example (e.g., `benchmarks/c/math_utils.c`)
- Keep title card visible for 3-4 seconds max

---

## SCENE 2: TOOL DEMONSTRATION (0:12 - 0:45)

### VISUAL - Part A (0:12 - 0:30)
1. Show terminal with clear, large font
2. Display the source code file you'll test (e.g., `math_utils.c` with 2-3 functions)
3. Execute command: `mltest generate benchmarks/c/math_utils.c -o test_math.c`
4. Show the tool working (can speed up if needed)
5. Display generated test file with multiple test cases

### NARRATION - Part A
> "Here's a simple C file with math utilities. With one command, MLTest analyzes the code, uses GPT-4 to understand edge cases, and generates comprehensive test suites. Notice how it creates tests for normal cases, boundary conditions, and error scenarios."

### TECHNICAL NOTES - Part A
- Use `--model gpt4o` or `--model claude` flag to show ML aspect
- Have the output ready or record it beforehand to avoid API delays
- Highlight 2-3 interesting test cases in the generated file

### VISUAL - Part B (0:30 - 0:45)
1. Execute: `mltest run benchmarks/c/math_utils.c test_math.c --coverage`
2. Show coverage results appearing in real-time
3. Display coverage percentages (line & branch coverage)
4. Show comparison: LLM vs Template side-by-side

### NARRATION - Part B
> "Running the tests shows impressive results. The ML-generated tests achieve over 80% line coverage, compared to just 44% from traditional template-based approaches - nearly double the coverage."

### TECHNICAL NOTES - Part B
- Use a benchmark that shows good results
- Optionally split screen to show template vs LLM comparison
- Keep the terminal output clean and readable

---

## SCENE 3: RESULTS SHOWCASE (0:45 - 1:15)

### VISUAL
1. Show `graphs/coverage_comparison.png` (0:45 - 0:55)
2. Show `graphs/benchmark_breakdown.png` (0:55 - 1:05)
3. Show `graphs/summary_dashboard.png` (1:05 - 1:15)

### NARRATION
> "Across 77 functions in C and Rust, MLTest consistently outperforms traditional methods. Line coverage improved by 37%, branch coverage by 32%, and test pass rates by 13%. The tool works equally well for both C and Rust, proving its versatility for system-level languages."

### TECHNICAL NOTES
- Ensure graphs are high-resolution and clearly visible
- Consider adding subtle zoom-ins on key metrics
- Use smooth transitions between graphs (1-2 seconds each)

---

## SCENE 4: CONCLUSION (1:15 - 1:30)

### VISUAL
- Return to terminal or show project structure
- Display key stats overlay:
  - "81.2% Line Coverage"
  - "70.7% Branch Coverage"
  - "77 Functions Tested"
- End with GitHub/project repository link

### NARRATION
> "MLTest demonstrates that AI can significantly improve automated testing for system-level languages, saving developer time while increasing code quality. The future of test generation is intelligent, adaptive, and powered by machine learning."

### TECHNICAL NOTES
- Keep final stats visible for 3-5 seconds
- Add contact info or repository link at the very end
- Consider subtle background music (optional)

---

## ALTERNATIVE SHORTER VERSION (1:00 exactly)

If you need to cut to exactly 60 seconds:

### TIMING ADJUSTMENTS
- **Intro**: 0:00 - 0:10 (cut 2 seconds)
- **Demo**: 0:10 - 0:35 (combine parts A and B, show one quick workflow)
- **Results**: 0:35 - 0:55 (show only 2 key graphs)
- **Conclusion**: 0:55 - 1:00 (brief wrap-up with stats)

### CONDENSED NARRATION
Keep the same structure but speak slightly faster and eliminate transition phrases.

---

## PRE-RECORDING CHECKLIST

### Setup
- [ ] Clean terminal with large, readable font (16-18pt minimum)
- [ ] Close unnecessary applications/tabs
- [ ] Test run all commands to ensure they work
- [ ] Pre-generate test files if API is slow
- [ ] Ensure graphs are exported and accessible
- [ ] Set up screen recording software (OBS, Camtasia, etc.)

### Commands to Prepare
```bash
# Have these ready to copy-paste or script
mltest generate benchmarks/c/math_utils.c -o test_math.c --model gpt4o
mltest run benchmarks/c/math_utils.c test_math.c --coverage
python run_evaluation.py --demo  # if you have this
```

### Visual Assets Needed
- Title card (create in PowerPoint/Canva)
- All graphs from `graphs/` folder
- Source code files (with syntax highlighting)
- Terminal output (clean, no errors)

### Recording Tips
1. **Screen Resolution**: Use 1920x1080 (1080p) for clarity
2. **Recording Speed**: You can record slower and speed up in editing
3. **Cursor Movement**: Keep mouse movements smooth and purposeful
4. **Text Size**: Zoom in terminal and code editor (Ctrl/Cmd +)
5. **Practice**: Do 2-3 dry runs before final recording
6. **Backup**: Record multiple takes for each scene

---

## POST-PRODUCTION NOTES

### Editing
- Add smooth transitions between scenes (0.5-1 second)
- Consider adding subtle zoom-ins on important metrics
- Add text overlays for key statistics
- Background music (optional, keep low volume)

### Text Overlays to Add
- "36.9% Improvement" (when showing coverage comparison)
- "LLM vs Template" (when showing comparison)
- "C & Rust Support" (when showing language comparison)
- Contact/Repository info at the end

### Export Settings
- Format: MP4 (H.264)
- Resolution: 1920x1080 (1080p)
- Frame Rate: 30fps
- Bitrate: 8-10 Mbps for good quality

---

## BACKUP: 30-SECOND VERSION

If you need an ultra-short version:

**VISUAL**: Quick command demo → Coverage results → Key graph
**NARRATION**: "MLTest uses AI to automatically generate unit tests for C and Rust. It achieves 81% line coverage - double that of traditional methods - across 77 functions. Intelligent testing, powered by machine learning."
**TIME**: 0:00-0:30

---

## Questions to Consider

Before recording, decide on:
1. Do you want to show your face (webcam in corner)?
2. Will you use background music?
3. Do you want captions/subtitles?
4. What platform is this for? (YouTube, presentation, defense?)
5. Do you need to add university branding/logo?

Good luck with your video! 🎬
