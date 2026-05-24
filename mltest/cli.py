"""
MLTest CLI - screening, LLM-assisted generation, and evaluation workflows.
"""

import click
import json
import os
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
LANGUAGE_LABELS = {"c": "C", "cpp": "C++", "rust": "Rust"}


@click.group()
@click.version_option(version="0.1.0")
def main():
    """MLTest - static screening and LLM-assisted unit testing.

    Screen C, C++, and Rust functions for LLM suitability, then generate and
    run tests for the languages that are supported end to end.
    """
    pass


@main.command()
@click.argument("source_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--language",
    "-l",
    type=click.Choice(["c", "rust", "cpp", "auto"]),
    default="auto",
    help="Source language (auto-detect by default)",
)
@click.option(
    "--model-path",
    default="models/strategy_selector.joblib",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
    help="Path to the trained suitability model",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Optional JSON output path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Console output format",
)
@click.option("--top", type=int, default=None, help="Limit displayed functions to the top N scores")
def screen(
    source_file: str,
    language: str,
    model_path: str,
    output: Optional[str],
    output_format: str,
    top: Optional[int],
):
    """Screen a source file for LLM test-generation suitability.

    Example: mltest screen benchmarks/c/math_utils.c
    """
    from .ml.strategy_selector import MLStrategySelector

    source_path = Path(source_file)
    resolved_language = _detect_language(source_path, language, {"c", "rust", "cpp"})
    source_code = source_path.read_text(encoding="utf-8", errors="ignore")

    console.print(
        Panel.fit(
            f"[bold blue]MLTest Screening[/bold blue]\n"
            f"Ranking {source_path.name} for LLM suitability",
            border_style="blue",
        )
    )
    console.print(f"Language: {LANGUAGE_LABELS.get(resolved_language, resolved_language)}")
    console.print(f"Model: {Path(model_path)}")

    console.print("Parsing source file...")
    functions = _parse_source(source_code, resolved_language)
    if not functions:
        console.print("[yellow]No testable functions found.[/yellow]")
        return

    console.print(f"Found {len(functions)} testable function(s)")
    console.print("Loading suitability model...")
    selector = MLStrategySelector()
    selector.load_model(Path(model_path))

    console.print("Scoring functions...")
    scored_functions = []
    for func in functions:
        score = selector.score_function(func, resolved_language)
        scored_functions.append(
            {
                "name": func.name,
                "line_start": int(getattr(func, "line_start", 0) or 0),
                "line_end": int(getattr(func, "line_end", 0) or 0),
                "llm_suitability_score": round(score["llm_suitability_score"], 4),
                "bucket": score["bucket"],
                "predicted_label": score["predicted_label"],
            }
        )

    report = _build_screening_report(
        source_path=source_path,
        language=resolved_language,
        model_path=Path(model_path),
        scored_functions=scored_functions,
    )

    display_report = report
    if top is not None and top > 0:
        display_report = dict(report)
        display_report["functions"] = report["functions"][:top]

    if output_format == "json":
        console.print_json(json.dumps(display_report, indent=2))
    else:
        _display_screening_table(display_report)

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        console.print(f"[green]Screening report written to {output_path}[/green]")


@main.command()
@click.option(
    "--report",
    "report_path",
    type=click.Path(exists=True, dir_okay=False),
    default="results/llm_testability_report.json",
    show_default=True,
    help="Path to the screening report JSON",
)
@click.option(
    "--graphs-dir",
    type=click.Path(exists=True, file_okay=False),
    default="graphs",
    show_default=True,
    help="Directory containing the presentation graphs",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="presentation/progress_dashboard.html",
    show_default=True,
    help="Output HTML dashboard path",
)
@click.option(
    "--title",
    default="MLTest Progress Dashboard",
    show_default=True,
    help="Title shown in the generated dashboard",
)
def present(report_path: str, graphs_dir: str, output: str, title: str):
    """Build a static HTML dashboard for progress reviews.

    Example: mltest present --report results/llm_testability_report.json
    """
    from .presentation import build_progress_dashboard

    report = Path(report_path)
    graphs = Path(graphs_dir)
    output_path = Path(output)

    console.print(
        Panel.fit(
            f"[bold blue]MLTest Presentation Dashboard[/bold blue]\n"
            f"Building a browser-ready progress view from {report.name}",
            border_style="blue",
        )
    )

    generated = build_progress_dashboard(
        report_path=report,
        graphs_dir=graphs,
        output_path=output_path,
        title=title,
    )
    console.print(f"[green]Dashboard written to {generated}[/green]")


@main.command()
@click.argument('source_file', type=click.Path(exists=True))
@click.option('--language', '-l', type=click.Choice(['c', 'cpp', 'rust', 'auto']),
              default='auto', help='Source language for generation')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'template']),
              default='openai', help='LLM provider or template baseline')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output file for generated tests')
@click.option('--api-key', help='API key for LLM provider (auto-loads from .env)')
def generate(source_file: str, language: str, provider: str,
             output: Optional[str], api_key: Optional[str]):
    """Generate unit tests for a source file.

    Example: mltest generate math_utils.c -o test_math.c
    """
    source_path = Path(source_file)

    language = _detect_language(source_path, language, {"c", "cpp", "rust"})

    # Auto-load API key from environment if not provided
    if not api_key:
        if provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')

    console.print(Panel(f"[bold]MLTest - Generating tests for {source_path.name}[/bold]"))
    console.print(f"Language: {language.upper()}")
    console.print(f"Provider: {provider}")

    # Read source file
    source_code = source_path.read_text()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Parse source
        task1 = progress.add_task("Parsing source code...", total=None)
        functions = _parse_source(source_code, language)
        progress.update(task1, completed=True)

        if not functions:
            console.print("[yellow]No testable functions found.[/yellow]")
            return

        console.print(f"Found {len(functions)} testable function(s)")

        # Generate tests
        task2 = progress.add_task("Generating tests...", total=len(functions))
        generated_tests = []

        for func in functions:
            test = _generate_test(func, language, provider, api_key)
            generated_tests.append(test)
            progress.advance(task2)

    # Combine all tests
    combined_code = _combine_tests(generated_tests, language, source_path.name)

    # Output
    if output:
        output_path = Path(output)
        output_path.write_text(combined_code)
        console.print(f"[green]Tests written to {output_path}[/green]")
    else:
        console.print("\n[bold]Generated Tests:[/bold]")
        console.print(combined_code)


@main.command()
@click.argument('source_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--language', '-l', type=click.Choice(['c', 'cpp', 'rust', 'auto']),
              default='auto', help='Source language for execution')
@click.option('--coverage/--no-coverage', default=True, help='Measure coverage')
def run(source_file: str, test_file: str, language: str, coverage: bool):
    """Run generated tests and measure coverage.

    Example: mltest run math_utils.c test_math.c
    """
    source_path = Path(source_file)
    test_path = Path(test_file)

    language = _detect_language(source_path, language, {"c", "cpp", "rust"})

    console.print(Panel(f"[bold]Running tests for {source_path.name}[/bold]"))

    source_code = source_path.read_text()
    test_code = test_path.read_text()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running tests...", total=None)

        if language == 'c':
            from .runners.c_runner import CTestRunner
            runner = CTestRunner()
            if coverage:
                result, cov = runner.run_with_coverage(source_code, test_code, source_path.name)
            else:
                result = runner.compile_and_run(source_code, test_code, source_path.name)
                cov = None
        elif language == 'cpp':
            from .runners.cpp_runner import CppTestRunner
            runner = CppTestRunner()
            if coverage:
                result, cov = runner.run_with_coverage(source_code, test_code, source_path.name)
            else:
                result = runner.compile_and_run(source_code, test_code, source_path.name)
                cov = None
        else:
            from .runners.rust_runner import RustTestRunner
            runner = RustTestRunner()
            if coverage:
                result, cov = runner.run_with_coverage(source_code, test_code)
            else:
                result = runner.compile_and_run(source_code, test_code)
                cov = None

        progress.update(task, completed=True)

    # Display results
    _display_results(result, cov)


@main.command()
@click.argument('benchmark_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='results',
              help='Output directory for results')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'template', 'all']),
              default='openai', help='LLM provider to evaluate')
def benchmark(benchmark_dir: str, output_dir: str, provider: str):
    """Run benchmarks on a directory of source files.

    Example: mltest benchmark benchmarks/c -o results
    """
    from .coverage.analyzer import CoverageAnalyzer, FunctionCoverage

    benchmark_path = Path(benchmark_dir)
    results_path = Path(output_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel(f"[bold]Running benchmarks on {benchmark_path}[/bold]"))

    analyzer = CoverageAnalyzer(results_path)

    # Find all source files
    c_files = list(benchmark_path.glob('*.c'))
    cpp_files = list(benchmark_path.glob('*.cpp'))
    rust_files = list(benchmark_path.glob('*.rs'))

    all_files = (
        [(f, 'c') for f in c_files]
        + [(f, 'cpp') for f in cpp_files]
        + [(f, 'rust') for f in rust_files]
    )

    if not all_files:
        console.print("[yellow]No source files found in benchmark directory.[/yellow]")
        return

    console.print(f"Found {len(all_files)} source file(s)")

    if provider == 'all':
        providers_to_test = ['openai', 'template']
    elif provider == 'template':
        providers_to_test = ['template']
    else:
        providers_to_test = [provider, 'template']

    for source_file, language in all_files:
        console.print(f"\nProcessing: {source_file.name}")

        source_code = source_file.read_text()
        functions = _parse_source(source_code, language)

        for prov in providers_to_test:
            for func in functions:
                test = _generate_test(func, language, prov, None)

                # Run test and get coverage
                if language == 'c':
                    from .runners.c_runner import CTestRunner
                    runner = CTestRunner()
                    result, cov = runner.run_with_coverage(source_code, test.test_code, source_file.name)
                elif language == 'cpp':
                    from .runners.cpp_runner import CppTestRunner
                    runner = CppTestRunner()
                    result, cov = runner.run_with_coverage(source_code, test.test_code, source_file.name)
                else:
                    from .runners.rust_runner import RustTestRunner
                    runner = RustTestRunner()
                    result, cov = runner.run_with_coverage(source_code, test.test_code)

                func_cov = FunctionCoverage(
                    function_name=func.name if hasattr(func, 'name') else str(func),
                    language=language,
                    line_coverage=cov.line_coverage,
                    branch_coverage=cov.branch_coverage,
                    lines_covered=cov.lines_covered,
                    lines_total=cov.lines_total,
                    test_passed=result.passed,
                    execution_time=result.execution_time,
                    generator_type='llm' if prov != 'template' else 'template'
                )
                analyzer.add_function_result(source_file.stem, language, func_cov)

    # Save and display results
    analyzer.save_results()
    console.print("\n" + analyzer.generate_report())
    console.print(f"\n[green]Results saved to {results_path / 'results.json'}[/green]")


@main.command()
@click.argument('results_file', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), default='graphs',
              help='Output directory for graphs')
def visualize(results_file: str, output_dir: str):
    """Generate visualization graphs from results.

    Example: mltest visualize results/results.json -o graphs
    """
    from .visualization import create_all_charts

    results_path = Path(results_file)
    graphs_path = Path(output_dir)
    graphs_path.mkdir(parents=True, exist_ok=True)

    console.print(Panel(f"[bold]Generating visualizations[/bold]"))

    with open(results_path) as f:
        data = json.load(f)

    create_all_charts(data, graphs_path)
    console.print(f"[green]Graphs saved to {graphs_path}[/green]")


def _parse_source(source_code: str, language: str):
    """Parse source code and return functions"""
    if language == 'c':
        from .parsers.c_parser import CParser
        parser = CParser()
    elif language == 'cpp':
        from .parsers.cpp_parser import CppParser
        parser = CppParser()
    elif language == 'rust':
        from .parsers.rust_parser import RustParser
        parser = RustParser()
    else:
        raise click.ClickException(
            f"Unsupported language '{language}'. Supported values: c, cpp, rust."
        )

    parser.parse_source(source_code)
    return parser.get_testable_functions()


def _detect_language(source_path: Path, language: str, supported_languages: set[str]) -> str:
    """Resolve explicit or auto-detected language with scope checks."""
    if language != 'auto':
        if language not in supported_languages:
            supported = ", ".join(sorted(supported_languages))
            raise click.ClickException(
                f"Language '{language}' is not supported for this command. Supported: {supported}."
            )
        return language

    suffix_map = {
        '.c': 'c',
        '.h': 'c',
        '.rs': 'rust',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.hpp': 'cpp',
        '.hh': 'cpp',
        '.hxx': 'cpp',
    }
    detected = suffix_map.get(source_path.suffix.lower())
    if not detected or detected not in supported_languages:
        supported = ", ".join(sorted(supported_languages))
        raise click.ClickException(
            f"Cannot auto-detect a supported language for {source_path.name}. Supported: {supported}."
        )
    return detected


def _build_screening_report(
    source_path: Path,
    language: str,
    model_path: Path,
    scored_functions: list[dict],
) -> dict:
    """Build a JSON-serializable report for `mltest screen`."""
    ranked_functions = sorted(
        scored_functions,
        key=lambda item: item["llm_suitability_score"],
        reverse=True,
    )
    summary = {
        "total_functions": len(ranked_functions),
        "good_candidate_count": sum(1 for item in ranked_functions if item["bucket"] == "good_candidate"),
        "borderline_count": sum(1 for item in ranked_functions if item["bucket"] == "borderline"),
        "risky_count": sum(1 for item in ranked_functions if item["bucket"] == "risky"),
    }
    return {
        "source_file": str(source_path.resolve()),
        "language": language,
        "model_path": str(model_path.resolve()),
        "summary": summary,
        "functions": ranked_functions,
    }


def _display_screening_table(report: dict):
    """Render screening results as a ranked table."""
    table = Table(title="LLM Suitability Screening")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Function", style="cyan")
    table.add_column("Lines", justify="right")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Bucket")
    table.add_column("Prediction")

    for index, item in enumerate(report["functions"], start=1):
        table.add_row(
            str(index),
            item["name"],
            f"{item['line_start']}-{item['line_end']}",
            f"{item['llm_suitability_score']:.3f}",
            item["bucket"],
            item["predicted_label"],
        )

    console.print(table)
    summary = report["summary"]
    console.print(
        "Summary: "
        f"{summary['total_functions']} total | "
        f"{summary['good_candidate_count']} good | "
        f"{summary['borderline_count']} borderline | "
        f"{summary['risky_count']} risky"
    )


def _generate_test(func, language: str, provider: str, api_key: Optional[str]):
    from .generators.llm_generator import LLMTestGenerator, TemplateTestGenerator

    template_gen = TemplateTestGenerator()

    def _generate_with_template():
        if language == 'c':
            return template_gen.generate_c_tests(func)
        if language == 'cpp':
            return template_gen.generate_cpp_tests(func)
        return template_gen.generate_rust_tests(func)

    def _generate_with_llm(llm_gen):
        if language == 'c':
            return llm_gen.generate_c_tests(func)
        if language == 'cpp':
            return llm_gen.generate_cpp_tests(func)
        return llm_gen.generate_rust_tests(func)

    # Explicit template request → no fallback needed
    if provider == 'template':
        return _generate_with_template()

    # LLM with fallback
    try:
        llm_gen = LLMTestGenerator(provider=provider, api_key=api_key)
        return _generate_with_llm(llm_gen)
    except Exception as e:
        console.print(
            f"[yellow]LLM generation failed for {func.name}, "
            f"falling back to template: {e}[/yellow]"
        )
        return _generate_with_template()


def _combine_tests(tests, language: str, source_name: str) -> str:
    """Combine multiple test codes into one file"""
    if language == 'c':
        header = f'''/*
 * Auto-generated tests for {source_name}
 * Generated by MLTest
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>

'''
        # Combine all test functions
        test_functions = '\n\n'.join(t.test_code for t in tests)

        # Create main that calls all test functions
        test_calls = '\n    '.join(f'{t.test_name}();' for t in tests)
        main_func = f'''
int main() {{
    printf("Running {len(tests)} test suite(s)...\\n\\n");
    {test_calls}
    printf("\\nAll tests passed!\\n");
    return 0;
}}
'''
        return header + test_functions + '\n' + main_func
    elif language == 'cpp':
        header = f'''// Auto-generated tests for {source_name}
// Generated by MLTest

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>

'''
        test_functions = '\n\n'.join(t.test_code for t in tests)
        test_calls = '\n    '.join(f'{t.test_name}();' for t in tests)
        main_func = f'''
int main() {{
    std::cout << "Running {len(tests)} test suite(s)...\\n\\n";
    {test_calls}
    std::cout << "\\nAll tests completed.\\n";
    return 0;
}}
'''
        return header + test_functions + '\n' + main_func
    else:
        header = f'''//! Auto-generated tests for {source_name}
//! Generated by MLTest

'''
        return header + '\n\n'.join(t.test_code for t in tests)


def _display_results(result, coverage):
    """Display test results in a table"""
    table = Table(title="Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green" if result.passed else "red")

    table.add_row("Status", "PASSED" if result.passed else "FAILED")
    table.add_row("Execution Time", f"{result.execution_time:.3f}s")

    if coverage:
        table.add_row("Line Coverage", f"{coverage.line_coverage:.1f}%")
        table.add_row("Branch Coverage", f"{coverage.branch_coverage:.1f}%")
        table.add_row("Lines Covered", f"{coverage.lines_covered}/{coverage.lines_total}")

    if result.error_message:
        table.add_row("Error", result.error_message[:100])

    console.print(table)

    if result.output:
        console.print("\n[bold]Output:[/bold]")
        console.print(result.output[:1000])


@main.command("train-model")
@click.option('--results-file', default='results/results.json',
              type=click.Path(exists=True), show_default=True,
              help='Path to results.json from a previous benchmark run')
@click.option('--benchmarks-dir', default='benchmarks',
              type=click.Path(exists=True), show_default=True,
              help='Directory containing benchmark source files')
@click.option('--model-output', default='models/strategy_selector.joblib',
              type=click.Path(), show_default=True,
              help='Output path for the trained model artifact')
@click.option('--threshold', default=-1.0, type=float, show_default=True,
              help='LLM coverage threshold for label=1 (-1 = auto/median split)')
@click.option('--n-estimators', default=100, type=int, show_default=True,
              help='Number of trees in the Random Forest')
def train_model(results_file: str, benchmarks_dir: str, model_output: str,
                threshold: float, n_estimators: int):
    """Train the ML strategy selector from existing benchmark results.

    Reads results.json (produced by the benchmark command), re-parses the
    benchmark source files to extract code complexity features, then trains a
    Random Forest classifier with 5-fold cross-validation to predict which
    test generation strategy (LLM vs template) achieves higher coverage.

    The trained model is saved to --model-output and can be used by the
    evaluate-ml command.
    """
    from .ml.strategy_selector import MLStrategySelector
    import warnings

    console.print(Panel.fit(
        "[bold blue]ML Strategy Selector — Training[/bold blue]\n"
        "Training a Random Forest to predict optimal test generation strategy",
        border_style="blue",
    ))

    selector = MLStrategySelector()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True) as progress:
        task = progress.add_task("Training model...", total=None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = selector.train(
                results_path=Path(results_file),
                benchmarks_dir=Path(benchmarks_dir),
                model_output_path=Path(model_output),
                n_estimators=n_estimators,
                coverage_threshold=threshold,
            )
        progress.update(task, description="Done.")

    for w in caught:
        console.print(f"[yellow]Warning:[/yellow] {w.message}")

    # Cross-validation metrics table
    cv = metrics.get("cv_metrics", {})
    cv_table = Table(title="5-Fold Cross-Validation Metrics", show_header=True)
    cv_table.add_column("Metric", style="cyan", min_width=28)
    cv_table.add_column("Mean", style="green", justify="right")
    cv_table.add_column("Std", justify="right")

    metric_labels = {
        "test_accuracy": "Accuracy",
        "test_f1_weighted": "F1 (weighted)",
        "test_precision_weighted": "Precision (weighted)",
        "test_recall_weighted": "Recall (weighted)",
    }
    for key, label in metric_labels.items():
        if key in cv:
            cv_table.add_row(
                label,
                f"{cv[key]['mean']:.4f}",
                f"± {cv[key]['std']:.4f}",
            )
    console.print(cv_table)

    # Class distribution
    dist = metrics.get("class_distribution", {})
    console.print(
        f"\n[bold]Training samples:[/bold] {metrics['training_samples']}  |  "
        f"High-coverage / label=1: {dist.get(1, 0)}  |  "
        f"Low-coverage / label=0: {dist.get(0, 0)}"
    )

    # Top-10 feature importances
    imp_df = selector.get_feature_importances().head(10)
    imp_table = Table(title="Top-10 Feature Importances", show_header=True)
    imp_table.add_column("Rank", justify="right", style="dim")
    imp_table.add_column("Feature", style="cyan")
    imp_table.add_column("Importance", justify="right", style="green")
    for rank, row in enumerate(imp_df.itertuples(), 1):
        imp_table.add_row(str(rank), row.feature, f"{row.importance:.4f}")
    console.print(imp_table)

    console.print(f"\n[bold green]Model saved to:[/bold green] {model_output}")


@main.command("evaluate-ml")
@click.argument('benchmark_dir', type=click.Path(exists=True))
@click.option('--model-path', default='models/strategy_selector.joblib',
              type=click.Path(exists=True), show_default=True,
              help='Path to trained model artifact')
@click.option('--results-file', default='results/results.json',
              type=click.Path(exists=True), show_default=True,
              help='Path to results.json for baseline comparison')
@click.option('--output-dir', '-o', default='results',
              type=click.Path(), show_default=True,
              help='Directory to save ml_comparison.json and charts')
def evaluate_ml(benchmark_dir: str, model_path: str, results_file: str, output_dir: str):
    """Run ML-guided evaluation and compare against baselines.

    Uses the trained ML strategy selector to predict, per function, whether
    to use LLM or template test generation. Compares the resulting coverage
    and API usage against always-LLM and always-template baselines.

    Outputs ml_comparison.json and four ML visualisation charts.
    """
    from .ml.strategy_selector import MLStrategySelector
    from .visualization import create_ml_charts

    console.print(Panel.fit(
        "[bold green]ML-Guided Evaluation — 3-Way Comparison[/bold green]\n"
        "Always-LLM  vs  Always-Template  vs  ML-Guided",
        border_style="green",
    ))

    selector = MLStrategySelector()
    console.print(f"Loading model from [cyan]{model_path}[/cyan]...")
    selector.load_model(Path(model_path))

    with open(results_file) as f:
        import json as _json
        data = _json.load(f)

    from .parsers.c_parser import CParser
    from .parsers.rust_parser import RustParser
    from .parsers.cpp_parser import CppParser

    results_dir = Path(output_dir)
    graphs_dir = Path(benchmark_dir).parent / "graphs"
    results_dir.mkdir(exist_ok=True)
    graphs_dir.mkdir(exist_ok=True)

    # Build result lookup
    result_lookup = {}
    for bench in data.get("benchmarks", []):
        bname, lang, gtype = bench["benchmark_name"], bench["language"], bench["generator_type"]
        for r in bench.get("function_results", []):
            result_lookup[(bname, lang, gtype, r["function_name"])] = r

    all_functions = set()
    for (bname, lang, gtype, fname) in result_lookup:
        all_functions.add((bname, lang, fname))

    c_parser, rust_parser = CParser(), RustParser()
    parse_cache: dict = {}

    def _parse_file_cached(filepath, lang):
        if filepath in parse_cache:
            return parse_cache[filepath]
        source = filepath.read_text(encoding="utf-8")
        if lang == "c":
            result = {f.name: f for f in c_parser.parse_source(source)}
        elif lang == "rust":
            result = {f.name: f for f in rust_parser.parse_source(source)}
        elif lang == "cpp":
            result = {f.name: f for f in CppParser().parse_source(source)}
        else:
            result = {}
        parse_cache[filepath] = result
        return result

    ml_decisions, always_llm_covs, always_tmpl_covs, ml_guided_covs = [], [], [], []
    always_llm_passed = always_tmpl_passed = ml_guided_passed = ml_llm_count = total_funcs = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  transient=True) as progress:
        task = progress.add_task(f"Evaluating {len(all_functions)} functions...", total=None)

        for bname, lang, fname in sorted(all_functions):
            llm_key = (bname, lang, "llm", fname)
            tmpl_key = (bname, lang, "template", fname)
            if llm_key not in result_lookup or tmpl_key not in result_lookup:
                continue
            llm_r, tmpl_r = result_lookup[llm_key], result_lookup[tmpl_key]
            always_llm_covs.append(llm_r["line_coverage"])
            always_tmpl_covs.append(tmpl_r["line_coverage"])
            always_llm_passed += int(llm_r.get("test_passed", False))
            always_tmpl_passed += int(tmpl_r.get("test_passed", False))
            total_funcs += 1

            ext = {"c": ".c", "rust": ".rs", "cpp": ".cpp"}.get(lang, ".c")
            filepath = Path(benchmark_dir) / lang / f"{bname}{ext}"
            func_map = _parse_file_cached(filepath, lang) if filepath.exists() else {}
            func = func_map.get(fname)
            strategy = selector.predict(func, lang) if func else "llm"
            proba = selector.predict_proba(func, lang) if func else {"llm": 1.0, "template": 0.0}

            chosen_r = llm_r if strategy == "llm" else tmpl_r
            ml_guided_covs.append(chosen_r["line_coverage"])
            ml_guided_passed += int(chosen_r.get("test_passed", False))
            if strategy == "llm":
                ml_llm_count += 1

            ml_decisions.append({
                "benchmark": bname, "language": lang, "function": fname,
                "ml_strategy": strategy, "llm_confidence": round(proba["llm"], 3),
                "llm_line_cov": round(llm_r["line_coverage"], 2),
                "template_line_cov": round(tmpl_r["line_coverage"], 2),
                "chosen_line_cov": round(chosen_r["line_coverage"], 2),
            })

        progress.update(task, description="Done.")

    if total_funcs == 0:
        console.print("[red]No matched functions found.[/red]")
        return

    ml_api_saved = total_funcs - ml_llm_count
    ml_api_savings_pct = round(100.0 * ml_api_saved / total_funcs, 1)
    always_llm_lc = round(sum(always_llm_covs) / len(always_llm_covs), 2)
    always_tmpl_lc = round(sum(always_tmpl_covs) / len(always_tmpl_covs), 2)
    ml_lc = round(sum(ml_guided_covs) / len(ml_guided_covs), 2)

    comparison = {
        "total_functions": total_funcs,
        "always_llm": {
            "avg_line_coverage": always_llm_lc,
            "pass_rate": round(100.0 * always_llm_passed / total_funcs, 1),
            "api_calls": total_funcs, "api_calls_saved": 0,
        },
        "always_template": {
            "avg_line_coverage": always_tmpl_lc,
            "pass_rate": round(100.0 * always_tmpl_passed / total_funcs, 1),
            "api_calls": 0, "api_calls_saved": total_funcs,
        },
        "ml_guided": {
            "avg_line_coverage": ml_lc,
            "pass_rate": round(100.0 * ml_guided_passed / total_funcs, 1),
            "api_calls": ml_llm_count, "api_calls_saved": ml_api_saved,
            "api_savings_pct": ml_api_savings_pct,
            "coverage_delta_vs_always_llm": round(ml_lc - always_llm_lc, 2),
            "llm_chosen_count": ml_llm_count,
            "template_chosen_count": total_funcs - ml_llm_count,
        },
        "per_function_decisions": ml_decisions,
    }

    out_file = results_dir / "ml_comparison.json"
    with open(out_file, "w") as f:
        import json as _json2
        _json2.dump(comparison, f, indent=2)

    # Results table
    res_table = Table(title="3-Way Strategy Comparison", show_header=True)
    res_table.add_column("Metric", style="cyan", min_width=24)
    res_table.add_column("Always-LLM", style="red", justify="right")
    res_table.add_column("Always-Template", style="dim", justify="right")
    res_table.add_column("ML-Guided", style="green", justify="right")

    res_table.add_row("Avg Line Coverage",
                      f"{always_llm_lc:.1f}%", f"{always_tmpl_lc:.1f}%", f"{ml_lc:.1f}%")
    res_table.add_row("Pass Rate",
                      f"{comparison['always_llm']['pass_rate']:.1f}%",
                      f"{comparison['always_template']['pass_rate']:.1f}%",
                      f"{comparison['ml_guided']['pass_rate']:.1f}%")
    res_table.add_row("API Calls Used",
                      str(total_funcs), "0", str(ml_llm_count))
    res_table.add_row("API Calls Saved",
                      "0", str(total_funcs), str(ml_api_saved))
    res_table.add_row("API Savings %", "0%", "100%", f"{ml_api_savings_pct}%")
    console.print(res_table)

    delta = comparison["ml_guided"]["coverage_delta_vs_always_llm"]
    delta_str = f"[green]+{delta:.2f}[/green]" if delta >= 0 else f"[red]{delta:.2f}[/red]"
    console.print(f"\nCoverage delta (ML-guided vs always-LLM): {delta_str} pp")
    console.print(f"ML chose LLM for [bold]{ml_llm_count}/{total_funcs}[/bold] functions "
                  f"([bold]{100*ml_llm_count/total_funcs:.1f}%[/bold])")
    console.print(f"\n[bold green]Results saved to:[/bold green] {out_file}")

    # Generate ML charts
    artifact = selector.get_artifact()
    ml_viz_data = {
        "comparison": comparison,
        "feature_importances": selector.get_feature_importances().to_dict(orient="records"),
        "confusion_matrix_cv": artifact.get("confusion_matrix_cv_aggregate", [[0, 0], [0, 0]]),
        "cv_metrics": artifact.get("cv_metrics", {}),
    }
    create_ml_charts(ml_viz_data, graphs_dir)


if __name__ == '__main__':
    main()
