"""
MLTest CLI - Command-line interface for ML-driven test generation
"""

import click
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """MLTest - Machine Learning-Driven Unit Test Generator

    Generate comprehensive unit tests for C and Rust code using LLM technology.
    """
    pass


@main.command()
@click.argument('source_file', type=click.Path(exists=True))
@click.option('--language', '-l', type=click.Choice(['c', 'rust', 'auto']),
              default='auto', help='Source language (auto-detect by default)')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'template']),
              default='openai', help='LLM provider or template baseline')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output file for generated tests')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API key for LLM provider')
def generate(source_file: str, language: str, provider: str,
             output: Optional[str], api_key: Optional[str]):
    """Generate unit tests for a source file.

    Example: mltest generate math_utils.c -o test_math.c
    """
    source_path = Path(source_file)

    # Auto-detect language
    if language == 'auto':
        if source_path.suffix == '.c' or source_path.suffix == '.h':
            language = 'c'
        elif source_path.suffix == '.rs':
            language = 'rust'
        else:
            console.print("[red]Cannot auto-detect language. Please specify with -l[/red]")
            return

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
@click.option('--language', '-l', type=click.Choice(['c', 'rust', 'auto']),
              default='auto', help='Source language')
@click.option('--coverage/--no-coverage', default=True, help='Measure coverage')
def run(source_file: str, test_file: str, language: str, coverage: bool):
    """Run generated tests and measure coverage.

    Example: mltest run math_utils.c test_math.c
    """
    source_path = Path(source_file)
    test_path = Path(test_file)

    # Auto-detect language
    if language == 'auto':
        language = 'c' if source_path.suffix in ['.c', '.h'] else 'rust'

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
                result, cov = runner.run_with_coverage(source_code, test_code)
            else:
                result = runner.compile_and_run(source_code, test_code)
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
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'all']),
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
    rust_files = list(benchmark_path.glob('*.rs'))

    all_files = [(f, 'c') for f in c_files] + [(f, 'rust') for f in rust_files]

    if not all_files:
        console.print("[yellow]No source files found in benchmark directory.[/yellow]")
        return

    console.print(f"Found {len(all_files)} source file(s)")

    providers_to_test = ['openai', 'template'] if provider == 'all' else [provider, 'template']

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
                    result, cov = runner.run_with_coverage(source_code, test.test_code)
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
        parser.parse_source(source_code)
        return parser.get_testable_functions()
    else:
        from .parsers.rust_parser import RustParser
        parser = RustParser()
        parser.parse_source(source_code)
        return parser.get_testable_functions()


def _generate_test(func, language: str, provider: str, api_key: Optional[str]):
    """Generate test for a function"""
    if provider == 'template':
        from .generators.llm_generator import TemplateTestGenerator
        gen = TemplateTestGenerator()
    else:
        from .generators.llm_generator import LLMTestGenerator
        gen = LLMTestGenerator(provider=provider, api_key=api_key)

    if language == 'c':
        return gen.generate_c_tests(func)
    else:
        return gen.generate_rust_tests(func)


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

'''
        return header + '\n\n'.join(t.test_code for t in tests)
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


if __name__ == '__main__':
    main()
