"""
Coverage Analyzer - Aggregates and analyzes coverage metrics
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class FunctionCoverage:
    """Coverage data for a single function"""
    function_name: str
    language: str
    line_coverage: float
    branch_coverage: float
    lines_covered: int
    lines_total: int
    test_passed: bool
    execution_time: float
    generator_type: str  # 'llm' or 'template'


@dataclass
class BenchmarkResult:
    """Result of running a benchmark"""
    benchmark_name: str
    language: str
    functions_tested: int
    total_line_coverage: float
    total_branch_coverage: float
    tests_passed: int
    tests_failed: int
    total_execution_time: float
    generator_type: str
    timestamp: str
    function_results: List[FunctionCoverage]


class CoverageAnalyzer:
    """Aggregates coverage data and computes statistics"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks: List[BenchmarkResult] = []

    def add_function_result(self, benchmark_name: str, language: str,
                           function_coverage: FunctionCoverage):
        """Add a single function result to the current benchmark"""
        # Find or create benchmark
        benchmark = None
        for b in self.benchmarks:
            if (b.benchmark_name == benchmark_name and
                b.generator_type == function_coverage.generator_type and
                b.language == language):
                benchmark = b
                break

        if benchmark is None:
            benchmark = BenchmarkResult(
                benchmark_name=benchmark_name,
                language=language,
                functions_tested=0,
                total_line_coverage=0,
                total_branch_coverage=0,
                tests_passed=0,
                tests_failed=0,
                total_execution_time=0,
                generator_type=function_coverage.generator_type,
                timestamp=datetime.now().isoformat(),
                function_results=[]
            )
            self.benchmarks.append(benchmark)

        benchmark.function_results.append(function_coverage)
        benchmark.functions_tested += 1
        if function_coverage.test_passed:
            benchmark.tests_passed += 1
        else:
            benchmark.tests_failed += 1
        benchmark.total_execution_time += function_coverage.execution_time

    def compute_aggregate_metrics(self) -> Dict:
        """Compute aggregate metrics across all benchmarks"""
        for benchmark in self.benchmarks:
            if benchmark.function_results:
                total_lines_covered = sum(f.lines_covered for f in benchmark.function_results)
                total_lines = sum(f.lines_total for f in benchmark.function_results)
                benchmark.total_line_coverage = (total_lines_covered / total_lines * 100) if total_lines > 0 else 0

                avg_branch = sum(f.branch_coverage for f in benchmark.function_results) / len(benchmark.function_results)
                benchmark.total_branch_coverage = avg_branch

        return {
            'benchmarks': [asdict(b) for b in self.benchmarks],
            'summary': self._compute_summary()
        }

    def _compute_summary(self) -> Dict:
        """Compute summary statistics"""
        if not self.benchmarks:
            return {}

        llm_benchmarks = [b for b in self.benchmarks if b.generator_type == 'llm']
        template_benchmarks = [b for b in self.benchmarks if b.generator_type == 'template']

        summary = {
            'total_benchmarks': len(self.benchmarks),
            'total_functions_tested': sum(b.functions_tested for b in self.benchmarks),
        }

        if llm_benchmarks:
            summary['llm'] = {
                'avg_line_coverage': sum(b.total_line_coverage for b in llm_benchmarks) / len(llm_benchmarks),
                'avg_branch_coverage': sum(b.total_branch_coverage for b in llm_benchmarks) / len(llm_benchmarks),
                'total_tests_passed': sum(b.tests_passed for b in llm_benchmarks),
                'total_tests_failed': sum(b.tests_failed for b in llm_benchmarks),
                'pass_rate': sum(b.tests_passed for b in llm_benchmarks) /
                            (sum(b.tests_passed + b.tests_failed for b in llm_benchmarks) or 1) * 100
            }

        if template_benchmarks:
            summary['template'] = {
                'avg_line_coverage': sum(b.total_line_coverage for b in template_benchmarks) / len(template_benchmarks),
                'avg_branch_coverage': sum(b.total_branch_coverage for b in template_benchmarks) / len(template_benchmarks),
                'total_tests_passed': sum(b.tests_passed for b in template_benchmarks),
                'total_tests_failed': sum(b.tests_failed for b in template_benchmarks),
                'pass_rate': sum(b.tests_passed for b in template_benchmarks) /
                            (sum(b.tests_passed + b.tests_failed for b in template_benchmarks) or 1) * 100
            }

        # Improvement metrics (LLM vs Template)
        if llm_benchmarks and template_benchmarks:
            summary['improvement'] = {
                'line_coverage_improvement': summary['llm']['avg_line_coverage'] - summary['template']['avg_line_coverage'],
                'branch_coverage_improvement': summary['llm']['avg_branch_coverage'] - summary['template']['avg_branch_coverage'],
                'pass_rate_improvement': summary['llm']['pass_rate'] - summary['template']['pass_rate']
            }

        return summary

    def save_results(self, filename: str = "results.json"):
        """Save all results to JSON file"""
        results = self.compute_aggregate_metrics()
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        return output_path

    def load_results(self, filename: str = "results.json"):
        """Load results from JSON file"""
        input_path = self.results_dir / filename
        if input_path.exists():
            with open(input_path) as f:
                data = json.load(f)
            # Reconstruct benchmark objects
            self.benchmarks = []
            for b_data in data.get('benchmarks', []):
                func_results = [FunctionCoverage(**f) for f in b_data.pop('function_results', [])]
                benchmark = BenchmarkResult(**b_data, function_results=func_results)
                self.benchmarks.append(benchmark)
            return data
        return None

    def get_comparison_data(self) -> Dict:
        """Get data formatted for comparison charts"""
        llm_data = {'line_coverage': [], 'branch_coverage': [], 'functions': []}
        template_data = {'line_coverage': [], 'branch_coverage': [], 'functions': []}

        for benchmark in self.benchmarks:
            for func in benchmark.function_results:
                target = llm_data if benchmark.generator_type == 'llm' else template_data
                target['line_coverage'].append(func.line_coverage)
                target['branch_coverage'].append(func.branch_coverage)
                target['functions'].append(func.function_name)

        return {
            'llm': llm_data,
            'template': template_data,
            'benchmarks': [b.benchmark_name for b in self.benchmarks]
        }

    def generate_report(self) -> str:
        """Generate a text report of results"""
        metrics = self.compute_aggregate_metrics()
        summary = metrics.get('summary', {})

        report = []
        report.append("=" * 60)
        report.append("ML-DRIVEN TEST GENERATION - EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        report.append(f"Total Benchmarks: {summary.get('total_benchmarks', 0)}")
        report.append(f"Total Functions Tested: {summary.get('total_functions_tested', 0)}")
        report.append("")

        if 'llm' in summary:
            report.append("-" * 40)
            report.append("LLM-GENERATED TESTS")
            report.append("-" * 40)
            report.append(f"  Average Line Coverage: {summary['llm']['avg_line_coverage']:.1f}%")
            report.append(f"  Average Branch Coverage: {summary['llm']['avg_branch_coverage']:.1f}%")
            report.append(f"  Tests Passed: {summary['llm']['total_tests_passed']}")
            report.append(f"  Tests Failed: {summary['llm']['total_tests_failed']}")
            report.append(f"  Pass Rate: {summary['llm']['pass_rate']:.1f}%")
            report.append("")

        if 'template' in summary:
            report.append("-" * 40)
            report.append("TEMPLATE-GENERATED TESTS (BASELINE)")
            report.append("-" * 40)
            report.append(f"  Average Line Coverage: {summary['template']['avg_line_coverage']:.1f}%")
            report.append(f"  Average Branch Coverage: {summary['template']['avg_branch_coverage']:.1f}%")
            report.append(f"  Tests Passed: {summary['template']['total_tests_passed']}")
            report.append(f"  Tests Failed: {summary['template']['total_tests_failed']}")
            report.append(f"  Pass Rate: {summary['template']['pass_rate']:.1f}%")
            report.append("")

        if 'improvement' in summary:
            report.append("-" * 40)
            report.append("IMPROVEMENT (LLM vs Template)")
            report.append("-" * 40)
            report.append(f"  Line Coverage: +{summary['improvement']['line_coverage_improvement']:.1f}%")
            report.append(f"  Branch Coverage: +{summary['improvement']['branch_coverage_improvement']:.1f}%")
            report.append(f"  Pass Rate: +{summary['improvement']['pass_rate_improvement']:.1f}%")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
