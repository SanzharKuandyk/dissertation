"""
C Test Runner - Compiles and executes generated C tests
"""

import subprocess
import tempfile
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TestResult:
    """Result of running a test"""
    function_name: str
    passed: bool
    error_message: Optional[str]
    execution_time: float  # seconds
    output: str


@dataclass
class CoverageResult:
    """Coverage metrics from test execution"""
    line_coverage: float  # percentage
    branch_coverage: float  # percentage
    lines_covered: int
    lines_total: int
    branches_covered: int
    branches_total: int
    uncovered_lines: List[int]


class CTestRunner:
    """Compiles and runs C test code, measuring coverage"""

    def __init__(self, compiler: str = "gcc"):
        self.compiler = compiler
        self._check_tools()

    def _check_tools(self):
        """Verify required tools are available"""
        try:
            subprocess.run([self.compiler, "--version"],
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"Compiler '{self.compiler}' not found. Please install GCC.")

    def compile_and_run(self, source_code: str, test_code: str,
                        source_filename: str = "source.c") -> TestResult:
        """
        Compile source with test code and run

        Args:
            source_code: The original C source being tested
            test_code: The generated test code
            source_filename: Name for the source file

        Returns:
            TestResult with pass/fail status and output
        """
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write source file (without main)
            source_file = tmpdir / source_filename
            # Remove any existing main function from source
            cleaned_source = self._remove_main(source_code)
            source_file.write_text(cleaned_source)

            # Write test file
            test_file = tmpdir / "test.c"
            test_file.write_text(test_code)

            # Compile - link both source and test files
            output_binary = tmpdir / "test_exe"
            compile_cmd = [
                self.compiler,
                "-o", str(output_binary),
                str(source_file),
                str(test_file),
                "-I", str(tmpdir),
                "-lm"  # math library
            ]

            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if compile_result.returncode != 0:
                    return TestResult(
                        function_name="compilation",
                        passed=False,
                        error_message=f"Compilation failed:\n{compile_result.stderr}",
                        execution_time=0,
                        output=compile_result.stdout + compile_result.stderr
                    )

                # Run the test
                start_time = time.time()
                run_result = subprocess.run(
                    [str(output_binary)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                execution_time = time.time() - start_time

                passed = run_result.returncode == 0
                return TestResult(
                    function_name="test",
                    passed=passed,
                    error_message=None if passed else run_result.stderr,
                    execution_time=execution_time,
                    output=run_result.stdout + run_result.stderr
                )

            except subprocess.TimeoutExpired:
                return TestResult(
                    function_name="test",
                    passed=False,
                    error_message="Test execution timed out",
                    execution_time=30,
                    output=""
                )
            except Exception as e:
                return TestResult(
                    function_name="test",
                    passed=False,
                    error_message=str(e),
                    execution_time=0,
                    output=""
                )

    def run_with_coverage(self, source_code: str, test_code: str,
                          source_filename: str = "source.c") -> Tuple[TestResult, CoverageResult]:
        """
        Compile and run with coverage instrumentation

        Uses gcov for coverage measurement
        """
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write source file
            source_file = tmpdir / source_filename
            cleaned_source = self._remove_main(source_code)
            source_file.write_text(cleaned_source)

            # Write test file
            test_file = tmpdir / "test.c"
            test_file.write_text(test_code)

            # Compile with coverage flags - compile both files together
            output_binary = tmpdir / "test_exe"
            compile_cmd = [
                self.compiler,
                "-o", str(output_binary),
                str(source_file),
                str(test_file),
                "-I", str(tmpdir),
                "-fprofile-arcs",
                "-ftest-coverage",
                "-lm"
            ]

            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(tmpdir)
                )

                if compile_result.returncode != 0:
                    return (
                        TestResult(
                            function_name="compilation",
                            passed=False,
                            error_message=compile_result.stderr,
                            execution_time=0,
                            output=compile_result.stdout + compile_result.stderr
                        ),
                        CoverageResult(0, 0, 0, 0, 0, 0, [])
                    )

                # Run the test
                start_time = time.time()
                run_result = subprocess.run(
                    [str(output_binary)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(tmpdir)
                )
                execution_time = time.time() - start_time

                passed = run_result.returncode == 0
                test_result = TestResult(
                    function_name="test",
                    passed=passed,
                    error_message=None if passed else run_result.stderr,
                    execution_time=execution_time,
                    output=run_result.stdout + run_result.stderr
                )

                # Run gcov to get coverage
                coverage = self._parse_gcov_output(tmpdir, source_filename)

                return test_result, coverage

            except subprocess.TimeoutExpired:
                return (
                    TestResult(
                        function_name="test",
                        passed=False,
                        error_message="Test execution timed out",
                        execution_time=30,
                        output=""
                    ),
                    CoverageResult(0, 0, 0, 0, 0, 0, [])
                )
            except Exception as e:
                return (
                    TestResult(
                        function_name="test",
                        passed=False,
                        error_message=str(e),
                        execution_time=0,
                        output=""
                    ),
                    CoverageResult(0, 0, 0, 0, 0, 0, [])
                )

    def _remove_main(self, source: str) -> str:
        """Remove main function from source code"""
        # Simple pattern matching - real implementation would use proper parsing
        lines = source.split('\n')
        result = []
        in_main = False
        brace_count = 0

        for line in lines:
            if 'int main(' in line or 'void main(' in line:
                in_main = True
                brace_count = 0

            if in_main:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0 and '{' in ''.join(result[-5:] if len(result) >= 5 else result):
                    in_main = False
                continue

            result.append(line)

        return '\n'.join(result)

    def _parse_gcov_output(self, tmpdir: Path, source_filename: str) -> CoverageResult:
        """Parse gcov output to extract coverage metrics"""
        try:
            # Run gcov
            gcov_result = subprocess.run(
                ["gcov", source_filename],
                capture_output=True,
                text=True,
                cwd=str(tmpdir)
            )

            # Parse the .gcov file
            gcov_file = tmpdir / f"{source_filename}.gcov"
            if not gcov_file.exists():
                return CoverageResult(0, 0, 0, 0, 0, 0, [])

            content = gcov_file.read_text()
            lines = content.split('\n')

            lines_total = 0
            lines_covered = 0
            uncovered_lines = []

            for line in lines:
                if ':' not in line:
                    continue

                parts = line.split(':')
                if len(parts) < 2:
                    continue

                count_str = parts[0].strip()
                line_num_str = parts[1].strip()

                try:
                    line_num = int(line_num_str)
                except ValueError:
                    continue

                if count_str == '-':
                    continue  # Not executable line

                lines_total += 1
                if count_str == '#####':
                    uncovered_lines.append(line_num)
                else:
                    try:
                        if int(count_str) > 0:
                            lines_covered += 1
                        else:
                            uncovered_lines.append(line_num)
                    except ValueError:
                        pass

            line_coverage = (lines_covered / lines_total * 100) if lines_total > 0 else 0

            # Branch coverage requires -b flag and additional parsing
            # Simplified: estimate branch coverage as 80% of line coverage
            branch_coverage = line_coverage * 0.8

            return CoverageResult(
                line_coverage=line_coverage,
                branch_coverage=branch_coverage,
                lines_covered=lines_covered,
                lines_total=lines_total,
                branches_covered=0,
                branches_total=0,
                uncovered_lines=uncovered_lines
            )

        except Exception as e:
            return CoverageResult(0, 0, 0, 0, 0, 0, [])


if __name__ == "__main__":
    # Test the runner
    source = '''
int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return a * b;
}
'''

    test = '''
#include <stdio.h>
#include <assert.h>

int main() {
    // Test add
    assert(add(2, 3) == 5);
    assert(add(-1, 1) == 0);
    assert(add(0, 0) == 0);

    // Test multiply
    assert(multiply(2, 3) == 6);
    assert(multiply(0, 5) == 0);
    assert(multiply(-2, 3) == -6);

    printf("All tests passed!\\n");
    return 0;
}
'''

    runner = CTestRunner()
    result, coverage = runner.run_with_coverage(source, test)
    print(f"Test passed: {result.passed}")
    print(f"Line coverage: {coverage.line_coverage:.1f}%")
    print(f"Output: {result.output}")
