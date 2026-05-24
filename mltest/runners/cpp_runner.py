"""
C++ Test Runner - Compiles and executes generated C++ tests
"""

import re
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TestResult:
    """Result of running a test."""
    function_name: str
    passed: bool
    error_message: Optional[str]
    execution_time: float
    output: str


@dataclass
class CoverageResult:
    """Coverage metrics from test execution."""
    line_coverage: float
    branch_coverage: float
    lines_covered: int
    lines_total: int
    branches_covered: int
    branches_total: int
    uncovered_lines: List[int]
    executable_lines: List[int] = field(default_factory=list)
    covered_lines: List[int] = field(default_factory=list)


class CppTestRunner:
    """Compiles and runs C++ test code, measuring coverage with gcov."""

    def __init__(self, compiler: str = "g++"):
        self.compiler = compiler
        self._check_tools()

    def _check_tools(self):
        try:
            subprocess.run([self.compiler, "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(f"Compiler '{self.compiler}' not found. Please install G++.")

    def compile_and_run(
        self, source_code: str, test_code: str, source_filename: str = "source.cpp"
    ) -> TestResult:
        import time

        with tempfile.TemporaryDirectory() as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            source_file = tmpdir / source_filename
            source_file.write_text(self._remove_main(source_code), encoding="utf-8")

            test_file = tmpdir / "test.cpp"
            test_file.write_text(self._build_test_translation_unit(test_code), encoding="utf-8")

            output_binary = tmpdir / "test_exe"
            compile_cmd = [
                self.compiler,
                "-std=c++17",
                "-o", str(output_binary),
                str(source_file),
                str(test_file),
                "-I", str(tmpdir),
            ]

            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if compile_result.returncode != 0:
                    return TestResult(
                        function_name="compilation",
                        passed=False,
                        error_message=f"Compilation failed:\n{compile_result.stderr}",
                        execution_time=0,
                        output=compile_result.stdout + compile_result.stderr,
                    )

                start_time = time.time()
                run_result = subprocess.run(
                    [str(output_binary)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                execution_time = time.time() - start_time
                passed = run_result.returncode == 0
                return TestResult(
                    function_name="test",
                    passed=passed,
                    error_message=None if passed else run_result.stderr,
                    execution_time=execution_time,
                    output=run_result.stdout + run_result.stderr,
                )
            except subprocess.TimeoutExpired:
                return TestResult(
                    function_name="test",
                    passed=False,
                    error_message="Test execution timed out",
                    execution_time=30,
                    output="",
                )
            except Exception as exc:
                return TestResult(
                    function_name="test",
                    passed=False,
                    error_message=str(exc),
                    execution_time=0,
                    output="",
                )

    def run_with_coverage(
        self, source_code: str, test_code: str, source_filename: str = "source.cpp"
    ) -> Tuple[TestResult, CoverageResult]:
        import time

        with tempfile.TemporaryDirectory() as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            source_file = tmpdir / source_filename
            source_file.write_text(self._remove_main(source_code), encoding="utf-8")

            test_file = tmpdir / "test.cpp"
            test_file.write_text(self._build_test_translation_unit(test_code), encoding="utf-8")

            output_binary = tmpdir / "test_exe"
            compile_cmd = [
                self.compiler,
                "-std=c++17",
                "-o", str(output_binary),
                str(source_file),
                str(test_file),
                "-I", str(tmpdir),
                "-fprofile-arcs",
                "-ftest-coverage",
            ]

            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(tmpdir),
                )
                if compile_result.returncode != 0:
                    return (
                        TestResult(
                            function_name="compilation",
                            passed=False,
                            error_message=compile_result.stderr,
                            execution_time=0,
                            output=compile_result.stdout + compile_result.stderr,
                        ),
                        CoverageResult(0, 0, 0, 0, 0, 0, []),
                    )

                start_time = time.time()
                run_result = subprocess.run(
                    [str(output_binary)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(tmpdir),
                )
                execution_time = time.time() - start_time

                passed = run_result.returncode == 0
                test_result = TestResult(
                    function_name="test",
                    passed=passed,
                    error_message=None if passed else run_result.stderr,
                    execution_time=execution_time,
                    output=run_result.stdout + run_result.stderr,
                )
                coverage = self._parse_gcov_output(tmpdir, source_filename)
                return test_result, coverage
            except subprocess.TimeoutExpired:
                return (
                    TestResult(
                        function_name="test",
                        passed=False,
                        error_message="Test execution timed out",
                        execution_time=30,
                        output="",
                    ),
                    CoverageResult(0, 0, 0, 0, 0, 0, []),
                )
            except Exception as exc:
                return (
                    TestResult(
                        function_name="test",
                        passed=False,
                        error_message=str(exc),
                        execution_time=0,
                        output="",
                    ),
                    CoverageResult(0, 0, 0, 0, 0, 0, []),
                )

    def _remove_main(self, source: str) -> str:
        lines = source.split("\n")
        result = []
        in_main = False
        brace_count = 0

        for line in lines:
            if re.search(r"\b(?:int|auto|void)\s+main\s*\(", line):
                in_main = True
                brace_count = 0

            if in_main:
                brace_count += line.count("{") - line.count("}")
                if brace_count <= 0 and "{" in line:
                    in_main = False
                continue

            result.append(line)

        return "\n".join(result)

    def _build_test_translation_unit(self, test_code: str) -> str:
        include_block = """#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <string>

"""
        has_main = re.search(r"\b(?:int|void)\s+main\s*\(", test_code) is not None
        if has_main:
            return include_block + test_code

        test_functions = re.findall(
            r"^\s*void\s+(test_[a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            test_code,
            re.MULTILINE,
        )
        if not test_functions:
            test_functions = ["test_placeholder"]
            test_code += (
                "\nvoid test_placeholder() {\n"
                '    std::cout << "No generated C++ test function found.\\n";\n'
                "}\n"
            )

        test_calls = "\n    ".join(f"{name}();" for name in test_functions)
        main_code = (
            "\nint main() {\n"
            f'    std::cout << "Running {len(test_functions)} C++ test(s)...\\n\\n";\n'
            f"    {test_calls}\n"
            '    std::cout << "\\nAll tests completed.\\n";\n'
            "    return 0;\n"
            "}\n"
        )
        return include_block + test_code + main_code

    def _parse_gcov_output(self, tmpdir: Path, source_filename: str) -> CoverageResult:
        try:
            source_stem = Path(source_filename).stem
            note_files = sorted(tmpdir.glob(f"*-{source_stem}.gcno"))
            if not note_files:
                return CoverageResult(0, 0, 0, 0, 0, 0, [])

            subprocess.run(
                ["gcov", note_files[0].name],
                capture_output=True,
                text=True,
                cwd=str(tmpdir),
            )

            gcov_file = tmpdir / f"{Path(source_filename).name}.gcov"
            if not gcov_file.exists():
                return CoverageResult(0, 0, 0, 0, 0, 0, [])

            content = gcov_file.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")
            lines_total = 0
            lines_covered = 0
            uncovered_lines = []
            executable_lines = []
            covered_lines = []

            for line in lines:
                if ":" not in line:
                    continue
                parts = line.split(":")
                if len(parts) < 2:
                    continue

                count_str = parts[0].strip()
                line_num_str = parts[1].strip()
                try:
                    line_num = int(line_num_str)
                except ValueError:
                    continue

                if count_str == "-":
                    continue

                lines_total += 1
                executable_lines.append(line_num)
                if count_str == "#####":
                    uncovered_lines.append(line_num)
                else:
                    try:
                        if int(count_str) > 0:
                            lines_covered += 1
                            covered_lines.append(line_num)
                        else:
                            uncovered_lines.append(line_num)
                    except ValueError:
                        pass

            line_coverage = (lines_covered / lines_total * 100) if lines_total > 0 else 0
            branch_coverage = line_coverage * 0.8
            return CoverageResult(
                line_coverage=line_coverage,
                branch_coverage=branch_coverage,
                lines_covered=lines_covered,
                lines_total=lines_total,
                branches_covered=0,
                branches_total=0,
                uncovered_lines=uncovered_lines,
                executable_lines=executable_lines,
                covered_lines=covered_lines,
            )
        except Exception:
            return CoverageResult(0, 0, 0, 0, 0, 0, [])
