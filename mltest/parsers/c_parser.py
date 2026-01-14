"""
C Code Parser - Extracts function signatures and code context
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class CFunction:
    """Represents a C function extracted from source code"""
    name: str
    return_type: str
    parameters: List[tuple]  # (type, name) pairs
    body: str
    line_start: int
    line_end: int
    includes: List[str]
    raw_signature: str

    def to_context(self) -> str:
        """Generate context string for LLM"""
        params_str = ", ".join(f"{t} {n}" for t, n in self.parameters)
        return f"""
Function: {self.name}
Signature: {self.return_type} {self.name}({params_str})
Body:
{self.body}

Required includes: {', '.join(self.includes) if self.includes else 'None'}
"""


class CParser:
    """Parser for C source code to extract function information"""

    # Regex patterns for C parsing
    INCLUDE_PATTERN = re.compile(r'#include\s*[<"]([^>"]+)[>"]')

    # Function pattern: return_type function_name(params) { body }
    FUNC_PATTERN = re.compile(
        r'(?P<return_type>(?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?'
        r'(?:static\s+)?(?:inline\s+)?'
        r'[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\*)*)\s+'
        r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*'
        r'\((?P<params>[^)]*)\)\s*'
        r'\{',
        re.MULTILINE
    )

    PARAM_PATTERN = re.compile(
        r'(?P<type>(?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?'
        r'[a-zA-Z_][a-zA-Z0-9_]*(?:\s*\*)*)\s+'
        r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)'
    )

    def __init__(self):
        self.functions: List[CFunction] = []
        self.includes: List[str] = []

    def parse_file(self, filepath: Path) -> List[CFunction]:
        """Parse a C file and extract all functions"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return self.parse_source(content)

    def parse_source(self, source: str) -> List[CFunction]:
        """Parse C source code string and extract functions"""
        self.functions = []
        lines = source.split('\n')

        # Extract includes
        self.includes = self.INCLUDE_PATTERN.findall(source)

        # Find all function definitions
        for match in self.FUNC_PATTERN.finditer(source):
            return_type = match.group('return_type').strip()
            name = match.group('name')
            params_str = match.group('params')

            # Skip main function and static helpers we don't want to test
            if name in ['main']:
                continue

            # Skip C keywords that aren't valid function names
            if name in ['if', 'while', 'for', 'switch', 'return', 'else', 'do']:
                continue

            # Skip invalid return types (comments parsed as types)
            invalid_return_types = ['overflow', 'underflow', 'multiplication', 'division',
                                   'addition', 'subtraction', 'check', 'error', 'success']
            if return_type.lower() in invalid_return_types:
                continue

            # Parse parameters
            parameters = self._parse_parameters(params_str)

            # Find function body
            body_start = match.end() - 1  # Position of opening brace
            body, body_end = self._extract_body(source, body_start)

            # Calculate line numbers
            line_start = source[:match.start()].count('\n') + 1
            line_end = source[:body_end].count('\n') + 1

            # Build raw signature
            raw_sig = f"{return_type} {name}({params_str})"

            func = CFunction(
                name=name,
                return_type=return_type,
                parameters=parameters,
                body=body,
                line_start=line_start,
                line_end=line_end,
                includes=self.includes.copy(),
                raw_signature=raw_sig
            )
            self.functions.append(func)

        return self.functions

    def _parse_parameters(self, params_str: str) -> List[tuple]:
        """Parse parameter string into list of (type, name) tuples"""
        params = []
        if not params_str.strip() or params_str.strip() == 'void':
            return params

        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue
            match = self.PARAM_PATTERN.search(param)
            if match:
                params.append((match.group('type').strip(), match.group('name')))
            else:
                # Handle cases like "int x" directly
                parts = param.rsplit(None, 1)
                if len(parts) == 2:
                    params.append((parts[0], parts[1]))

        return params

    def _extract_body(self, source: str, start_pos: int) -> tuple:
        """Extract function body by matching braces"""
        brace_count = 0
        pos = start_pos
        body_start = start_pos + 1

        while pos < len(source):
            char = source[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return source[body_start:pos].strip(), pos
            pos += 1

        return source[body_start:].strip(), len(source)

    def get_testable_functions(self) -> List[CFunction]:
        """Get functions that are suitable for unit testing"""
        testable = []
        for func in self.functions:
            # Skip static functions (internal linkage)
            if 'static' in func.return_type:
                continue
            # Skip void functions with no parameters (likely side-effect only)
            if func.return_type == 'void' and not func.parameters:
                continue
            testable.append(func)
        return testable


if __name__ == "__main__":
    # Test the parser
    test_code = '''
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    return a + b;
}

char* concat(const char* s1, const char* s2) {
    // Implementation
    return NULL;
}

static int helper(int x) {
    return x * 2;
}
'''
    parser = CParser()
    functions = parser.parse_source(test_code)
    for func in functions:
        print(f"Found: {func.raw_signature}")
        print(f"  Params: {func.parameters}")
        print()
