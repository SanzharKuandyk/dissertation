"""
Rust Code Parser - Extracts function signatures and code context
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class RustFunction:
    """Represents a Rust function extracted from source code"""
    name: str
    visibility: str  # pub, pub(crate), etc.
    is_async: bool
    generics: Optional[str]
    parameters: List[tuple]  # (name, type) pairs
    return_type: Optional[str]
    body: str
    line_start: int
    line_end: int
    attributes: List[str]
    raw_signature: str

    def to_context(self) -> str:
        """Generate context string for LLM"""
        params_str = ", ".join(f"{n}: {t}" for n, t in self.parameters)
        ret_str = f" -> {self.return_type}" if self.return_type else ""
        async_str = "async " if self.is_async else ""

        return f"""
Function: {self.name}
Signature: {self.visibility} {async_str}fn {self.name}({params_str}){ret_str}
Attributes: {', '.join(self.attributes) if self.attributes else 'None'}
Body:
{self.body}
"""


class RustParser:
    """Parser for Rust source code to extract function information"""

    # Regex patterns for Rust parsing
    USE_PATTERN = re.compile(r'use\s+([^;]+);')

    # Function pattern for Rust
    FUNC_PATTERN = re.compile(
        r'(?P<attrs>(?:#\[[^\]]+\]\s*)*)'  # Attributes
        r'(?P<vis>pub(?:\s*\([^)]+\))?\s+)?'  # Visibility
        r'(?P<async>async\s+)?'  # Async
        r'fn\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)'  # Function name
        r'(?P<generics><[^>]+>)?'  # Generics
        r'\s*\((?P<params>[^)]*)\)'  # Parameters
        r'(?:\s*->\s*(?P<return>[^{]+?))?'  # Return type
        r'\s*\{',
        re.MULTILINE | re.DOTALL
    )

    def __init__(self):
        self.functions: List[RustFunction] = []
        self.uses: List[str] = []

    def parse_file(self, filepath: Path) -> List[RustFunction]:
        """Parse a Rust file and extract all functions"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return self.parse_source(content)

    def parse_source(self, source: str) -> List[RustFunction]:
        """Parse Rust source code string and extract functions"""
        self.functions = []

        # Extract use statements
        self.uses = self.USE_PATTERN.findall(source)

        # Find all function definitions
        for match in self.FUNC_PATTERN.finditer(source):
            attrs_str = match.group('attrs') or ''
            visibility = (match.group('vis') or '').strip()
            is_async = bool(match.group('async'))
            name = match.group('name')
            generics = match.group('generics')
            params_str = match.group('params')
            return_type = (match.group('return') or '').strip() or None

            # Skip test functions and main
            if name in ['main'] or '#[test]' in attrs_str:
                continue

            # Parse attributes
            attributes = re.findall(r'#\[([^\]]+)\]', attrs_str)

            # Parse parameters
            parameters = self._parse_parameters(params_str)

            # Find function body
            body_start = match.end() - 1
            body, body_end = self._extract_body(source, body_start)

            # Calculate line numbers
            line_start = source[:match.start()].count('\n') + 1
            line_end = source[:body_end].count('\n') + 1

            # Build raw signature
            async_str = "async " if is_async else ""
            gen_str = generics or ""
            ret_str = f" -> {return_type}" if return_type else ""
            raw_sig = f"{visibility} {async_str}fn {name}{gen_str}({params_str}){ret_str}".strip()

            func = RustFunction(
                name=name,
                visibility=visibility,
                is_async=is_async,
                generics=generics,
                parameters=parameters,
                return_type=return_type,
                body=body,
                line_start=line_start,
                line_end=line_end,
                attributes=attributes,
                raw_signature=raw_sig
            )
            self.functions.append(func)

        return self.functions

    def _parse_parameters(self, params_str: str) -> List[tuple]:
        """Parse parameter string into list of (name, type) tuples"""
        params = []
        if not params_str.strip():
            return params

        # Handle self parameter
        params_str = params_str.strip()
        if params_str.startswith('&mut self') or params_str.startswith('&self') or params_str.startswith('self'):
            # Extract self variant
            if params_str.startswith('&mut self'):
                params.append(('self', '&mut Self'))
                params_str = params_str[9:].lstrip(',').strip()
            elif params_str.startswith('&self'):
                params.append(('self', '&Self'))
                params_str = params_str[5:].lstrip(',').strip()
            elif params_str.startswith('self'):
                params.append(('self', 'Self'))
                params_str = params_str[4:].lstrip(',').strip()

        if not params_str:
            return params

        # Parse remaining parameters
        # This is simplified - full Rust parsing would need to handle nested generics
        depth = 0
        current_param = ""

        for char in params_str:
            if char in '<([':
                depth += 1
                current_param += char
            elif char in '>)]':
                depth -= 1
                current_param += char
            elif char == ',' and depth == 0:
                self._add_param(current_param.strip(), params)
                current_param = ""
            else:
                current_param += char

        if current_param.strip():
            self._add_param(current_param.strip(), params)

        return params

    def _add_param(self, param: str, params: List[tuple]):
        """Parse a single parameter and add to list"""
        if ':' in param:
            name, type_str = param.split(':', 1)
            params.append((name.strip(), type_str.strip()))

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

    def get_testable_functions(self) -> List[RustFunction]:
        """Get functions that are suitable for unit testing"""
        testable = []
        for func in self.functions:
            # Focus on public functions
            if not func.visibility.startswith('pub'):
                continue
            # Skip functions that only have &self (methods without inputs)
            if func.parameters == [('self', '&Self')] or func.parameters == [('self', '&mut Self')]:
                continue
            testable.append(func)
        return testable


if __name__ == "__main__":
    # Test the parser
    test_code = '''
use std::collections::HashMap;

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub async fn fetch_data(url: &str) -> Result<String, Error> {
    // Implementation
    Ok(String::new())
}

fn private_helper(x: i32) -> i32 {
    x * 2
}

impl Calculator {
    pub fn new() -> Self {
        Self { value: 0 }
    }

    pub fn add(&mut self, x: i32) {
        self.value += x;
    }
}
'''
    parser = RustParser()
    functions = parser.parse_source(test_code)
    for func in functions:
        print(f"Found: {func.raw_signature}")
        print(f"  Params: {func.parameters}")
        print(f"  Return: {func.return_type}")
        print()
