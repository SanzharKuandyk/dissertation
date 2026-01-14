"""
LLM-Based Test Generator - Uses LLM APIs to generate unit tests
"""

import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

from ..parsers.c_parser import CFunction
from ..parsers.rust_parser import RustFunction

C_KEYWORDS = {
    "if", "else", "for", "while", "switch", "case", "return",
    "break", "continue", "default", "goto", "sizeof", "do",
    "static", "struct", "union", "typedef", "enum", "const",
    "volatile", "extern", "register", "inline", "restrict"
}


def safe_c_name(name: str) -> str:
    return f"{name}_fn" if name in C_KEYWORDS else name


@dataclass
class GeneratedTest:
    """Represents a generated unit test"""
    function_name: str
    test_code: str
    test_name: str
    language: str
    description: str
    edge_cases_covered: List[str]


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except ImportError:
            raise RuntimeError("OpenAI package not installed. Run: pip install openai")
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model

    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except ImportError:
            raise RuntimeError("Anthropic package not installed. Run: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


class LLMTestGenerator:
    """Generates unit tests using LLM APIs"""

    C_SYSTEM_PROMPT = """You are an expert C programmer and testing specialist.
Your task is to generate comprehensive unit tests for C functions.

CRITICAL RULES:
- Return ONLY valid C code, no markdown, no explanations
- DO NOT include headers (#include) - they will be added separately
- DO NOT redefine the function being tested
- DO NOT create a main() function
- ONLY write: forward declaration + ONE test function
- Use assert() for tests and printf() for output
- Follow the exact format shown in the example
- Function names MUST be valid C identifiers and MUST NOT be C keywords
- Do NOT invent new syntax or pseudo-constructs (e.g., "overflow if(...)")

ABSOLUTE PROHIBITIONS:
- NEVER invent new keywords, types, macros, or labels
- NEVER write pseudo-code or explanatory constructs
- NEVER write lines like:
    overflow(...)
    multiplication(...)
    division(...)
    addition(...)
- If checking overflow or conditions, use ONLY assert(expression)

Output: VALID C CODE ONLY. No explanations. No comments outside code."""

    RUST_SYSTEM_PROMPT = """You are an expert Rust programmer and testing specialist.
Your task is to generate comprehensive unit tests for Rust functions.

Guidelines:
- Generate tests using Rust's built-in #[test] attribute
- Include tests for normal cases, edge cases, and boundary conditions
- Test Result types properly with is_ok(), is_err()
- Test Option types with is_some(), is_none()
- Use assert!, assert_eq!, assert_ne! appropriately
- Include comments explaining each test case
- Generate valid, compilable Rust test code

Output format: Return ONLY the test code, no explanations."""

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize the test generator

        Args:
            provider: "openai" or "anthropic"
            api_key: API key (or set via environment variable)
        """
        if provider == "openai":
            self.llm = OpenAIProvider(api_key)
        elif provider == "anthropic":
            self.llm = AnthropicProvider(api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.provider_name = provider

    def generate_c_tests(self, function: CFunction) -> GeneratedTest:
        safe_name = safe_c_name(function.name)

        prompt = self._build_c_prompt(function, safe_name)

        response = self.llm.generate(prompt, self.C_SYSTEM_PROMPT)
        test_code = self._extract_code(response, "c")

        # Validate generated code
        test_code = self._validate_and_clean_c_code(test_code, safe_name)

        return GeneratedTest(
            function_name=safe_name,
            test_code=test_code,
            test_name=f"test_{safe_name}",
            language="c",
            description=f"Generated tests for {function.raw_signature}",
            edge_cases_covered=self._identify_edge_cases(function, "c")
        )

    def generate_rust_tests(self, function: RustFunction) -> GeneratedTest:
        """Generate unit tests for a Rust function"""
        prompt = self._build_rust_prompt(function)

        response = self.llm.generate(prompt, self.RUST_SYSTEM_PROMPT)
        test_code = self._extract_code(response, "rust")

        return GeneratedTest(
            function_name=function.name,
            test_code=test_code,
            test_name=f"test_{function.name}",
            language="rust",
            description=f"Generated tests for {function.raw_signature}",
            edge_cases_covered=self._identify_edge_cases(function, "rust")
        )

    def _build_c_prompt(self, function: CFunction, safe_name: str) -> str:
        """Build prompt for C test generation"""
        return f"""Generate unit tests for this C function:

{function.to_context()}

EXACT FORMAT TO FOLLOW:
```c
// Forward declaration
{function.raw_signature.replace(function.name, safe_name)};

void test_{safe_name}() {{
    printf("Testing {safe_name}...\\n");

    // Write assert() statements to test the function
    assert({safe_name}(/* test values */) == /* expected */);

    printf("  All tests passed\\n");
}}
```

RULES:
- NO headers, NO main(), NO function implementations
- ONLY: forward declaration + test_{safe_name}() function
- Use ONLY assert() and printf()
- Test normal cases, edge cases (0, INT_MAX, INT_MIN, NULL if applicable)
- NO pseudo-code like "overflow(...)", "multiplication(...)", etc.
- Write ONLY valid C code that compiles

Generate the code now:
"""

    def _build_rust_prompt(self, function: RustFunction) -> str:
        """Build prompt for Rust test generation"""
        return f"""Generate comprehensive unit tests for the following Rust function:

{function.to_context()}

Requirements:
1. Create a test module with #[cfg(test)]
2. Test normal operation with typical inputs
3. Test edge cases (empty strings, zero values, etc.)
4. Test boundary conditions
5. If the function returns Result or Option, test both success and failure paths
6. Use appropriate assertion macros

The tests should be runnable with: cargo test
"""

    def _extract_code(self, response: str, language: str) -> str:
        """Extract code from LLM response, handling markdown code blocks"""
        # Check for markdown code blocks
        if "```" in response:
            # Extract code between backticks
            lines = response.split('\n')
            in_code = False
            code_lines = []

            for line in lines:
                if line.strip().startswith('```') and not in_code:
                    in_code = True
                    continue
                elif line.strip().startswith('```') and in_code:
                    in_code = False
                    continue
                elif in_code:
                    code_lines.append(line)

            extracted = '\n'.join(code_lines)
            if extracted.strip():
                return extracted

        # No code blocks, return as-is
        result = response.strip()

        # Clean up common issues
        # Remove explanatory text before/after code
        if language == 'c':
            # Find the first line that looks like C code
            lines = result.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('//') or line.strip().startswith('/*') or \
                   'void test_' in line or line.strip().endswith(';'):
                    start_idx = i
                    break
            result = '\n'.join(lines[start_idx:])

        return result

    def _validate_and_clean_c_code(self, code: str, func_name: str) -> str:
        """Validate and clean generated C code"""
        lines = code.split('\n')
        cleaned_lines = []

        invalid_patterns = [
            'overflow ', 'underflow ', 'multiplication ', 'division ',
            'addition ', 'subtraction ', 'modulo ', 'result '
        ]

        found_test_func = False
        in_test_func = False
        brace_count = 0

        for line in lines:
            stripped = line.strip()

            # Skip lines with invalid pseudo-code patterns
            is_invalid = False
            if not stripped.startswith('//') and not stripped.startswith('/*'):
                for pattern in invalid_patterns:
                    if stripped.startswith(pattern) and '(' in stripped:
                        is_invalid = True
                        break

            # Track if we're inside the correct test function
            if f'void test_{func_name}' in line:
                if found_test_func:
                    # Already found one test function, skip duplicates
                    continue
                found_test_func = True
                in_test_func = True
                brace_count = 0

            # Track braces to know when test function ends
            if in_test_func:
                brace_count += line.count('{') - line.count('}')
                if not is_invalid:
                    cleaned_lines.append(line)
                if brace_count <= 0 and found_test_func:
                    in_test_func = False
            elif not in_test_func and not is_invalid:
                # Only add lines outside test functions if they're valid
                # This includes forward declarations
                if stripped and (stripped.startswith('//') or
                               stripped.startswith('/*') or
                               stripped.endswith(';')):
                    cleaned_lines.append(line)

        cleaned = '\n'.join(cleaned_lines)

        # Ensure we have the test function
        if f'void test_{func_name}' not in cleaned:
            # If validation failed completely, return a minimal valid test
            return f'''// Forward declaration
extern int {func_name}();

void test_{func_name}() {{
    printf("Testing {func_name}...\\n");
    // LLM generated invalid code, using placeholder
    printf("  Test placeholder\\n");
}}
'''

        # Ensure we have only ONE forward declaration for this function
        # Remove duplicate or wrong declarations
        final_lines = []
        seen_declarations = set()

        for line in cleaned.split('\n'):
            stripped = line.strip()
            # Check if it's a forward declaration (ends with ;, not in function)
            if stripped.endswith(');') and not 'void test_' in stripped:
                if func_name in stripped:
                    # This is a declaration for our function
                    if func_name not in seen_declarations:
                        seen_declarations.add(func_name)
                        final_lines.append(line)
                # Skip declarations for other functions
            else:
                final_lines.append(line)

        return '\n'.join(final_lines)

    def _identify_edge_cases(self, function: Union[CFunction, RustFunction], language: str) -> List[str]:
        """Identify potential edge cases based on function signature"""
        edge_cases = []

        if language == "c":
            for param_type, param_name in function.parameters:
                if '*' in param_type:
                    edge_cases.append(f"NULL {param_name}")
                if 'int' in param_type:
                    edge_cases.extend([f"{param_name}=0", f"{param_name}=INT_MAX", f"{param_name}=INT_MIN"])
                if 'char*' in param_type or 'char *' in param_type:
                    edge_cases.append(f"empty string for {param_name}")
        else:  # Rust
            for param_name, param_type in function.parameters:
                if param_name == 'self':
                    continue
                if 'i32' in param_type or 'i64' in param_type:
                    edge_cases.extend([f"{param_name}=0", f"{param_name}=MAX", f"{param_name}=MIN"])
                if '&str' in param_type or 'String' in param_type:
                    edge_cases.append(f"empty string for {param_name}")
                if 'Option' in param_type:
                    edge_cases.extend([f"{param_name}=None", f"{param_name}=Some"])

        return edge_cases


# Fallback generator that doesn't require API keys
class TemplateTestGenerator:
    """Generates tests using templates (no LLM required) - for baseline comparison"""

    def generate_c_tests(self, function: CFunction) -> GeneratedTest:
        """Generate basic template-based tests for C"""
        safe_name = safe_c_name(function.name)
        params = function.parameters

        # Build simple test calls based on parameter types
        test_calls = []

        if len(params) == 0:
            # No parameters, just call it
            test_calls.append(f'    {safe_name}();')
        elif len(params) == 1:
            ptype, pname = params[0]
            if 'int' in ptype.lower():
                test_calls.append(f'    {safe_name}(0);')
                test_calls.append(f'    {safe_name}(1);')
                test_calls.append(f'    {safe_name}(-1);')
        elif len(params) == 2:
            # Two parameters - common case
            t1, n1 = params[0]
            t2, n2 = params[1]
            if 'int' in t1.lower() and 'int' in t2.lower():
                test_calls.append(f'    {safe_name}(0, 0);')
                test_calls.append(f'    {safe_name}(1, 1);')
                test_calls.append(f'    {safe_name}(5, 3);')
        else:
            # Multiple params, just call with zeros
            args = ', '.join('0' for _ in params)
            test_calls.append(f'    {safe_name}({args});')

        test_body = '\n'.join(test_calls) if test_calls else f'    // No test generated for {safe_name}'

        test_code = f'''// Forward declaration
{function.raw_signature.replace(function.name, safe_name)};

void test_{safe_name}() {{
    printf("Testing {safe_name}...\\n");
{test_body}
    printf("  Basic tests completed\\n");
}}
'''
        return GeneratedTest(
            function_name=safe_name,
            test_code=test_code,
            test_name=f"test_{safe_name}",
            language="c",
            description=f"Template-based tests for {function.raw_signature}",
            edge_cases_covered=["basic values only"]
        )

    def generate_rust_tests(self, function: RustFunction) -> GeneratedTest:
        """Generate basic template-based tests for Rust"""
        test_code = f'''#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_{function.name}_basic() {{
        // TODO: Add appropriate test values based on function signature
        // Template-generated placeholder test
        assert!(true);
    }}
}}
'''
        return GeneratedTest(
            function_name=function.name,
            test_code=test_code,
            test_name=f"test_{function.name}",
            language="rust",
            description=f"Template-based tests for {function.raw_signature}",
            edge_cases_covered=["basic values only"]
        )
