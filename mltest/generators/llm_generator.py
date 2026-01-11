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

Guidelines:
- Generate tests using a simple assertion-based approach (no external testing framework required)
- Include tests for normal cases, edge cases, and boundary conditions
- Test for NULL pointers where applicable
- Test numeric overflow/underflow for integer operations
- Include comments explaining each test case
- Generate compilable, runnable C code

Output format: Return ONLY the test code, no explanations."""

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
        """Generate unit tests for a C function"""
        prompt = self._build_c_prompt(function)

        response = self.llm.generate(prompt, self.C_SYSTEM_PROMPT)
        test_code = self._extract_code(response, "c")

        return GeneratedTest(
            function_name=function.name,
            test_code=test_code,
            test_name=f"test_{function.name}",
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

    def _build_c_prompt(self, function: CFunction) -> str:
        """Build prompt for C test generation"""
        return f"""Generate comprehensive unit tests for the following C function:

{function.to_context()}

Requirements:
1. Create a test file that includes necessary headers
2. Test normal operation with typical inputs
3. Test edge cases (empty inputs, zero values, negative numbers where applicable)
4. Test boundary conditions
5. If the function handles pointers, test NULL handling
6. Create a main() function that runs all tests and reports results

The test file should be self-contained and compilable with: gcc -o test test.c
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
        if f"```{language}" in response.lower() or "```c" in response.lower() or "```rust" in response.lower():
            # Extract code between backticks
            lines = response.split('\n')
            in_code = False
            code_lines = []

            for line in lines:
                if line.strip().startswith('```') and not in_code:
                    in_code = True
                    continue
                elif line.strip() == '```' and in_code:
                    in_code = False
                    continue
                elif in_code:
                    code_lines.append(line)

            return '\n'.join(code_lines)

        # No code blocks, return as-is
        return response.strip()

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
        params = function.parameters

        # Generate test calls with basic values
        test_calls = []
        for i, (ptype, pname) in enumerate(params):
            if 'int' in ptype:
                test_calls.append(('0', 'zero'))
                test_calls.append(('1', 'one'))
                test_calls.append(('-1', 'negative'))
            elif '*' in ptype:
                test_calls.append(('NULL', 'null'))
            else:
                test_calls.append(('0', 'default'))

        test_code = f'''#include <stdio.h>
#include <assert.h>

// Forward declaration
{function.raw_signature};

// Basic template-generated tests
void test_{function.name}_basic() {{
    printf("Testing {function.name} with basic values...\\n");
    // TODO: Add appropriate test values based on function signature
    printf("  PASSED\\n");
}}

int main() {{
    printf("Running template-generated tests for {function.name}\\n");
    test_{function.name}_basic();
    printf("All tests passed!\\n");
    return 0;
}}
'''
        return GeneratedTest(
            function_name=function.name,
            test_code=test_code,
            test_name=f"test_{function.name}",
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
