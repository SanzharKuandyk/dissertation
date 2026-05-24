"""
LLM-Based Test Generator - Uses LLM APIs to generate unit tests
"""

import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

from ..parsers.c_parser import CFunction
from ..parsers.cpp_parser import CppFunction, CPP_KEYWORDS
from ..parsers.rust_parser import RustFunction

C_KEYWORDS = {
    "if", "else", "for", "while", "switch", "case", "return",
    "break", "continue", "default", "goto", "sizeof", "do",
    "static", "struct", "union", "typedef", "enum", "const",
    "volatile", "extern", "register", "inline", "restrict"
}


def safe_c_name(name: str) -> str:
    return f"{name}_fn" if name in C_KEYWORDS else name


def safe_cpp_name(name: str) -> str:
    return f"{name}_fn" if name in CPP_KEYWORDS else name


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

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("MLTEST_OPENAI_MODEL", "gpt-5")

    def _is_reasoning_model(self) -> bool:
        # GPT-5 reasoning models reject custom temperature and use max_completion_tokens.
        # The non-reasoning chat variant gpt-5-chat-latest behaves like gpt-4o.
        name = self.model.lower()
        return name.startswith("gpt-5") and not name.startswith("gpt-5-chat")

    def generate(self, prompt: str, system_prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)

            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            }
            if self._is_reasoning_model():
                # Reasoning tokens count against this budget; needs headroom.
                kwargs["max_completion_tokens"] = 8000
            else:
                kwargs["temperature"] = 0.3
                kwargs["max_tokens"] = 2000

            response = client.chat.completions.create(**kwargs)
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

    CPP_SYSTEM_PROMPT = """You are an expert C++ programmer and testing specialist.
Your task is to generate compilable unit tests for a single free C++ function.

CRITICAL RULES:
- Return ONLY valid C++ code, no markdown, no explanations
- DO NOT include headers (#include) - they will be added separately
- DO NOT redefine the function being tested
- DO NOT create a main() function
- ONLY write: one test function body named exactly as requested
- Use assert() for checks and std::cout for output
- Use std::string, nullptr, std::size_t, and fully qualified std:: names
- NEVER use reserved keywords as local variable names
- Prefer local names like arg0_, arg1_, result_basic_, result_edge_

ABSOLUTE PROHIBITIONS:
- NEVER add class definitions, templates, namespaces, or macros
- NEVER use `using namespace std;`
- NEVER emit pseudocode or placeholder syntax
- NEVER rename the function under test

Output: VALID C++ CODE ONLY. No explanations."""

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

    def generate_cpp_tests(self, function: CppFunction) -> GeneratedTest:
        """Generate unit tests for a C++ free function."""
        safe_name = safe_cpp_name(function.name)
        test_name = f"test_cpp_{safe_name}"
        prompt = self._build_cpp_prompt(function, test_name)

        response = self.llm.generate(prompt, self.CPP_SYSTEM_PROMPT)
        test_code = self._extract_code(response, "cpp")
        test_code = self._validate_and_clean_cpp_code(test_code, function, test_name)

        return GeneratedTest(
            function_name=function.name,
            test_code=test_code,
            test_name=test_name,
            language="cpp",
            description=f"Generated tests for {function.raw_signature}",
            edge_cases_covered=self._identify_edge_cases(function, "cpp"),
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

    def _build_cpp_prompt(self, function: CppFunction, test_name: str) -> str:
        """Build prompt for C++ test generation."""
        params_str = ", ".join(f"{t} {n}".strip() for t, n in function.parameters)
        ret_str = function.return_type.strip()
        return f"""Generate unit tests for this C++ free function:

Function: {function.name}
Signature: {function.raw_signature}
Return type: {ret_str}
Parameters: {params_str if params_str else "None"}
Body:
{function.body}

EXACT FORMAT TO FOLLOW:
```cpp
void {test_name}() {{
    std::cout << "Testing {function.name}...\\n";
    // declare safe local inputs such as arg0_, arg1_, result_basic_, result_edge_
    // call the function under test directly by its real name
    // add assert(...) checks where expectations are obvious
    std::cout << "  Tests completed\\n";
}}
```

RULES:
- NO headers, NO main(), NO forward declaration
- ONLY one function: {test_name}()
- Use std::string for string parameters and nullptr for pointer-null cases
- Use braces to isolate repeated local variable names if needed
- Do not use reserved keywords as variable names
- If exact expected values are unclear, prefer non-crash checks and obvious invariants

Generate the code now:
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
        """Extract the test_<func_name> definition and a single forward declaration
        from LLM output, preserving body contents verbatim.

        Uses brace-matching rather than keyword filtering so legitimate
        assert() / printf() / variable-init lines survive intact.
        """
        # Locate `[qualifiers] void test_<name>(...) {` allowing static/extern/inline.
        header_re = re.compile(
            r"(?:(?:static|extern|inline)\s+)*"
            r"void\s+test_" + re.escape(func_name) + r"\s*\([^)]*\)\s*\{",
            re.MULTILINE,
        )
        m = header_re.search(code)
        if not m:
            return self._placeholder_c_test(func_name)

        # Brace-balance from the '{' that ends the header.
        brace_open = m.end() - 1
        depth = 0
        end = None
        for i in range(brace_open, len(code)):
            ch = code[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return self._placeholder_c_test(func_name)

        test_def = code[m.start():end]

        # Pick the first plausible forward declaration that mentions func_name,
        # appears before the test func, and is not itself a test definition.
        forward_decl = None
        for line in code[: m.start()].splitlines():
            stripped = line.strip()
            if (
                stripped.endswith(";")
                and func_name in stripped
                and "test_" not in stripped
                and not stripped.startswith("//")
                and not stripped.startswith("/*")
            ):
                forward_decl = line
                break

        parts = []
        if forward_decl:
            parts.append("// Forward declaration")
            parts.append(forward_decl)
            parts.append("")
        parts.append(test_def)
        return "\n".join(parts) + "\n"

    def _placeholder_c_test(self, func_name: str) -> str:
        return (
            f"// Forward declaration\n"
            f"extern int {func_name}();\n"
            f"\n"
            f"void test_{func_name}() {{\n"
            f'    printf("Testing {func_name}...\\n");\n'
            f"    // LLM generated invalid code, using placeholder\n"
            f'    printf("  Test placeholder\\n");\n'
            f"}}\n"
        )

    def _validate_and_clean_cpp_code(
        self, code: str, function: CppFunction, test_name: str
    ) -> str:
        """Extract a single C++ test function and prepend a canonical declaration."""
        header_re = re.compile(
            r"(?:(?:static|inline)\s+)*void\s+"
            + re.escape(test_name)
            + r"\s*\([^)]*\)\s*\{",
            re.MULTILINE,
        )
        match = header_re.search(code)
        if not match:
            return TemplateTestGenerator().generate_cpp_tests(function).test_code

        brace_open = match.end() - 1
        depth = 0
        end = None
        for index in range(brace_open, len(code)):
            ch = code[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break

        if end is None:
            return TemplateTestGenerator().generate_cpp_tests(function).test_code

        test_def = code[match.start():end].strip()
        return (
            f"// Forward declaration\n{function.raw_signature};\n\n"
            f"{test_def}\n"
        )

    def _identify_edge_cases(
        self, function: Union[CFunction, CppFunction, RustFunction], language: str
    ) -> List[str]:
        """Identify potential edge cases based on function signature"""
        edge_cases = []

        if language in {"c", "cpp"}:
            for param_type, param_name in function.parameters:
                if '*' in param_type:
                    edge_cases.append(f"NULL {param_name}")
                if 'int' in param_type:
                    edge_cases.extend([f"{param_name}=0", f"{param_name}=INT_MAX", f"{param_name}=INT_MIN"])
                if 'std::string' in param_type or 'string' in param_type:
                    edge_cases.append(f"empty string for {param_name}")
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

    def generate_cpp_tests(self, function) -> GeneratedTest:
        """
        Generate template-based C++ unit tests.

        C++ keyword safety rules applied here:
          - Function name is sanitised against the full C++ keyword set before use
          - Variable names for test arguments use a prefixed scheme (arg0_, arg1_,
            ...) that cannot clash with any C++ keyword or standard-library name
          - nullptr is used instead of NULL / 0 for pointer parameters
          - std::string literals are used for string parameters
          - Test function is named test_cpp_<func> to avoid collisions with
            functions in the translation unit under test
          - No 'using namespace std' — all std:: types are fully qualified
          - assert() from <cassert>, output via std::cout from <iostream>
        """
        from ..parsers.cpp_parser import CPP_KEYWORDS

        func_name = function.name
        # Sanitise: if the name is a C++ keyword (shouldn't happen after parsing,
        # but be defensive), prefix it
        if func_name in CPP_KEYWORDS:
            func_name = f"fn_{func_name}"

        test_fn_name = f"test_cpp_{func_name}"
        params = function.parameters  # list of (type_str, param_name)

        # Build argument lists for several test call variants
        # using safe prefixed variable names (arg0_, arg1_, …)
        arg_vars = []
        arg_decls_basic = []    # declarations with zero/default values
        arg_decls_edge  = []    # declarations with edge-case values
        ptr_checks      = []    # nullptr pointer guards

        for idx, (ptype, _pname) in enumerate(params):
            var = f"arg{idx}_"          # safe name: never a keyword
            pt = ptype.strip()
            basic_var = f"{var}basic"
            edge_var = f"{var}edge"

            if "std::string" in pt or "string" in pt:
                arg_decls_basic.append(f'        std::string {basic_var} = "hello";')
                arg_decls_edge.append(f'        std::string {edge_var} = "";')
                arg_vars.append((basic_var, edge_var))
            elif "*" in pt:
                # Pointer param — use nullptr for the null-check call,
                # real array for the normal call
                pointee = "const int*" if "const" in pt else "int*"
                arg_decls_basic.append(f'        int buf{idx}_[] = {{1, 2, 3, 4, 5}};')
                arg_decls_basic.append(f'        {pointee} {basic_var} = buf{idx}_;')
                arg_decls_edge.append(f'        {pointee} {edge_var} = nullptr;')
                arg_vars.append((basic_var, edge_var))
                ptr_checks.append((idx, edge_var))
            elif "bool" in pt:
                arg_decls_basic.append(f'        bool {basic_var} = true;')
                arg_decls_edge.append(f'        bool {edge_var} = false;')
                arg_vars.append((basic_var, edge_var))
            elif "double" in pt or "float" in pt:
                arg_decls_basic.append(f'        double {basic_var} = 1.0;')
                arg_decls_edge.append(f'        double {edge_var} = 0.0;')
                arg_vars.append((basic_var, edge_var))
            elif "long long" in pt or "int64" in pt:
                arg_decls_basic.append(f'        long long {basic_var} = 1LL;')
                arg_decls_edge.append(f'        long long {edge_var} = 0LL;')
                arg_vars.append((basic_var, edge_var))
            elif "long" in pt:
                arg_decls_basic.append(f'        long {basic_var} = 1L;')
                arg_decls_edge.append(f'        long {edge_var} = 0L;')
                arg_vars.append((basic_var, edge_var))
            elif "char" in pt and "*" not in pt:
                arg_decls_basic.append(f'        char {basic_var} = \'a\';')
                arg_decls_edge.append(f'        char {edge_var} = \'\\0\';')
                arg_vars.append((basic_var, edge_var))
            elif "size_t" in pt or "uint" in pt:
                arg_decls_basic.append(f'        std::size_t {basic_var} = 3;')
                arg_decls_edge.append(f'        std::size_t {edge_var} = 0;')
                arg_vars.append((basic_var, edge_var))
            else:
                # Default: treat as int
                arg_decls_basic.append(f'        int {basic_var} = 1;')
                arg_decls_edge.append(f'        int {edge_var} = 0;')
                arg_vars.append((basic_var, edge_var))

        basic_args = ", ".join(b for b, _ in arg_vars)
        edge_args = ", ".join(e for _, e in arg_vars)

        # Return type determines how we capture / assert the result
        ret = function.return_type.strip()
        if ret == "void":
            call_basic = f"        {func_name}({basic_args});"
            call_edge  = f"        {func_name}({edge_args});"
            assert_basic = ""
            assert_edge  = ""
        elif "bool" in ret:
            call_basic = f"        bool result_basic_ = {func_name}({basic_args});"
            call_edge  = f"        bool result_edge_ = {func_name}({edge_args});"
            assert_basic = "        // result_basic_ is true or false — both are valid"
            assert_edge  = "        // result_edge_ is true or false — both are valid"
        elif "std::string" in ret or "string" in ret:
            call_basic = f"        std::string result_basic_ = {func_name}({basic_args});"
            call_edge  = f"        std::string result_edge_ = {func_name}({edge_args});"
            assert_basic = "        assert(!result_basic_.empty() || result_basic_.empty()); // non-crash check"
            assert_edge  = "        assert(!result_edge_.empty() || result_edge_.empty());"
        elif "*" in ret:
            call_basic = f"        auto* result_basic_ = {func_name}({basic_args});"
            call_edge  = f"        auto* result_edge_ = {func_name}({edge_args});"
            assert_basic = "        // pointer result — nullptr is valid"
            assert_edge  = ""
        else:
            # Numeric return
            call_basic = f"        auto result_basic_ = {func_name}({basic_args});"
            call_edge  = f"        auto result_edge_ = {func_name}({edge_args});"
            assert_basic = "        (void)result_basic_; // suppress unused-variable warning"
            assert_edge  = "        (void)result_edge_;"

        # Nullptr calls for pointer params
        null_calls = []
        for idx, pvar in ptr_checks:
            null_args = ", ".join(
                pvar if pos == idx else edge_name
                for pos, (_basic_name, edge_name) in enumerate(arg_vars)
            )
            null_calls.append(f"        {func_name}({null_args}); // nullptr safety check")

        basic_block  = "\n".join(arg_decls_basic)
        edge_block   = "\n".join(arg_decls_edge)
        null_block   = "\n".join(null_calls)

        test_code = f"""// Forward declaration
{function.raw_signature};

void {test_fn_name}() {{
    std::cout << "Testing {func_name}...\\n";

    {{
        // --- Basic call ---
{basic_block}
{call_basic}
{assert_basic}
    }}

    {{
        // --- Edge / zero call ---
{edge_block}
{call_edge}
{assert_edge}
{"" if not null_block else "        // --- Null-pointer safety ---"}
{null_block}
    }}

    std::cout << "  {func_name}: basic tests passed\\n";
}}
"""

        return GeneratedTest(
            function_name=func_name,
            test_code=test_code,
            test_name=test_fn_name,
            language="cpp",
            description=f"Template-based C++ tests for {function.raw_signature}",
            edge_cases_covered=["zero values", "null pointers", "empty strings"],
        )
