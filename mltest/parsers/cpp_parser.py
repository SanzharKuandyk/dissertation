"""
C++ Code Parser — Extracts free function signatures and bodies.

Design decisions (conservative by intent):
  - Only extracts FREE functions at file/namespace scope, not class methods
  - Skips template<> functions entirely (too complex for reliable regex parsing)
  - Skips operator overloads
  - Skips constructors / destructors (no return type)
  - Handles std:: qualified types, references, pointers, const qualifiers
  - Handles multi-line signatures (opening brace on next line)

C++ has many more keywords and ambiguities than C.  Rather than try to be
exhaustive, the parser is deliberately strict: if a pattern is not clearly a
free function definition, it is skipped.  This trades recall for precision —
better to miss a function than to generate a corrupt test.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


# -------------------------------------------------------------------------
# Complete C++ keyword set (ISO C++23)
# Used to filter out false-positive function name matches.
# -------------------------------------------------------------------------
CPP_KEYWORDS: frozenset = frozenset({
    # Control flow
    "if", "else", "for", "while", "do", "switch", "case", "break",
    "continue", "return", "goto", "default",
    # Types (built-in)
    "void", "bool", "char", "short", "int", "long", "float", "double",
    "signed", "unsigned", "wchar_t", "char8_t", "char16_t", "char32_t",
    # Storage / qualifiers
    "const", "volatile", "mutable", "constexpr", "consteval", "constinit",
    "static", "extern", "register", "thread_local", "inline", "virtual",
    "explicit", "friend", "auto",
    # OOP
    "class", "struct", "union", "enum", "namespace", "using", "typedef",
    "public", "protected", "private",
    # Templates / generics
    "template", "typename", "concept", "requires", "export",
    # Memory
    "new", "delete", "sizeof", "alignof", "alignas", "decltype", "typeid",
    # Casts
    "static_cast", "dynamic_cast", "reinterpret_cast", "const_cast",
    # Exception
    "try", "catch", "throw", "noexcept",
    # C++ operators as identifiers (alternative tokens)
    "and", "or", "not", "xor", "bitand", "bitor", "compl",
    "and_eq", "or_eq", "not_eq", "xor_eq",
    # Literals / special
    "true", "false", "nullptr", "this",
    # Coroutines (C++20)
    "co_await", "co_return", "co_yield",
    # Modules (C++20)
    "import", "module",
    # Misc
    "operator", "override", "final", "main",
    # Common macro-like names that trip up parsers
    "NULL", "TRUE", "FALSE", "assert",
})

# Return types we never want to see (noise from comments / macros)
_INVALID_RETURN_TYPES: frozenset = frozenset({
    "overflow", "underflow", "if", "while", "for", "else", "do",
    "switch", "case", "return", "break", "continue", "goto",
    "public", "private", "protected", "class", "struct", "namespace",
    "template", "typename", "virtual", "override", "friend", "new",
    "delete", "operator", "typedef", "using", "static_assert",
    "#include", "#define", "#if", "#ifdef", "#endif",
})

# Simple base types allowed as return types (avoids matching random keywords)
_VALID_RETURN_BASE = re.compile(
    r"^(?:void|bool|int|long|short|char|float|double|size_t|"
    r"int8_t|int16_t|int32_t|int64_t|uint8_t|uint16_t|uint32_t|uint64_t|"
    r"ptrdiff_t|ssize_t|intptr_t|uintptr_t|"
    r"string|std(?:::[a-zA-Z_][a-zA-Z0-9_<>, ]*)?)"
)


@dataclass
class CppFunction:
    """Represents a C++ free function extracted from source code."""
    name: str
    return_type: str
    parameters: List[Tuple[str, str]]   # (type_str, param_name)
    body: str
    line_start: int
    line_end: int
    includes: List[str]
    raw_signature: str
    is_const_method: bool = False       # true if () const — informational only
    namespace: Optional[str] = None    # enclosing namespace if any

    def to_context(self) -> str:
        """Generate context string for test generation prompts."""
        params_str = ", ".join(f"{t} {n}" for t, n in self.parameters)
        ns_prefix = f"Namespace: {self.namespace}\n" if self.namespace else ""
        return (
            f"\n{ns_prefix}"
            f"Function: {self.name}\n"
            f"Signature: {self.return_type} {self.name}({params_str})"
            f"{'const' if self.is_const_method else ''}\n"
            f"Body:\n{self.body}\n\n"
            f"Required includes: {', '.join(self.includes) if self.includes else 'None'}\n"
        )


class CppParser:
    """
    Conservative parser for C++ source code.

    Extracts only free (non-member, non-template) functions.  Uses a two-pass
    approach: first strips class/struct bodies and template prefixes, then
    applies regex to the cleaned source.
    """

    INCLUDE_PATTERN = re.compile(r'#include\s*[<"]([^>"]+)[>"]')

    # Match a function definition:
    #   [qualifiers] return_type func_name ( params ) [const] [noexcept] {
    #
    # Return type captures:
    #   - optional: const, unsigned, static, inline, constexpr, [[nodiscard]]
    #   - base type: word characters + optional std:: prefix
    #   - optional: one level of template angle brackets <...>
    #   - optional: pointer/reference decorators * & &&
    #
    # We intentionally do NOT match templates (handled by pre-filtering).
    FUNC_PATTERN = re.compile(
        r"^[ \t]*"
        # Optional leading qualifiers
        r"(?:(?:static|inline|constexpr|explicit|virtual|"
        r"(?:\[\[nodiscard\]\])|(?:\[\[maybe_unused\]\]))\s+)*"
        # Return type (greedy but bounded)
        r"(?P<return_type>"
            r"(?:const\s+)?"                        # optional leading const
            r"(?:unsigned\s+|signed\s+)?"            # optional sign
            r"[a-zA-Z_][a-zA-Z0-9_]*"              # base identifier (int, long, std, ...)
            r"(?:\s+long)?"                         # allow 'long long'
            r"(?:::[a-zA-Z_][a-zA-Z0-9_]*)*"       # optional ::name chains
            r"(?:<[^>]{0,60}>)?"                    # optional <T> (max 60 chars)
            r"(?:\s*(?:const\s*)?)?"                # optional trailing const
            r"(?:\s*[*&]{1,2})?"                    # optional ptr/ref/rref
        r")\s+"
        # Function name — must not be a keyword
        r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
        r"\s*\(\s*"
        # Parameters (stop at first unmatched ')' — simplified, no nested parens)
        r"(?P<params>[^)]*)"
        r"\)\s*"
        # Optional const / noexcept / override qualifiers
        r"(?:const\s*)?(?:noexcept(?:\([^)]*\))?\s*)?(?:override\s*)?(?:final\s*)?"
        # Opening brace (on same line or next line handled below)
        r"\{",
        re.MULTILINE,
    )

    # Match namespace declarations to track scope
    NAMESPACE_PATTERN = re.compile(
        r"^\s*namespace\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*\{",
        re.MULTILINE,
    )

    def __init__(self):
        self.functions: List[CppFunction] = []
        self.includes: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, filepath: Path) -> List[CppFunction]:
        """Parse a .cpp/.hpp file and return extracted functions."""
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        return self.parse_source(content)

    def parse_source(self, source: str) -> List[CppFunction]:
        """Parse C++ source string and return extracted CppFunction objects."""
        self.functions = []
        self.includes = self.INCLUDE_PATTERN.findall(source)

        # Pre-process: remove block comments to avoid matching inside them
        cleaned = self._strip_block_comments(source)

        # Remove template<...> function prefixes and class/struct bodies
        # so the regex doesn't match template functions or member functions
        cleaned = self._mask_template_functions(cleaned)
        cleaned = self._mask_class_bodies(cleaned)

        # Now parse free functions from cleaned source
        for match in self.FUNC_PATTERN.finditer(cleaned):
            self._process_match(match, source, cleaned)

        return self.functions

    def get_testable_functions(self) -> List[CppFunction]:
        """Return functions suitable for unit testing."""
        testable = []
        for func in self.functions:
            # Skip void no-arg functions (likely side-effect only)
            if func.return_type.strip() == "void" and not func.parameters:
                continue
            testable.append(func)
        return testable

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_match(self, match, original_source: str, cleaned: str):
        """Validate a regex match and build a CppFunction if valid."""
        return_type = match.group("return_type").strip()
        name = match.group("name").strip()
        params_str = match.group("params").strip()

        # --- Hard filters ---
        if name in CPP_KEYWORDS:
            return
        if name.startswith("_"):          # skip implementation-reserved names
            return
        if "operator" in name:
            return
        # Reject if return type looks invalid
        rt_base = return_type.split()[0].lstrip("*&").lower()
        if rt_base in {r.lower() for r in _INVALID_RETURN_TYPES}:
            return
        # Ensure return type starts with a known valid base
        if not _VALID_RETURN_BASE.match(return_type.strip("*& \t")):
            return
        # Skip destructors/constructors heuristic (name repeated in return type)
        if return_type.strip() == "" or name == rt_base:
            return

        # --- Parse parameters ---
        parameters = self._parse_parameters(params_str)

        # --- Extract body from ORIGINAL source (not cleaned) ---
        brace_pos = match.end() - 1   # position of '{'
        body, body_end = self._extract_body(cleaned, brace_pos)

        line_start = original_source[:match.start()].count("\n") + 1
        line_end = original_source[:body_end].count("\n") + 1

        # Build is_const from the portion between ')' and '{'
        between = cleaned[match.end(0) - match.end(0): match.end(0)]
        sig_tail = cleaned[match.start("params") + len(params_str): match.end()]
        is_const = "const" in sig_tail.split("{")[0]

        raw_sig = (
            f"{return_type} {name}({params_str})"
            + (" const" if is_const else "")
        )

        func = CppFunction(
            name=name,
            return_type=return_type,
            parameters=parameters,
            body=body,
            line_start=line_start,
            line_end=line_end,
            includes=self.includes.copy(),
            raw_signature=raw_sig,
            is_const_method=is_const,
        )
        self.functions.append(func)

    def _parse_parameters(self, params_str: str) -> List[Tuple[str, str]]:
        """
        Parse a C++ parameter list into (type, name) pairs.

        Handles:
          int a, double b
          const std::string& s, int* ptr
          std::vector<int> v, size_t n
          int x = 0           (default arg — name still extracted)
        """
        params: List[Tuple[str, str]] = []
        if not params_str.strip() or params_str.strip() == "void":
            return params

        # Split on commas, but respect angle-bracket depth for templates
        parts = self._split_params(params_str)

        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Strip default value
            if "=" in part:
                part = part[:part.index("=")].strip()
            # The last token (possibly after * or &) is the parameter name
            # unless it ends with '>' (template, no name) or ')'
            tokens = part.rstrip().rsplit(None, 1)
            if len(tokens) == 2:
                raw_type, pname = tokens
                # Clean up: name should not contain type characters
                pname = pname.lstrip("*&")
                # Skip unnamed parameters (pname is a type keyword)
                if pname in CPP_KEYWORDS or not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", pname):
                    params.append((part, ""))
                else:
                    params.append((raw_type.strip(), pname))
            elif len(tokens) == 1:
                # Unnamed parameter
                params.append((tokens[0], ""))

        return params

    def _split_params(self, params_str: str) -> List[str]:
        """Split parameter list on commas, respecting <> depth."""
        parts = []
        depth = 0
        current = []
        for ch in params_str:
            if ch == "<":
                depth += 1
                current.append(ch)
            elif ch == ">":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    def _extract_body(self, source: str, start_pos: int) -> Tuple[str, int]:
        """Extract function body by tracking brace depth."""
        brace_count = 0
        pos = start_pos
        body_start = start_pos + 1

        while pos < len(source):
            ch = source[pos]
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    return source[body_start:pos].strip(), pos
            pos += 1

        return source[body_start:].strip(), len(source)

    def _strip_block_comments(self, source: str) -> str:
        """Replace /* ... */ block comments with whitespace (preserve line counts)."""
        result = []
        i = 0
        while i < len(source):
            if source[i:i+2] == "/*":
                j = source.find("*/", i + 2)
                if j == -1:
                    # Unterminated comment — blank the rest
                    newlines = source[i:].count("\n")
                    result.append("\n" * newlines)
                    break
                # Replace with same number of newlines to keep line numbers intact
                comment = source[i:j+2]
                result.append("\n" * comment.count("\n"))
                i = j + 2
            else:
                result.append(source[i])
                i += 1
        return "".join(result)

    def _mask_template_functions(self, source: str) -> str:
        """
        Blank out template<...> function definitions so the main regex
        never matches them.  Also blanks out class/function forward declarations
        with template<...> on the preceding line.
        """
        # Find template<...> blocks and replace with spaces (preserve newlines)
        pattern = re.compile(
            r"template\s*<[^>]*(?:>[^>]*)*>\s*",   # template<...> (simple)
            re.DOTALL,
        )
        def blank(m):
            return "\n" * m.group(0).count("\n") + " " * (len(m.group(0)) - m.group(0).count("\n"))
        return pattern.sub(blank, source)

    def _mask_class_bodies(self, source: str) -> str:
        """
        Blank out the bodies of class/struct/union declarations so member
        functions inside them are never matched as free functions.

        Handles nested braces correctly.
        """
        # Find class/struct/union keywords followed eventually by a {
        class_kw = re.compile(
            r"\b(?:class|struct|union)\b[^;{]*\{",
            re.DOTALL,
        )
        chars = list(source)
        for m in class_kw.finditer(source):
            # Find the matching closing brace
            brace_start = m.end() - 1  # position of '{'
            depth = 0
            pos = brace_start
            while pos < len(chars):
                if chars[pos] == "{":
                    depth += 1
                elif chars[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        # Blank everything inside (keep newlines)
                        for i in range(brace_start + 1, pos):
                            if chars[i] != "\n":
                                chars[i] = " "
                        break
                pos += 1
        return "".join(chars)


if __name__ == "__main__":
    test_code = r"""
#include <string>
#include <vector>
#include <cstdint>

namespace math {

int safe_add(int a, int b) {
    if (a > 0 && b > INT_MAX - a) return INT_MAX;
    return a + b;
}

}  // namespace math

std::string reverse_string(const std::string& s) {
    return std::string(s.rbegin(), s.rend());
}

template<typename T>
T max_val(T a, T b) { return a > b ? a : b; }

class Foo {
public:
    int get() const { return 42; }
};

double safe_divide(double num, double denom) {
    if (denom == 0.0) return 0.0;
    return num / denom;
}
"""
    parser = CppParser()
    funcs = parser.parse_source(test_code)
    print(f"Extracted {len(funcs)} functions:")
    for f in funcs:
        print(f"  {f.raw_signature}  (lines {f.line_start}-{f.line_end})")
        print(f"    params: {f.parameters}")
