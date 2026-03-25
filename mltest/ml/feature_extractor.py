"""
Feature extraction for ML strategy selection.

Extracts code complexity and structural features from parsed C and Rust
function objects. These features are used to train and run the ML model
that decides which test generation strategy (LLM vs template) to use.
"""

import re
from typing import Union, List, Dict
import pandas as pd

from ..parsers.c_parser import CFunction
from ..parsers.rust_parser import RustFunction

# Ordered list — this is the single source of truth for column ordering.
# Must remain stable across training and inference.
FEATURE_NAMES = [
    # Structural
    "body_line_count",
    "param_count",
    "has_return_value",
    "body_char_count",
    "avg_line_length",
    # Complexity
    "cyclomatic_complexity",
    "loop_count",
    "conditional_count",
    "return_count",
    "early_return_count",
    "nested_depth",
    "null_check_count",
    "recursion_likely",
    # Type-based
    "has_pointer_params",
    "has_output_params",
    "param_has_string",
    "param_has_array",
    "has_option_return",
    "has_result_return",
    "has_generic_params",
    "is_async",
    # Language
    "language",
]


class FunctionFeatureExtractor:
    """
    Extracts a fixed-length feature vector from a CFunction or RustFunction.

    All features are numeric (int or float) so they can be used directly
    with scikit-learn estimators without encoding.
    """

    FEATURE_NAMES = FEATURE_NAMES

    def extract(
        self, func: Union[CFunction, RustFunction], language: str
    ) -> Dict[str, float]:
        """
        Extract features from a single parsed function.

        Args:
            func: A CFunction or RustFunction dataclass instance.
            language: 'c' or 'rust'

        Returns:
            Dict mapping feature name → float value, with keys in FEATURE_NAMES order.
        """
        body = func.body or ""
        lang_flag = 0 if language == "c" else 1

        # --- Structural ---
        body_line_count = max(1, (func.line_end or 1) - (func.line_start or 0) + 1)
        param_count = len(func.parameters) if func.parameters else 0

        ret = func.return_type or ""
        has_return_value = 0 if ret.strip() in ("void", "", "None") else 1

        body_char_count = len(body)
        avg_line_length = body_char_count / body_line_count

        # --- Complexity ---
        cc = self._cyclomatic_complexity(body, language)
        loop_count = self._loop_count(body, language)
        conditional_count = len(re.findall(r"\bif\b", body))
        return_count = len(re.findall(r"\breturn\b", body))
        early_return_count = max(0, return_count - 1)
        nested_depth = self._nested_depth(body)
        null_check_count = self._null_check_count(body, language)
        recursion_likely = 1 if func.name and re.search(
            r"\b" + re.escape(func.name) + r"\s*\(", body
        ) else 0

        # --- Type-based ---
        if language in ("c", "cpp"):
            params = func.parameters  # list of (type, name)
            has_pointer_params = 1 if any("*" in (t or "") for t, _ in params) else 0
            has_output_params = 1 if any(
                "*" in (t or "") and "const" not in (t or "")
                for t, _ in params
            ) else 0
            # C: char* strings;  C++: std::string / const std::string&
            param_has_string = 1 if any(
                ("char" in (t or "") and "*" in (t or "")) or
                "string" in (t or "").lower()
                for t, _ in params
            ) else 0
            param_has_array = 1 if any(
                "*" in (t or "") or "vector" in (t or "").lower()
                for t, _ in params
            ) else 0
            ret_str = func.return_type or ""
            has_option_return = 0
            has_result_return = 0
            has_generic_params = 0
            is_async_flag = 0
        else:
            params = func.parameters  # list of (name, type)
            has_pointer_params = 0
            has_output_params = 0
            param_has_string = 1 if any(
                "&str" in (t or "") or "String" in (t or "")
                for _, t in params
            ) else 0
            param_has_array = 1 if any(
                "&[" in (t or "") or "Vec" in (t or "")
                for _, t in params
            ) else 0
            ret_str = func.return_type or ""
            has_option_return = 1 if "Option" in ret_str else 0
            has_result_return = 1 if "Result" in ret_str else 0
            has_generic_params = 1 if getattr(func, "generics", None) else 0
            is_async_flag = 1 if getattr(func, "is_async", False) else 0

        return {
            "body_line_count": float(body_line_count),
            "param_count": float(param_count),
            "has_return_value": float(has_return_value),
            "body_char_count": float(body_char_count),
            "avg_line_length": float(avg_line_length),
            "cyclomatic_complexity": float(cc),
            "loop_count": float(loop_count),
            "conditional_count": float(conditional_count),
            "return_count": float(return_count),
            "early_return_count": float(early_return_count),
            "nested_depth": float(nested_depth),
            "null_check_count": float(null_check_count),
            "recursion_likely": float(recursion_likely),
            "has_pointer_params": float(has_pointer_params),
            "has_output_params": float(has_output_params),
            "param_has_string": float(param_has_string),
            "param_has_array": float(param_has_array),
            "has_option_return": float(has_option_return),
            "has_result_return": float(has_result_return),
            "has_generic_params": float(has_generic_params),
            "is_async": float(is_async_flag),
            "language": float(lang_flag),
        }

    def extract_batch(
        self, funcs: List[Union[CFunction, RustFunction]], language: str
    ) -> pd.DataFrame:
        """Extract features for a list of functions, returning a DataFrame."""
        rows = [self.extract(f, language) for f in funcs]
        return pd.DataFrame(rows, columns=FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cyclomatic_complexity(self, body: str, language: str) -> int:
        """
        Approximate cyclomatic complexity by counting decision points + 1.
        Uses regex on source text — no full AST required.
        """
        count = 1
        if language in ("c", "cpp"):
            patterns = [
                r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bdo\b",
                r"\bcase\b", r"&&", r"\|\|", r"\?[^:]",
                # C++ extras
                r"\bcatch\b", r"\bthrow\b",
            ]
        else:
            patterns = [
                r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bloop\b",
                r"=>", r"&&", r"\|\|",
            ]
        for pattern in patterns:
            count += len(re.findall(pattern, body))
        return count

    def _loop_count(self, body: str, language: str) -> int:
        if language in ("c", "cpp"):
            patterns = [r"\bfor\b", r"\bwhile\b", r"\bdo\b"]
        else:
            patterns = [r"\bfor\b", r"\bwhile\b", r"\bloop\b"]
        return sum(len(re.findall(p, body)) for p in patterns)

    def _nested_depth(self, body: str) -> int:
        """Track maximum brace nesting depth inside the function body."""
        max_depth = 0
        depth = 0
        for ch in body:
            if ch == "{":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == "}":
                depth = max(0, depth - 1)
        return max_depth

    def _null_check_count(self, body: str, language: str) -> int:
        if language in ("c", "cpp"):
            patterns = [
                r"==\s*NULL", r"!=\s*NULL",
                r"==\s*nullptr", r"!=\s*nullptr",
                r"==\s*0\b", r"!=\s*0\b",
            ]
        else:
            patterns = [r"\.is_none\(\)", r"\.is_some\(\)", r"== None\b", r"!= None\b"]
        return sum(len(re.findall(p, body)) for p in patterns)
