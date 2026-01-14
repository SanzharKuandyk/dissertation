#!/usr/bin/env python3
"""Debug script to see what functions are being parsed"""

from mltest.parsers.c_parser import CParser
from pathlib import Path

# Parse the math_utils.c file
parser = CParser()
source = Path('benchmarks/c/math_utils.c').read_text()
functions = parser.parse_source(source)

print(f"Found {len(functions)} functions:\n")

for i, func in enumerate(functions, 1):
    print(f"{i}. {func.name}")
    print(f"   Return type: {func.return_type}")
    print(f"   Raw signature: {func.raw_signature}")
    print(f"   Parameters: {func.parameters}")
    print()
