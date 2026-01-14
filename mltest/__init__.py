"""
MLTest - Machine Learning-Driven Automated Unit Test Generation
for System-Level Programming Languages (C, Rust)
"""

__version__ = "0.1.0"
__author__ = "Dissertation Research"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from .cli import main

__all__ = ["main"]
