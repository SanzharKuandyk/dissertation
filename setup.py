"""Setup script for MLTest package"""

from setuptools import setup, find_packages

setup(
    name="mltest",
    version="0.1.0",
    description="Machine Learning-Driven Automated Unit Test Generation for System-Level Languages",
    author="Dissertation Research",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ],
        "full": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "tree-sitter>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mltest=mltest.cli:main",
        ],
    },
)
