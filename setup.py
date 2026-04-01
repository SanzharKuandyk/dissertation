"""Setup script for MLTest package"""

from setuptools import setup, find_packages

setup(
    name="mltest",
    version="0.1.0",
    description="Static screening and LLM-assisted unit test generation for system-level languages",
    author="Dissertation Research",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "llm": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
        ],
        "full": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "tree-sitter>=0.20.0",
            "scikit-learn>=1.3.0",
            "joblib>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mltest=mltest.cli:main",
        ],
    },
)
