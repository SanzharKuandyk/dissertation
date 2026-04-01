"""
ML subpackage for static feature extraction and LLM suitability prediction.

Provides:
  - FunctionFeatureExtractor: extracts code features from parsed functions
  - MLStrategySelector: trains and scores the current suitability model
"""

from .feature_extractor import FunctionFeatureExtractor
from .strategy_selector import MLStrategySelector

__all__ = ["FunctionFeatureExtractor", "MLStrategySelector"]
