"""
LLM suitability predictor based on static code features.

The current model is trained on a binary label derived from LLM coverage
quality. During inference we interpret the positive-class probability as an
LLM suitability score, which supports both screening/triage and the legacy
routing workflow.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate

from .feature_extractor import FEATURE_NAMES, FunctionFeatureExtractor
from ..parsers.c_parser import CFunction, CParser
from ..parsers.cpp_parser import CppFunction, CppParser
from ..parsers.rust_parser import RustFunction, RustParser

logger = logging.getLogger(__name__)

# Coverage quality threshold: functions where LLM achieves >= this percentage
# are labelled 1 ("high-quality testable"). When set to -1 (auto), the median
# LLM coverage across the training set is used to create a balanced split.
COVERAGE_QUALITY_THRESHOLD = -1  # -1 = use median (recommended)

GOOD_CANDIDATE_THRESHOLD = 0.70
BORDERLINE_THRESHOLD = 0.40

ParsedFunction = Union[CFunction, RustFunction, CppFunction]


class MLStrategySelector:
    """
    Trains and applies an ML model for LLM suitability prediction.

    Usage (training):
        selector = MLStrategySelector()
        metrics = selector.train(
            results_path=Path("results/results.json"),
            benchmarks_dir=Path("benchmarks"),
            model_output_path=Path("models/strategy_selector.joblib"),
        )

    Usage (inference):
        selector = MLStrategySelector()
        selector.load_model(Path("models/strategy_selector.joblib"))
        score = selector.score_function(func, language="c")
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = FEATURE_NAMES
        self.extractor = FunctionFeatureExtractor()
        self._artifact: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        results_path: Path,
        benchmarks_dir: Path,
        model_output_path: Path,
        n_estimators: int = 100,
        random_state: int = 42,
        coverage_threshold: float = COVERAGE_QUALITY_THRESHOLD,
    ) -> Dict:
        """
        Train the suitability model from existing benchmark results.

        Steps:
          1. Load paired (LLM vs template) results from results.json
          2. Re-parse benchmark source files to extract code features
          3. Create binary labels (1 = LLM achieves high coverage, 0 = lower coverage)
          4. Train Random Forest with 5-fold stratified cross-validation
          5. Fit final model on all data and persist with joblib

        Returns:
            Dict with cross-validation metrics and model info.
        """
        logger.info("Loading paired results from %s", results_path)
        paired_df = self._load_paired_results(results_path, coverage_threshold)

        logger.info("Building feature dataset from %s", benchmarks_dir)
        X, y, _matched_df = self._build_feature_dataset(paired_df, benchmarks_dir)

        logger.info(
            "Dataset: %d samples, %d features | class distribution: %s",
            len(y), X.shape[1], y.value_counts().to_dict(),
        )

        minority_frac = y.value_counts(normalize=True).min()
        if minority_frac < 0.30:
            warnings.warn(
                f"Class imbalance detected: minority class is {minority_frac:.1%} of samples. "
                "Using class_weight='balanced' to compensate.",
                UserWarning,
            )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            max_features="sqrt",
            random_state=random_state,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

        logger.info("Running 5-fold stratified cross-validation...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_validate(
                clf, X, y, cv=cv, scoring=scoring, return_train_score=True
            )

        fold_cms = []
        for train_idx, test_idx in cv.split(X, y):
            clf_fold = RandomForestClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                max_features="sqrt",
                random_state=random_state,
            )
            clf_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
            y_pred = clf_fold.predict(X.iloc[test_idx])
            cm_fold = confusion_matrix(y.iloc[test_idx], y_pred, labels=[0, 1])
            fold_cms.append(cm_fold.tolist())

        agg_cm = np.array(fold_cms).sum(axis=0).tolist()

        logger.info("Training final model on full dataset...")
        clf.fit(X, y)
        self.model = clf

        cv_metrics = {
            key: {"mean": float(values.mean()), "std": float(values.std())}
            for key, values in cv_results.items()
        }

        train_preds = clf.predict(X)
        train_report = classification_report(y, train_preds, output_dict=True)

        actual_threshold = float(paired_df["coverage_threshold_used"].iloc[0])

        artifact = {
            "model": clf,
            "feature_names": FEATURE_NAMES,
            "label_threshold": actual_threshold,
            "label_definition": (
                f"1 = LLM line coverage >= {actual_threshold:.1f}% (high-quality testable); "
                f"0 = below threshold (complex/hard function)"
            ),
            "training_samples": int(len(y)),
            "class_distribution": {int(k): int(v) for k, v in y.value_counts().items()},
            "cv_metrics": cv_metrics,
            "confusion_matrix_cv_aggregate": agg_cm,
            "train_classification_report": train_report,
        }
        self._artifact = artifact

        model_output_path = Path(model_output_path)
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, model_output_path)
        logger.info("Model saved to %s", model_output_path)

        return {
            "training_samples": len(y),
            "class_distribution": artifact["class_distribution"],
            "cv_metrics": cv_metrics,
            "confusion_matrix_cv_aggregate": agg_cm,
            "model_path": str(model_output_path),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load_model(self, model_path: Path) -> None:
        """Load a previously trained model from a joblib file."""
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]
        self._artifact = artifact
        logger.info(
            "Model loaded from %s (%d training samples)",
            model_path,
            artifact.get("training_samples", "?"),
        )

    def predict(self, func: ParsedFunction, language: str) -> str:
        """
        Predict the recommended downstream strategy for a function.

        Returns:
            'llm' or 'template'
        """
        return self.score_function(func, language)["recommended_strategy"]

    def predict_proba(self, func: ParsedFunction, language: str) -> Dict[str, float]:
        """
        Return probability estimates for each strategy.

        Returns:
            {'llm': 0.73, 'template': 0.27}
        """
        return self.score_function(func, language)["probabilities"]

    def score_function(self, func: ParsedFunction, language: str) -> Dict:
        """
        Score a function for LLM suitability.

        Returns:
            Dict containing score, bucket, predicted label, strategy, and features.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load_model() first.")

        features = self.extractor.extract(func, language)
        X = pd.DataFrame([features])[self.feature_names]
        pred = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        proba_map = dict(zip(classes, proba))
        llm_score = float(proba_map.get(1, 0.0))
        template_score = float(proba_map.get(0, 0.0))

        return {
            "llm_suitability_score": llm_score,
            "bucket": self._bucket_for_score(llm_score),
            "predicted_label": "high_llm_success" if pred == 1 else "low_llm_success",
            "recommended_strategy": "llm" if pred == 1 else "template",
            "probabilities": {
                "llm": llm_score,
                "template": template_score,
            },
            "features": features,
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        importances = self.model.feature_importances_
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def get_artifact(self) -> Optional[Dict]:
        """Return the full saved artifact dict."""
        return self._artifact

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bucket_for_score(self, llm_score: float) -> str:
        """Map an LLM suitability score to a screening bucket."""
        if llm_score >= GOOD_CANDIDATE_THRESHOLD:
            return "good_candidate"
        if llm_score >= BORDERLINE_THRESHOLD:
            return "borderline"
        return "risky"

    def _load_paired_results(
        self, results_path: Path, coverage_threshold: float
    ) -> pd.DataFrame:
        """
        Load results.json and join LLM vs template results per function.

        Label definition:
          Label 1: LLM achieves >= threshold line coverage.
          Label 0: LLM achieves < threshold line coverage.

        When coverage_threshold == -1 (default), the median LLM coverage
        across the dataset is used, producing a balanced class split.
        """
        with open(results_path, encoding="utf-8") as f:
            data = json.load(f)

        index: Dict[Tuple[str, str, str], Dict] = {}
        for bench in data.get("benchmarks", []):
            key = (bench["benchmark_name"], bench["language"], bench["generator_type"])
            index[key] = {r["function_name"]: r for r in bench.get("function_results", [])}

        rows = []
        bench_lang_pairs = {(bname, lang) for (bname, lang, _gtype) in index}

        for bname, lang in bench_lang_pairs:
            llm_funcs = index.get((bname, lang, "llm"), {})
            tmpl_funcs = index.get((bname, lang, "template"), {})

            for func_name in llm_funcs:
                if func_name not in tmpl_funcs:
                    continue
                llm_r = llm_funcs[func_name]
                tmpl_r = tmpl_funcs[func_name]
                rows.append(
                    {
                        "benchmark_name": bname,
                        "language": lang,
                        "function_name": func_name,
                        "llm_line_coverage": llm_r.get("line_coverage", 0.0),
                        "llm_branch_coverage": llm_r.get("branch_coverage", 0.0),
                        "llm_test_passed": int(llm_r.get("test_passed", False)),
                        "template_line_coverage": tmpl_r.get("line_coverage", 0.0),
                        "template_branch_coverage": tmpl_r.get("branch_coverage", 0.0),
                        "template_test_passed": int(tmpl_r.get("test_passed", False)),
                    }
                )

        df = pd.DataFrame(rows)

        if coverage_threshold < 0:
            threshold = float(df["llm_line_coverage"].median())
            logger.info(
                "Auto-threshold: median LLM coverage = %.2f%% "
                "(label=1 if llm_line_coverage >= threshold)",
                threshold,
            )
        else:
            threshold = float(coverage_threshold)
            logger.info("Using fixed threshold: %.2f%%", threshold)

        df["label"] = (df["llm_line_coverage"] >= threshold).astype(int)
        df["coverage_threshold_used"] = threshold

        logger.info(
            "Loaded %d paired results | label=1 (high cov): %d, label=0 (low cov): %d",
            len(df),
            df["label"].sum(),
            (df["label"] == 0).sum(),
        )
        return df

    def _build_feature_dataset(
        self, results_df: pd.DataFrame, benchmarks_dir: Path
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Re-parse benchmark source files and extract features for each function.

        Returns:
            (X DataFrame, y Series, matched_df with function metadata)
        """
        benchmarks_dir = Path(benchmarks_dir)
        feature_rows = []
        labels = []
        matched_indices = []

        c_parser = CParser()
        rust_parser = RustParser()
        cpp_parser = CppParser()
        parse_cache: Dict[Path, List[ParsedFunction]] = {}

        for idx, row in results_df.iterrows():
            lang = row["language"]
            bname = row["benchmark_name"]
            func_name = row["function_name"]

            if lang == "c":
                ext, parser = ".c", c_parser
            elif lang == "rust":
                ext, parser = ".rs", rust_parser
            elif lang == "cpp":
                ext, parser = ".cpp", cpp_parser
            else:
                logger.warning("Unknown language '%s' (skipping %s)", lang, func_name)
                continue

            filepath = benchmarks_dir / lang / f"{bname}{ext}"
            if not filepath.exists():
                logger.warning("Benchmark file not found: %s (skipping %s)", filepath, func_name)
                continue

            if filepath not in parse_cache:
                source = filepath.read_text(encoding="utf-8")
                if lang == "cpp":
                    parser = CppParser()
                parse_cache[filepath] = parser.parse_source(source)

            functions = parse_cache[filepath]
            func = next((f for f in functions if f.name == func_name), None)
            if func is None:
                logger.warning("Function '%s' not found in %s (skipping)", func_name, filepath)
                continue

            features = self.extractor.extract(func, lang)
            feature_rows.append(features)
            labels.append(row["label"])
            matched_indices.append(idx)

        total = len(results_df)
        matched = len(feature_rows)
        unmatched = total - matched
        if unmatched > 0:
            logger.warning("%d/%d functions could not be matched to source", unmatched, total)
        if total and unmatched / total > 0.20:
            raise RuntimeError(
                f"Too many unmatched functions ({unmatched}/{total}). "
                "Check that benchmark files match the results.json entries."
            )

        X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
        y = pd.Series(labels, name="label")
        matched_df = results_df.loc[matched_indices].reset_index(drop=True)
        return X, y, matched_df
