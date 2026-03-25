"""
ML Strategy Selector — core ML component of the dissertation.

Trains a Random Forest classifier that learns which test generation strategy
(LLM vs template) achieves higher code coverage for a given function, based
on code complexity features.  At inference time the model selects the strategy
per function, enabling intelligent orchestration that reduces unnecessary API
calls while maintaining near-LLM coverage.
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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report

from .feature_extractor import FunctionFeatureExtractor, FEATURE_NAMES
from ..parsers.c_parser import CParser, CFunction
from ..parsers.rust_parser import RustParser, RustFunction
from ..parsers.cpp_parser import CppParser, CppFunction

logger = logging.getLogger(__name__)

# Coverage quality threshold: functions where LLM achieves >= this percentage
# are labelled 1 ("high-quality testable") and get LLM generation in the
# ML-guided pipeline.  Functions below are labelled 0 ("complex/hard") and
# get template generation to save API cost.  When set to -1 (auto), the
# median LLM coverage across the training set is used, giving a balanced split.
COVERAGE_QUALITY_THRESHOLD = -1  # -1 = use median (recommended)


class MLStrategySelector:
    """
    Trains and applies an ML model to select the best test generation strategy.

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
        strategy = selector.predict(func, language="c")  # 'llm' or 'template'
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: List[str] = FEATURE_NAMES
        self.extractor = FunctionFeatureExtractor()
        self._artifact: Optional[Dict] = None  # full saved artifact dict

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
        Train the strategy selector from existing benchmark results.

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
        X, y, matched_df = self._build_feature_dataset(paired_df, benchmarks_dir)

        logger.info(
            "Dataset: %d samples, %d features | class distribution: %s",
            len(y), X.shape[1], y.value_counts().to_dict()
        )

        # Warn if class imbalance is severe
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

        # Compute per-fold confusion matrices (handle single-class folds gracefully)
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

        # Aggregate confusion matrix across folds
        agg_cm = np.array(fold_cms).sum(axis=0).tolist()

        # Train final model on all data
        logger.info("Training final model on full dataset...")
        clf.fit(X, y)
        self.model = clf

        # Build metrics dict
        cv_metrics = {
            k: {"mean": float(v.mean()), "std": float(v.std())}
            for k, v in cv_results.items()
        }

        # Full classification report on training data (informational)
        train_preds = clf.predict(X)
        train_report = classification_report(y, train_preds, output_dict=True)

        # Resolve actual threshold used (may be auto/median)
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
        logger.info("Model loaded from %s (%d training samples)",
                    model_path, artifact.get("training_samples", "?"))

    def predict(
        self, func: Union[CFunction, RustFunction], language: str
    ) -> str:
        """
        Predict the best test generation strategy for a function.

        Returns:
            'llm' or 'template'
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load_model() first.")
        features = self.extractor.extract(func, language)
        X = pd.DataFrame([features])[self.feature_names]
        pred = self.model.predict(X)[0]
        return "llm" if pred == 1 else "template"

    def predict_proba(
        self, func: Union[CFunction, RustFunction], language: str
    ) -> Dict[str, float]:
        """
        Return probability estimates for each strategy.

        Returns:
            {'llm': 0.73, 'template': 0.27}
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load_model() first.")
        features = self.extractor.extract(func, language)
        X = pd.DataFrame([features])[self.feature_names]
        proba = self.model.predict_proba(X)[0]
        # classes_ is [0, 1] → [template, llm]
        classes = self.model.classes_
        proba_map = dict(zip(classes, proba))
        return {
            "llm": float(proba_map.get(1, 0.0)),
            "template": float(proba_map.get(0, 0.0)),
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def get_artifact(self) -> Optional[Dict]:
        """Return the full saved artifact dict (includes CV metrics etc.)."""
        return self._artifact

    # ------------------------------------------------------------------
    # Private: data loading
    # ------------------------------------------------------------------

    def _load_paired_results(
        self, results_path: Path, coverage_threshold: float
    ) -> pd.DataFrame:
        """
        Load results.json and join LLM vs template results per function.

        Label definition (binary classification):
          Label 1 ("high-quality"): LLM achieves >= threshold line coverage.
                                    These functions are "LLM-testable" — ML
                                    predicts high coverage, pipeline uses LLM.
          Label 0 ("complex/hard"):  LLM achieves < threshold line coverage.
                                    Pipeline uses template to save API cost.

        When coverage_threshold == -1 (default), the median LLM coverage
        across the dataset is used, producing a balanced 50/50 class split.

        Returns a DataFrame with one row per unique (benchmark, language, function).
        """
        with open(results_path) as f:
            data = json.load(f)

        # Index: {(benchmark_name, language, generator_type): {func_name: result_dict}}
        index: Dict[Tuple, Dict] = {}
        for bench in data.get("benchmarks", []):
            key = (bench["benchmark_name"], bench["language"], bench["generator_type"])
            index[key] = {
                r["function_name"]: r
                for r in bench.get("function_results", [])
            }

        rows = []
        bench_lang_pairs = set()
        for (bname, lang, _gtype) in index:
            bench_lang_pairs.add((bname, lang))

        for bname, lang in bench_lang_pairs:
            llm_funcs = index.get((bname, lang, "llm"), {})
            tmpl_funcs = index.get((bname, lang, "template"), {})

            for func_name in llm_funcs:
                if func_name not in tmpl_funcs:
                    continue
                llm_r = llm_funcs[func_name]
                tmpl_r = tmpl_funcs[func_name]
                rows.append({
                    "benchmark_name": bname,
                    "language": lang,
                    "function_name": func_name,
                    "llm_line_coverage": llm_r.get("line_coverage", 0.0),
                    "llm_branch_coverage": llm_r.get("branch_coverage", 0.0),
                    "llm_test_passed": int(llm_r.get("test_passed", False)),
                    "template_line_coverage": tmpl_r.get("line_coverage", 0.0),
                    "template_branch_coverage": tmpl_r.get("branch_coverage", 0.0),
                    "template_test_passed": int(tmpl_r.get("test_passed", False)),
                })

        df = pd.DataFrame(rows)

        # Determine threshold (auto = median split for balanced classes)
        if coverage_threshold < 0:
            threshold = float(df["llm_line_coverage"].median())
            logger.info(
                "Auto-threshold: median LLM coverage = %.2f%% "
                "(label=1 if llm_line_coverage >= threshold)", threshold
            )
        else:
            threshold = float(coverage_threshold)
            logger.info("Using fixed threshold: %.2f%%", threshold)

        df["label"] = (df["llm_line_coverage"] >= threshold).astype(int)
        df["coverage_threshold_used"] = threshold

        logger.info(
            "Loaded %d paired results | label=1 (high cov): %d, label=0 (low cov): %d",
            len(df), df["label"].sum(), (df["label"] == 0).sum()
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

        # Cache parsed files to avoid re-parsing the same file multiple times
        parse_cache: Dict[Path, List] = {}

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
                # CppParser is stateful — use a fresh instance per file
                if lang == "cpp":
                    p = CppParser()
                    parse_cache[filepath] = p.parse_source(source)
                else:
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
        if unmatched / total > 0.20:
            raise RuntimeError(
                f"Too many unmatched functions ({unmatched}/{total}). "
                "Check that benchmark files match the results.json entries."
            )

        X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
        y = pd.Series(labels, name="label")
        matched_df = results_df.loc[matched_indices].reset_index(drop=True)
        return X, y, matched_df
