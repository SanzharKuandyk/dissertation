"""
Generate a small unseen-code case study for the thesis.

The goal is not to estimate generalization statistically. Instead, this script
screens a few external open-source files that were not part of the benchmark
corpus and packages the outputs into a reproducible report plus a simple figure
for the dissertation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from mltest.ml.strategy_selector import MLStrategySelector
from mltest.parsers.c_parser import CParser
from mltest.parsers.cpp_parser import CppParser
from mltest.parsers.rust_parser import RustParser


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "strategy_selector.joblib"
SOURCE_DIR = ROOT / "external_case_study" / "sources"
RESULTS_PATH = ROOT / "results" / "external_case_study.json"
GRAPH_PATH = ROOT / "graphs" / "external_case_study_breakdown.png"

BUCKET_ORDER = ["good_candidate", "borderline", "risky"]
BUCKET_COLORS = {
    "good_candidate": "#1f7a4d",
    "borderline": "#b7791f",
    "risky": "#b83232",
}
LANGUAGE_LABELS = {"c": "C", "cpp": "C++", "rust": "Rust"}


@dataclass(frozen=True)
class CaseStudySource:
    source_id: str
    language: str
    repo: str
    commit: str
    relative_path: str
    local_filename: str
    note: str

    @property
    def local_path(self) -> Path:
        return SOURCE_DIR / self.local_filename

    @property
    def source_url(self) -> str:
        return (
            "https://raw.githubusercontent.com/"
            f"{self.repo}/{self.commit}/{self.relative_path}"
        )

    @property
    def display_name(self) -> str:
        return f"{LANGUAGE_LABELS[self.language]} / {self.repo} / {self.relative_path}"


CASE_STUDY_SOURCES = [
    CaseStudySource(
        source_id="curl_strcase",
        language="c",
        repo="curl/curl",
        commit="aafbe089a88c42bf81ff8cf868f20263661ad3e4",
        relative_path="lib/strcase.c",
        local_filename="curl_strcase.c",
        note="Utility-style C string comparison helpers from a mature systems project.",
    ),
    CaseStudySource(
        source_id="fmt_os",
        language="cpp",
        repo="fmtlib/fmt",
        commit="2f18a88e68af42bdda0e5210f80fac07ae756bd6",
        relative_path="src/os.cc",
        local_filename="fmt_os.cpp",
        note="Conservative C++ parser scope yields two plain free functions in this file.",
    ),
    CaseStudySource(
        source_id="re2_strutil",
        language="cpp",
        repo="google/re2",
        commit="972a15cedd008d846f1a39b2e88ce48d7f166cbd",
        relative_path="util/strutil.cc",
        local_filename="re2_strutil.cpp",
        note="Single free-function utility file that produces a low-suitability example.",
    ),
    CaseStudySource(
        source_id="percent_encoding",
        language="rust",
        repo="servo/rust-url",
        commit="55ff72dd676ae6657d0b8d7dd005233210da76d2",
        relative_path="percent_encoding/src/lib.rs",
        local_filename="percent_encoding_lib.rs",
        note="Public utility functions from a widely used Rust URL-processing crate.",
    ),
]


def parse_testable_functions(source_code: str, language: str) -> list[Any]:
    if language == "c":
        parser = CParser()
    elif language == "cpp":
        parser = CppParser()
    elif language == "rust":
        parser = RustParser()
    else:
        raise ValueError(f"Unsupported language: {language}")

    parser.parse_source(source_code)
    return parser.get_testable_functions()


def bucket_counts(scored_functions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        bucket: sum(1 for item in scored_functions if item["bucket"] == bucket)
        for bucket in BUCKET_ORDER
    }


def screen_source(
    selector: MLStrategySelector,
    source: CaseStudySource,
) -> dict[str, Any]:
    source_code = source.local_path.read_text(encoding="utf-8", errors="ignore")
    functions = parse_testable_functions(source_code, source.language)
    scored_functions = []

    for func in functions:
        score = selector.score_function(func, source.language)
        scored_functions.append(
            {
                "name": func.name,
                "line_start": int(getattr(func, "line_start", 0) or 0),
                "line_end": int(getattr(func, "line_end", 0) or 0),
                "llm_suitability_score": round(score["llm_suitability_score"], 4),
                "bucket": score["bucket"],
                "predicted_label": score["predicted_label"],
                "raw_signature": getattr(func, "raw_signature", ""),
            }
        )

    scored_functions.sort(
        key=lambda item: item["llm_suitability_score"],
        reverse=True,
    )

    return {
        "source_id": source.source_id,
        "language": source.language,
        "language_label": LANGUAGE_LABELS[source.language],
        "repo": source.repo,
        "commit": source.commit,
        "relative_path": source.relative_path,
        "source_url": source.source_url,
        "display_name": source.display_name,
        "local_path": str(source.local_path.resolve()),
        "note": source.note,
        "summary": {
            "total_functions": len(scored_functions),
            **bucket_counts(scored_functions),
        },
        "functions": scored_functions,
    }


def create_case_study_chart(report: dict[str, Any], output_path: Path) -> None:
    cases = report["cases"]
    if not cases:
        return

    labels = [case["display_name"] for case in cases]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    left = np.zeros(len(labels))

    for bucket in BUCKET_ORDER:
        values = [case["summary"][bucket] for case in cases]
        bars = ax.barh(
            y,
            values,
            left=left,
            color=BUCKET_COLORS[bucket],
            edgecolor="black",
            linewidth=0.8,
            label=bucket.replace("_", " "),
        )

        for bar, value in zip(bars, values):
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                str(value),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
        left += values

    totals = [case["summary"]["total_functions"] for case in cases]
    for ypos, total in zip(y, totals):
        ax.text(
            total + 0.12,
            ypos,
            f"n={total}",
            va="center",
            fontsize=10,
            color="#334155",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Screened function count", fontsize=12, fontweight="bold")
    ax.set_title(
        "External Unseen-Code Case Study: Screening Buckets by Source File",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=10)

    note = (
        "External files were sourced from pinned upstream commits and were not used "
        "to train the Random Forest model."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=10, color="#475569")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_report() -> dict[str, Any]:
    selector = MLStrategySelector()
    selector.load_model(MODEL_PATH)

    cases = [screen_source(selector, source) for source in CASE_STUDY_SOURCES]
    all_functions = [
        {
            "source_id": case["source_id"],
            "display_name": case["display_name"],
            "language": case["language"],
            **func,
        }
        for case in cases
        for func in case["functions"]
    ]

    overall_counts = {
        bucket: sum(case["summary"][bucket] for case in cases)
        for bucket in BUCKET_ORDER
    }

    by_language: dict[str, dict[str, int]] = {}
    for language in LANGUAGE_LABELS:
        language_cases = [case for case in cases if case["language"] == language]
        by_language[language] = {
            "total_functions": sum(case["summary"]["total_functions"] for case in language_cases),
            **{
                bucket: sum(case["summary"][bucket] for case in language_cases)
                for bucket in BUCKET_ORDER
            },
        }

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(MODEL_PATH.resolve()),
        "summary": {
            "total_sources": len(cases),
            "total_functions": len(all_functions),
            **overall_counts,
        },
        "language_breakdown": by_language,
        "cases": cases,
        "highest_scored_functions": sorted(
            all_functions,
            key=lambda item: item["llm_suitability_score"],
            reverse=True,
        )[:5],
        "lowest_scored_functions": sorted(
            all_functions,
            key=lambda item: item["llm_suitability_score"],
        )[:5],
    }
    return report


def main() -> None:
    report = build_report()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    create_case_study_chart(report, GRAPH_PATH)
    print(f"Saved report: {RESULTS_PATH}")
    print(f"Saved chart:  {GRAPH_PATH}")


if __name__ == "__main__":
    main()
