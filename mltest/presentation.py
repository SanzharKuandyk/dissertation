"""
Presentation helpers for progress-review dashboards.
"""

from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable


LANGUAGE_LABELS = {"c": "C", "cpp": "C++", "rust": "Rust"}
BUCKET_COLORS = {
    "good_candidate": "#1f7a4d",
    "borderline": "#b7791f",
    "risky": "#b83232",
}


def build_progress_dashboard(
    report_path: Path,
    graphs_dir: Path,
    output_path: Path,
    title: str = "MLTest Progress Dashboard",
) -> Path:
    """Build a static HTML dashboard for progress presentations."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    language_breakdown = report.get("language_breakdown", {})
    functions = _collect_functions(report)

    top_candidates = sorted(
        functions,
        key=lambda item: item.get("llm_suitability_score", 0.0),
        reverse=True,
    )[:8]
    risky_candidates = sorted(
        functions,
        key=lambda item: item.get("llm_suitability_score", 0.0),
    )[:8]

    existing_graphs = [
        ("System Architecture", graphs_dir / "architecture_diagram.png"),
        ("Feature Importance", graphs_dir / "feature_importance.png"),
        ("Suitability Distribution", graphs_dir / "llm_suitability_distribution.png"),
        ("Bucket Breakdown by Language", graphs_dir / "llm_candidate_breakdown_by_language.png"),
    ]
    graph_blocks = [
        _render_graph_card(label, path, output_path.parent)
        for label, path in existing_graphs
        if path.exists()
    ]

    generated_at = report.get("generated_at")
    if generated_at:
        try:
            generated_at = datetime.fromisoformat(generated_at).strftime("%B %d, %Y %H:%M")
        except ValueError:
            pass

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #5c6773;
      --accent: #0f4c5c;
      --accent-soft: #e2edf0;
      --good: #1f7a4d;
      --borderline: #b7791f;
      --risky: #b83232;
      --line: #d7d2c4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, #fff7da 0, transparent 32%),
        linear-gradient(180deg, #f6f1e7 0%, #ece6d8 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, #163b47 0%, #2b6574 100%);
      color: white;
      padding: 28px;
      border-radius: 20px;
      box-shadow: 0 18px 40px rgba(16, 42, 51, 0.18);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      line-height: 1.15;
    }}
    .hero p {{
      margin: 0;
      max-width: 880px;
      font-size: 1.05rem;
      color: rgba(255, 255, 255, 0.88);
    }}
    .meta {{
      margin-top: 14px;
      font-size: 0.95rem;
      color: rgba(255, 255, 255, 0.82);
    }}
    .grid {{
      display: grid;
      gap: 18px;
      margin-top: 22px;
    }}
    .cards {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }}
    .two {{
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 10px 26px rgba(28, 39, 49, 0.08);
    }}
    .panel h2, .panel h3 {{
      margin: 0 0 12px;
      line-height: 1.2;
    }}
    .stat {{
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .stat .label {{
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .stat .value {{
      font-size: 2rem;
      font-weight: 700;
    }}
    .good {{ color: var(--good); }}
    .borderline {{ color: var(--borderline); }}
    .risky {{ color: var(--risky); }}
    .muted {{
      color: var(--muted);
    }}
    .language-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    .language-card {{
      padding: 14px;
      border-radius: 14px;
      background: #f8f5ee;
      border: 1px solid var(--line);
    }}
    .language-card h3 {{
      margin-bottom: 10px;
    }}
    .mini {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 0.95rem;
      padding: 4px 0;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
      line-height: 1.5;
    }}
    li + li {{
      margin-top: 8px;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      background: var(--accent-soft);
      padding: 2px 6px;
      border-radius: 6px;
    }}
    .graphs {{
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .graph-card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
    }}
    .graph-card h3 {{
      margin-bottom: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }}
    .bucket-pill {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      color: white;
      font-size: 0.8rem;
      font-weight: 700;
    }}
    .footer {{
      margin-top: 24px;
      font-size: 0.92rem;
      color: var(--muted);
      text-align: center;
    }}
    @media (max-width: 720px) {{
      .page {{ padding: 18px 14px 34px; }}
      .hero h1 {{ font-size: 1.6rem; }}
      .panel {{ padding: 16px; }}
      table {{ font-size: 0.88rem; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>
        Midterm review view of the dissertation project: static-code-based prediction
        of LLM suitability for unit test generation in C, C++, and Rust.
      </p>
      <div class="meta">
        {html.escape(f"Generated from {report_path}")}<br>
        {html.escape(f"Screening report timestamp: {generated_at or 'unknown'}")}
      </div>
    </section>

    <section class="grid cards">
      {_render_stat_card("Total Functions", summary.get("total_functions", 0), "")}
      {_render_stat_card("Good Candidates", summary.get("good_candidate_count", 0), "good")}
      {_render_stat_card("Borderline", summary.get("borderline_count", 0), "borderline")}
      {_render_stat_card("Risky", summary.get("risky_count", 0), "risky")}
    </section>

    <section class="grid two">
      <div class="panel">
        <h2>Current Goal</h2>
        <ul>
          <li>Present the dissertation as a screening-first system rather than a broad template-vs-LLM comparison.</li>
          <li>Show that the implemented model can rank functions by likely LLM usefulness before expensive generation.</li>
          <li>Demonstrate that the remaining work is final polishing, defense preparation, and presentation refinement.</li>
        </ul>
      </div>
      <div class="panel">
        <h2>Recommended Live Demo</h2>
        <ul>
          <li>Run <code>python -m mltest screen benchmarks/cpp/math_utils.cpp</code>.</li>
          <li>Explain the score, bucket, and why static screening is safer than a live cloud-generation demo.</li>
          <li>Use this dashboard as the visual summary while answering progress questions.</li>
        </ul>
      </div>
    </section>

    <section class="panel">
      <h2>Language Breakdown</h2>
      <div class="language-grid">
        {_render_language_cards(language_breakdown)}
      </div>
    </section>

    <section class="grid graphs">
      {''.join(graph_blocks)}
    </section>

    <section class="grid two">
      <div class="panel">
        <h2>What Is Already Done</h2>
        <ul>
          <li>Benchmark corpus prepared across C, C++, and Rust.</li>
          <li>Static feature extraction and trained Random Forest model are integrated into the project.</li>
          <li>Runnable screening workflow is available through <code>mltest screen</code>.</li>
          <li>Primary figures and evaluation artifacts have been generated.</li>
          <li>The thesis manuscript has already been rewritten around the narrowed scope.</li>
        </ul>
      </div>
      <div class="panel">
        <h2>What Still Needs To Be Done</h2>
        <ul>
          <li>Final polishing of slides, narration, and defense wording.</li>
          <li>Short stable demo preparation around the screening workflow.</li>
          <li>Final cleanup of appendix vs. main-claim materials in the defense assets.</li>
          <li>Practice concise answers for scope, model target, usefulness, and limitations.</li>
        </ul>
      </div>
    </section>

    <section class="grid two">
      <div class="panel">
        <h2>Top Candidate Functions</h2>
        {_render_function_table(top_candidates)}
      </div>
      <div class="panel">
        <h2>Lowest-Score Functions</h2>
        {_render_function_table(risky_candidates)}
      </div>
    </section>

    <section class="panel">
      <h2>Useful Short Answer For Review</h2>
      <p class="muted">
        The project is no longer being defined from scratch. The benchmark corpus,
        feature extraction, model-based screening workflow, figures, and revised
        thesis narrative are already in place. The remaining work is mainly final
        polishing, presentation refinement, and defense preparation.
      </p>
    </section>

    <div class="footer">
      MLTest progress dashboard for dissertation review.
    </div>
  </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _collect_functions(report: dict) -> list[dict]:
    functions = []
    if report.get("functions"):
        functions.extend(report["functions"])

    for file_entry in report.get("files", []):
        for func in file_entry.get("functions", []):
            merged = dict(func)
            merged.setdefault("language", file_entry.get("language"))
            merged.setdefault("benchmark_name", file_entry.get("benchmark_name"))
            functions.append(merged)

    return functions


def _render_stat_card(label: str, value: int, class_name: str) -> str:
    return (
        '<div class="panel stat">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value {class_name}">{html.escape(str(value))}</div>'
        "</div>"
    )


def _render_language_cards(language_breakdown: dict) -> str:
    parts = []
    for language in ("c", "cpp", "rust"):
        data = language_breakdown.get(language)
        if not data:
            continue
        parts.append(
            '<div class="language-card">'
            f"<h3>{html.escape(LANGUAGE_LABELS.get(language, language))}</h3>"
            f'<div class="mini"><span>Total</span><strong>{data.get("total_functions", 0)}</strong></div>'
            f'<div class="mini"><span class="good">Good</span><strong>{data.get("good_candidate_count", 0)}</strong></div>'
            f'<div class="mini"><span class="borderline">Borderline</span><strong>{data.get("borderline_count", 0)}</strong></div>'
            f'<div class="mini"><span class="risky">Risky</span><strong>{data.get("risky_count", 0)}</strong></div>'
            "</div>"
        )
    return "".join(parts)


def _render_graph_card(label: str, image_path: Path, output_dir: Path) -> str:
    relative = Path(os.path.relpath(image_path, output_dir))
    return (
        '<div class="panel graph-card">'
        f"<h3>{html.escape(label)}</h3>"
        f'<img src="{html.escape(relative.as_posix())}" alt="{html.escape(label)}">'
        "</div>"
    )


def _render_function_table(functions: Iterable[dict]) -> str:
    rows = []
    for func in functions:
        bucket = func.get("bucket", "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(func.get('name', 'unknown'))}</td>"
            f"<td>{html.escape(LANGUAGE_LABELS.get(func.get('language'), func.get('language', '')))}</td>"
            f"<td>{html.escape(func.get('benchmark_name', ''))}</td>"
            f"<td>{func.get('llm_suitability_score', 0.0):.3f}</td>"
            f'<td><span class="bucket-pill" style="background:{BUCKET_COLORS.get(bucket, "#4a5568")}">{html.escape(bucket)}</span></td>'
            "</tr>"
        )

    if not rows:
        rows.append('<tr><td colspan="5" class="muted">No functions available.</td></tr>')

    return (
        "<table>"
        "<thead><tr><th>Function</th><th>Language</th><th>Benchmark</th><th>Score</th><th>Bucket</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
