"""
Visualization helpers for evaluation and screening artifacts.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


LEGACY_COMPARISON_GRAPH_NAMES = [
    "summary_dashboard.png",
    "coverage_comparison.png",
    "coverage_distribution.png",
    "execution_time.png",
    "improvement.png",
    "language_comparison.png",
    "pass_rate.png",
    "benchmark_breakdown.png",
    "ml_vs_baselines.png",
    "strategy_distribution.png",
]


def remove_legacy_comparison_graphs(output_dir: Path):
    """Delete legacy routing/comparison graphs that no longer fit the thesis framing."""
    output_dir = Path(output_dir)
    for name in LEGACY_COMPARISON_GRAPH_NAMES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def create_all_charts(data: Dict, output_dir: Path):
    """Generate the legacy comparison chart set from results data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create individual charts
    create_coverage_comparison(data, output_dir / "coverage_comparison.png")
    create_pass_rate_chart(data, output_dir / "pass_rate.png")
    create_benchmark_breakdown(data, output_dir / "benchmark_breakdown.png")
    create_improvement_chart(data, output_dir / "improvement.png")
    create_execution_time_chart(data, output_dir / "execution_time.png")
    create_coverage_distribution(data, output_dir / "coverage_distribution.png")
    create_language_comparison(data, output_dir / "language_comparison.png")
    create_summary_dashboard(data, output_dir / "summary_dashboard.png")


def _language_label(language: str) -> str:
    return {"c": "C", "cpp": "C++", "rust": "Rust"}.get(language, language.upper())


def _benchmark_entries(data: Dict) -> List[Dict]:
    """Normalize benchmark entries to unique per-file rows."""
    benchmarks = data.get("benchmarks", [])
    entries = []

    for benchmark in benchmarks:
        entry = dict(benchmark)
        entry["language_label"] = _language_label(benchmark["language"])
        entry["benchmark_label"] = f"{entry['language_label']} / {benchmark['benchmark_name']}"
        entries.append(entry)

    return sorted(
        entries,
        key=lambda item: (["c", "cpp", "rust"].index(item["language"]), item["benchmark_name"], item["generator_type"]),
    )


def _paired_function_results(data: Dict) -> List[Dict]:
    """Collect per-function LLM/template result pairs for scatter-style comparisons."""
    lookup: Dict[tuple, Dict[str, Any]] = {}

    for benchmark in data.get("benchmarks", []):
        for func in benchmark.get("function_results", []):
            key = (benchmark["benchmark_name"], benchmark["language"], func["function_name"])
            record = lookup.setdefault(
                key,
                {
                    "benchmark_name": benchmark["benchmark_name"],
                    "language": benchmark["language"],
                    "function_name": func["function_name"],
                },
            )
            record[benchmark["generator_type"]] = func

    return [
        record for record in lookup.values()
        if "llm" in record and "template" in record
    ]


def create_coverage_comparison(data: Dict, output_path: Path):
    """Create per-function line-coverage scatter for LLM vs template."""
    paired_results = _paired_function_results(data)
    if not paired_results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    language_colors = {"c": "#1f77b4", "cpp": "#ff7f0e", "rust": "#2ca02c"}
    plotted_languages = []

    for language in ["c", "cpp", "rust"]:
        points = [row for row in paired_results if row["language"] == language]
        if not points:
            continue
        plotted_languages.append(language)
        ax.scatter(
            [row["template"]["line_coverage"] for row in points],
            [row["llm"]["line_coverage"] for row in points],
            label=_language_label(language),
            color=language_colors[language],
            alpha=0.78,
            s=46,
            edgecolors="white",
            linewidths=0.7,
        )

    ax.plot([0, 100], [0, 100], linestyle="--", color="#4a5568", linewidth=1.2, label="Parity line")
    ax.set_xlabel("Template line coverage (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("LLM line coverage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Function Line Coverage: LLM vs Template", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_pass_rate_chart(data: Dict, output_path: Path):
    """Create pie charts for test pass rates"""
    summary = data.get('summary', {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (name, ax) in enumerate([('llm', axes[0]), ('template', axes[1])]):
        if name not in summary:
            continue

        info = summary[name]
        passed = info['total_tests_passed']
        failed = info['total_tests_failed']
        total = passed + failed

        if total == 0:
            continue

        sizes = [passed, failed]
        labels = [f'Passed ({passed})', f'Failed ({failed})']
        colors = ['#2ecc71', '#e74c3c']
        explode = (0.05, 0)

        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90
        )

        title = 'ML-Generated Tests' if name == 'llm' else 'Template Tests'
        ax.set_title(f'{title}\n(Total: {total} tests)', fontsize=12, fontweight='bold')

    plt.suptitle('Test Pass Rate Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_benchmark_breakdown(data: Dict, output_path: Path):
    """Create grouped bar chart showing per-file benchmark results."""
    entries = _benchmark_entries(data)
    if not entries:
        return

    benchmark_data = {}
    for entry in entries:
        label = entry["benchmark_label"]
        benchmark_data.setdefault(label, {})
        benchmark_data[label][entry["generator_type"]] = entry["total_line_coverage"]

    names = list(benchmark_data.keys())
    llm_coverage = [benchmark_data[n].get('llm', 0) for n in names]
    template_coverage = [benchmark_data[n].get('template', 0) for n in names]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, llm_coverage, width, label='ML-Generated',
                   color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, template_coverage, width, label='Template',
                   color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Line Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Coverage by Benchmark File', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_improvement_chart(data: Dict, output_path: Path):
    """Create per-file line-coverage improvement over template."""
    entries = _benchmark_entries(data)
    if not entries:
        return

    benchmark_data = {}
    for entry in entries:
        label = entry["benchmark_label"]
        benchmark_data.setdefault(label, {})
        benchmark_data[label][entry["generator_type"]] = entry["total_line_coverage"]

    labels = []
    values = []
    for label, values_map in benchmark_data.items():
        if "llm" in values_map and "template" in values_map:
            labels.append(label)
            values.append(values_map["llm"] - values_map["template"])

    if not values:
        return

    order = np.argsort(values)
    labels = [labels[idx] for idx in order]
    values = [values[idx] for idx in order]

    fig, ax = plt.subplots(figsize=(11, 6.8))

    colors = ['#27ae60' if v >= 0 else '#c0392b' for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor='black', linewidth=1.0)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Line coverage improvement (percentage points)', fontsize=12, fontweight='bold')
    ax.set_title('Line Coverage Improvement by Benchmark File',
                fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.annotate(f'{val:+.1f}',
                   xy=(width, bar.get_y() + bar.get_height() / 2),
                   xytext=(5 if width >= 0 else -5, 0),
                   textcoords="offset points",
                   ha='left' if width >= 0 else 'right',
                   va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_execution_time_chart(data: Dict, output_path: Path):
    """Create execution time comparison chart"""
    benchmarks = data.get('benchmarks', [])

    if not benchmarks:
        return

    llm_times = []
    template_times = []

    for b in benchmarks:
        if b['generator_type'] == 'llm':
            llm_times.append(b['total_execution_time'])
        else:
            template_times.append(b['total_execution_time'])

    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['ML-Generated', 'Template']
    avg_times = [
        np.mean(llm_times) if llm_times else 0,
        np.mean(template_times) if template_times else 0
    ]

    colors = ['#3498db', '#9b59b6']
    bars = ax.bar(categories, avg_times, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Average Execution Time (s)', fontsize=12, fontweight='bold')
    ax.set_title('Test Execution Time Comparison', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, avg_times):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_coverage_distribution(data: Dict, output_path: Path):
    """Create box plot showing coverage distribution"""
    benchmarks = data.get('benchmarks', [])

    if not benchmarks:
        return

    llm_coverage = []
    template_coverage = []

    for b in benchmarks:
        for func in b.get('function_results', []):
            if b['generator_type'] == 'llm':
                llm_coverage.append(func['line_coverage'])
            else:
                template_coverage.append(func['line_coverage'])

    if not llm_coverage and not template_coverage:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    labels = []
    if llm_coverage:
        data_to_plot.append(llm_coverage)
        labels.append('ML-Generated')
    if template_coverage:
        data_to_plot.append(template_coverage)
        labels.append('Template')

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

    colors = ['#2ecc71', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Line Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Coverage Distribution by Generator Type',
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_language_comparison(data: Dict, output_path: Path):
    """Create chart comparing language-level coverage results."""
    benchmarks = data.get('benchmarks', [])

    if not benchmarks:
        return

    language_order = ['c', 'cpp', 'rust']
    language_labels = {'c': 'C', 'cpp': 'C++', 'rust': 'Rust'}
    lang_data = {lang: {'llm': [], 'template': []} for lang in language_order}

    for b in benchmarks:
        lang = b['language']
        gen_type = b['generator_type']
        if lang in lang_data:
            lang_data[lang][gen_type].append(b['total_line_coverage'])

    available_languages = [
        lang for lang in language_order
        if lang_data[lang]['llm'] or lang_data[lang]['template']
    ]
    if not available_languages:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(available_languages))
    width = 0.35

    llm_values = [
        np.mean(lang_data[lang]['llm']) if lang_data[lang]['llm'] else 0
        for lang in available_languages
    ]
    template_values = [
        np.mean(lang_data[lang]['template']) if lang_data[lang]['template'] else 0
        for lang in available_languages
    ]

    bars1 = ax.bar(x - width/2, llm_values, width, label='LLM-Generated',
                   color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, template_values, width, label='Template',
                   color='#e74c3c', edgecolor='black')

    ax.set_ylabel('Average Line Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Coverage by Language', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([language_labels[lang] for lang in available_languages], fontsize=12)
    ax.legend()
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_dataset_composition_chart(dataset_info: Dict, output_path: Path):
    """Create a two-panel chart for language composition and class balance."""
    language_counts = dataset_info.get("language_counts", {})
    class_distribution = dataset_info.get("class_distribution", {})
    label_threshold = dataset_info.get("label_threshold")

    languages = [lang for lang in ["c", "cpp", "rust"] if language_counts.get(lang, 0) > 0]
    if not languages:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    lang_labels = [_language_label(lang) for lang in languages]
    lang_values = [language_counts[lang] for lang in languages]
    lang_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"][:len(languages)]
    bars = axes[0].bar(lang_labels, lang_values, color=lang_colors, edgecolor="black", linewidth=1.0)
    axes[0].set_title("Training Samples by Language", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Function count", fontsize=12)
    axes[0].set_ylim(0, max(lang_values) * 1.25)
    for bar, value in zip(bars, lang_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(lang_values) * 0.03,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    label0 = int(class_distribution.get(0, class_distribution.get("0", 0)))
    label1 = int(class_distribution.get(1, class_distribution.get("1", 0)))
    class_labels = ["Label 0\nBelow threshold", "Label 1\nAt/above threshold"]
    class_values = [label0, label1]
    class_colors = ["#c0392b", "#27ae60"]
    bars = axes[1].bar(class_labels, class_values, color=class_colors, edgecolor="black", linewidth=1.0)
    axes[1].set_title("Binary Label Balance", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Function count", fontsize=12)
    axes[1].set_ylim(0, max(class_values) * 1.25 if max(class_values) else 1)
    for bar, value in zip(bars, class_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(class_values) * 0.03 if max(class_values) else 0.03,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    threshold_note = ""
    if label_threshold is not None:
        threshold_note = f"Median-based label threshold: {label_threshold:.2f}% LLM line coverage"
    fig.suptitle("Training Dataset Composition and Class Balance", fontsize=15, fontweight="bold", y=0.98)
    if threshold_note:
        fig.text(0.5, 0.02, threshold_note, ha="center", fontsize=10, color="#4a5568")

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_architecture_diagram(output_path: Path):
    """Create a screening-first workflow diagram aligned with the thesis scope."""
    fig, ax = plt.subplots(figsize=(14, 9.4))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9.4)
    ax.axis("off")

    title_color = "#1f2937"
    text_color = "#334155"
    border_color = "#cbd5e1"
    arrow_color = "#475569"

    def add_box(x, y, w, h, title, subtitle, facecolor, edgecolor):
        box = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h * 0.68,
            title,
            fontsize=13,
            fontweight="bold",
            color=title_color,
            ha="center",
            va="center",
        )
        ax.text(
            x + w / 2,
            y + h * 0.35,
            subtitle,
            fontsize=10.5,
            color=text_color,
            ha="center",
            va="center",
        )

    def connect(x1, y1, x2, y2, linestyle="-", lw=2.0):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color=arrow_color,
                lw=lw,
                linestyle=linestyle,
                mutation_scale=16,
            ),
        )

    fig.suptitle("MLTest Screening Workflow", fontsize=17, fontweight="bold", y=0.96)
    ax.text(
        0.8,
        8.45,
        "Primary path: static analysis and suitability screening before generation",
        fontsize=10.5,
        color="#475569",
        ha="left",
    )

    main_panel = mpatches.FancyBboxPatch(
        (0.6, 0.2),
        12.8,
        7.55,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor="#f8fafc",
        edgecolor=border_color,
        linewidth=1.4,
    )
    ax.add_patch(main_panel)

    top_y = 5.8
    box_w = 2.35
    box_h = 1.2

    add_box(0.95, top_y, box_w, box_h, "Source Functions", "C / C++ / Rust", "#e0f2fe", "#7dd3fc")
    add_box(3.65, top_y, box_w, box_h, "Parsers", "Function extraction", "#dbeafe", "#93c5fd")
    add_box(6.35, top_y, box_w, box_h, "Static Features", "22 features per function", "#dcfce7", "#86efac")
    add_box(9.05, top_y, box_w, box_h, "Random Forest", "Suitability score", "#ede9fe", "#c4b5fd")

    connect(3.3, top_y + box_h / 2, 3.65, top_y + box_h / 2)
    connect(6.0, top_y + box_h / 2, 6.35, top_y + box_h / 2)
    connect(8.7, top_y + box_h / 2, 9.05, top_y + box_h / 2)

    ranking_box = mpatches.FancyBboxPatch(
        (8.67, 4.0),
        3.11,
        1.1,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#fff7ed",
        edgecolor="#fdba74",
        linewidth=1.6,
    )
    ax.add_patch(ranking_box)
    ax.text(10.225, 4.63, "Bucketed Ranking", fontsize=13, fontweight="bold", color=title_color, ha="center")
    ax.text(10.225, 4.28, "good / borderline / risky", fontsize=10.5, color=text_color, ha="center")
    connect(10.225, top_y, 10.225, 5.1)

    chip_y = 3.18
    chip_width = 1.15
    chip_gap = 0.22
    chip_start_x = 10.225 - ((3 * chip_width) + (2 * chip_gap)) / 2
    chip_specs = [
        (chip_start_x + 0 * (chip_width + chip_gap), "Good Candidate", "#bbf7d0", "#22c55e"),
        (chip_start_x + 1 * (chip_width + chip_gap), "Borderline", "#fde68a", "#f59e0b"),
        (chip_start_x + 2 * (chip_width + chip_gap), "Risky", "#fecaca", "#ef4444"),
    ]
    for x, label, face, edge in chip_specs:
        chip = mpatches.FancyBboxPatch(
            (x, chip_y),
            chip_width,
            0.42,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.2,
        )
        ax.add_patch(chip)
        ax.text(x + chip_width / 2, chip_y + 0.21, label, fontsize=8.5, color=title_color, ha="center", va="center")

    reports_box = mpatches.FancyBboxPatch(
        (1.15, 4.0),
        5.55,
        1.1,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="#f1f5f9",
        edgecolor="#94a3b8",
        linewidth=1.6,
    )
    ax.add_patch(reports_box)
    ax.text(3.925, 4.75, "Reports and Visualizations", fontsize=13, fontweight="bold", color=title_color, ha="center")
    ax.text(3.925, 4.38, "scores, rankings, language breakdowns, feature plots", fontsize=10.3, color=text_color, ha="center")
    connect(8.65, 4.55, 6.7, 4.55)

    optional_box = mpatches.FancyBboxPatch(
        (7.98, 1.5),
        4.5,
        1.15,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor="white",
        edgecolor="#f87171",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(optional_box)
    ax.text(10.225, 2.22, "Optional Downstream Generation", fontsize=11.8, fontweight="bold", color=title_color, ha="center")
    ax.text(
        10.225,
        1.88,
        "selected functions only:\nLLM / template / run",
        fontsize=9.3,
        color=text_color,
        ha="center",
        va="center",
        linespacing=1.15,
    )
    ax.plot([11.78, 12.35], [4.55, 4.55], color=arrow_color, lw=1.8, linestyle="--")
    ax.annotate(
        "",
        xy=(12.35, 2.65),
        xytext=(12.35, 4.55),
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.8, linestyle="--", mutation_scale=16),
    )

    ax.text(
        7.0,
        0.9,
        "Generation is not the primary thesis claim;\nthe main output is screening support.",
        fontsize=9.8,
        color="#64748b",
        ha="center",
        va="center",
        linespacing=1.3,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()


def create_summary_dashboard(data: Dict, output_path: Path):
    """Create a comprehensive summary dashboard"""
    summary = data.get('summary', {})
    benchmarks = _benchmark_entries(data)

    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Coverage comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'llm' in summary and 'template' in summary:
        categories = ['Line Coverage', 'Branch Coverage', 'Pass Rate']
        llm_vals = [summary['llm']['avg_line_coverage'],
                   summary['llm']['avg_branch_coverage'],
                   summary['llm']['pass_rate']]
        template_vals = [summary['template']['avg_line_coverage'],
                        summary['template']['avg_branch_coverage'],
                        summary['template']['pass_rate']]

        x = np.arange(len(categories))
        width = 0.35

        ax1.bar(x - width/2, llm_vals, width, label='ML-Generated', color='#2ecc71')
        ax1.bar(x + width/2, template_vals, width, label='Template', color='#e74c3c')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.set_ylabel('%')
        ax1.set_title('Legacy Evaluation Summary: LLM vs Template', fontweight='bold')
        ax1.legend()
        ax1.set_ylim(0, 100)

    # 2. Key metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    if 'llm' in summary:
        metrics_text = f"""
        LEGACY EVALUATION SNAPSHOT
        ──────────────────────────
        Total benchmark files: {summary.get('total_benchmarks', 0) // 2}
        Total function pairs:  {summary.get('total_functions_tested', 0) // 2}

        LLM line coverage:     {summary['llm']['avg_line_coverage']:.1f}%
        Template coverage:     {summary['template']['avg_line_coverage']:.1f}%
        LLM pass rate:         {summary['llm']['pass_rate']:.1f}%
        Template pass rate:    {summary['template']['pass_rate']:.1f}%
        """
        ax2.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#ecf0f1'))

    # 3. Per-benchmark bar chart (middle, spans all columns)
    ax3 = fig.add_subplot(gs[1, :])
    benchmark_data = {}
    for entry in benchmarks:
        label = entry["benchmark_label"]
        benchmark_data.setdefault(label, {})
        benchmark_data[label][entry['generator_type']] = entry['total_line_coverage']

    if benchmark_data:
        names = list(benchmark_data.keys())
        llm_cov = [benchmark_data[n].get('llm', 0) for n in names]
        template_cov = [benchmark_data[n].get('template', 0) for n in names]

        x = np.arange(len(names))
        width = 0.35

        ax3.bar(x - width/2, llm_cov, width, label='ML-Generated', color='#2ecc71')
        ax3.bar(x + width/2, template_cov, width, label='Template', color='#e74c3c')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=40, ha='right')
        ax3.set_ylabel('Line Coverage (%)')
        ax3.set_title('Coverage by Benchmark File', fontweight='bold')
        ax3.legend()
        ax3.set_ylim(0, 100)

    # 4. Improvement summary (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])
    if 'improvement' in summary:
        improvement = summary['improvement']
        metrics = ['Line\nCoverage', 'Branch\nCoverage', 'Pass\nRate']
        values = [improvement['line_coverage_improvement'],
                 improvement['branch_coverage_improvement'],
                 improvement['pass_rate_improvement']]
        colors = ['#27ae60' if v >= 0 else '#c0392b' for v in values]

        ax4.barh(metrics, values, color=colors)
        ax4.axvline(x=0, color='black', linewidth=0.8)
        ax4.set_xlabel('Improvement (%)')
        ax4.set_title('ML Improvement over Baseline', fontweight='bold')

    # 5. Function count by language (bottom center)
    ax5 = fig.add_subplot(gs[2, 1])
    lang_counts = {'C': 0, 'C++': 0, 'Rust': 0}
    counted_files = set()
    for entry in benchmarks:
        file_key = (entry['benchmark_name'], entry['language'])
        if file_key in counted_files:
            continue
        counted_files.add(file_key)
        lang_counts[_language_label(entry['language'])] += entry['functions_tested']

    pie_labels = [label for label, count in lang_counts.items() if count > 0]
    pie_values = [lang_counts[label] for label in pie_labels]
    pie_colors = ['#3498db', '#ff7f0e', '#2ca02c'][:len(pie_labels)]
    ax5.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors)
    ax5.set_title('Functions by Language', fontweight='bold')

    # 6. Execution time (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    llm_times = [b['total_execution_time'] for b in benchmarks if b['generator_type'] == 'llm']
    template_times = [b['total_execution_time'] for b in benchmarks if b['generator_type'] == 'template']

    ax6.bar(['ML-Generated', 'Template'],
           [np.mean(llm_times) if llm_times else 0,
            np.mean(template_times) if template_times else 0],
           color=['#3498db', '#9b59b6'])
    ax6.set_ylabel('Avg Time (s)')
    ax6.set_title('Execution Time', fontweight='bold')

    plt.suptitle('Legacy Evaluation Dashboard',
                fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        "summary": {
            "total_benchmarks": 6,
            "total_functions_tested": 50,
            "llm": {
                "avg_line_coverage": 78.5,
                "avg_branch_coverage": 65.2,
                "total_tests_passed": 45,
                "total_tests_failed": 5,
                "pass_rate": 90.0
            },
            "template": {
                "avg_line_coverage": 45.3,
                "avg_branch_coverage": 35.1,
                "total_tests_passed": 30,
                "total_tests_failed": 20,
                "pass_rate": 60.0
            },
            "improvement": {
                "line_coverage_improvement": 33.2,
                "branch_coverage_improvement": 30.1,
                "pass_rate_improvement": 30.0
            }
        },
        "benchmarks": [
            {"benchmark_name": "math_utils", "language": "c", "generator_type": "llm",
             "total_line_coverage": 82.0, "functions_tested": 10, "total_execution_time": 0.5,
             "function_results": [{"function_name": "add", "line_coverage": 85.0}]},
            {"benchmark_name": "math_utils", "language": "c", "generator_type": "template",
             "total_line_coverage": 50.0, "functions_tested": 10, "total_execution_time": 0.3,
             "function_results": [{"function_name": "add", "line_coverage": 50.0}]},
        ]
    }

    output_dir = Path("test_graphs")
    create_all_charts(sample_data, output_dir)
    print(f"Charts saved to {output_dir}")


# ---------------------------------------------------------------------------
# ML Strategy Selector Visualizations
# ---------------------------------------------------------------------------

# Feature category colours — used to distinguish feature types in bar charts
_FEATURE_CATEGORIES = {
    "body_line_count": "structural",
    "param_count": "structural",
    "has_return_value": "structural",
    "body_char_count": "structural",
    "avg_line_length": "structural",
    "cyclomatic_complexity": "complexity",
    "loop_count": "complexity",
    "conditional_count": "complexity",
    "return_count": "complexity",
    "early_return_count": "complexity",
    "nested_depth": "complexity",
    "null_check_count": "complexity",
    "recursion_likely": "complexity",
    "has_pointer_params": "type",
    "has_output_params": "type",
    "param_has_string": "type",
    "param_has_array": "type",
    "has_option_return": "type",
    "has_result_return": "type",
    "has_generic_params": "type",
    "is_async": "type",
    "language": "language",
}

_CATEGORY_COLORS = {
    "structural": "#3498db",
    "complexity": "#e67e22",
    "type": "#27ae60",
    "language": "#9b59b6",
}


def create_feature_importance_chart(model_data: List[Dict], output_path: Path):
    """
    Horizontal bar chart of top-15 feature importances, coloured by category.

    Args:
        model_data: list of {'feature': str, 'importance': float} dicts
                    (from MLStrategySelector.get_feature_importances().to_dict('records'))
        output_path: where to save the PNG
    """
    df = sorted(model_data, key=lambda x: x["importance"], reverse=True)[:15]
    features = [d["feature"] for d in df]
    importances = [d["importance"] for d in df]
    colors = [_CATEGORY_COLORS.get(_FEATURE_CATEGORIES.get(f, "structural"), "#3498db")
              for f in features]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(features)), importances, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(
        "LLM Suitability Predictor - Top Feature Importances\n(Random Forest, Gini impurity)",
        fontsize=14,
        fontweight="bold",
    )

    # Annotate bars
    for bar, val in zip(bars, importances):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)

    # Legend
    legend_handles = [
        mpatches.Patch(color=_CATEGORY_COLORS[cat], label=cat.capitalize())
        for cat in ["structural", "complexity", "type", "language"]
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_ml_confusion_matrix(cm_data: List[List[int]], output_path: Path):
    """
    2×2 seaborn heatmap of the cross-validation aggregate confusion matrix.

    Args:
        cm_data: [[TN, FP], [FN, TP]] — rows = actual, cols = predicted
        output_path: where to save the PNG
    """
    cm = np.array(cm_data)
    total = cm.sum()
    labels = np.array([
        [f"{cm[0,0]}\n({100*cm[0,0]/max(total,1):.1f}%)", f"{cm[0,1]}\n({100*cm[0,1]/max(total,1):.1f}%)"],
        [f"{cm[1,0]}\n({100*cm[1,0]/max(total,1):.1f}%)", f"{cm[1,1]}\n({100*cm[1,1]/max(total,1):.1f}%)"],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", ax=ax,
        xticklabels=["Predicted: Template", "Predicted: LLM"],
        yticklabels=["Actual: Template", "Actual: LLM"],
        linewidths=1, linecolor="white", cbar_kws={"label": "Count"},
    )
    ax.set_title("ML Strategy Selector — Confusion Matrix\n(5-fold CV aggregate)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Actual Strategy", fontsize=12)
    ax.set_xlabel("Predicted Strategy", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_ml_vs_baselines_chart(comparison_data: Dict, output_path: Path):
    """
    Grouped bar chart: always-LLM vs always-template vs ML-guided across key metrics.
    This is the headline chart for the thesis.

    Args:
        comparison_data: the 'comparison' sub-dict from ml_comparison.json
        output_path: where to save the PNG
    """
    always_llm = comparison_data.get("always_llm", {})
    always_tmpl = comparison_data.get("always_template", {})
    ml_guided = comparison_data.get("ml_guided", {})
    total = comparison_data.get("total_functions", 1)

    metrics = ["Avg Line\nCoverage (%)", "Pass Rate (%)", "API Calls\n(% of max)"]
    llm_vals = [
        always_llm.get("avg_line_coverage", 0),
        always_llm.get("pass_rate", 0),
        100.0,  # always uses 100% of possible API calls
    ]
    tmpl_vals = [
        always_tmpl.get("avg_line_coverage", 0),
        always_tmpl.get("pass_rate", 0),
        0.0,
    ]
    ml_vals = [
        ml_guided.get("avg_line_coverage", 0),
        ml_guided.get("pass_rate", 0),
        round(100.0 * ml_guided.get("api_calls", 0) / max(total, 1), 1),
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    bars_llm = ax.bar(x - width, llm_vals, width, label="Always-LLM", color="#e74c3c", alpha=0.85, edgecolor="white")
    bars_tmpl = ax.bar(x, tmpl_vals, width, label="Always-Template", color="#95a5a6", alpha=0.85, edgecolor="white")
    bars_ml = ax.bar(x + width, ml_vals, width, label="ML-Guided", color="#2ecc71", alpha=0.85, edgecolor="white")

    # Annotate bars
    for bars in [bars_llm, bars_tmpl, bars_ml]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("ML-Guided vs Baselines — Performance Comparison\n(lower API calls = more efficient; higher coverage = better quality)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # Annotate API savings
    savings_pct = ml_guided.get("api_savings_pct", 0)
    ax.annotate(
        f"ML saves {savings_pct}% API calls\nvs always-LLM",
        xy=(x[2] + width, ml_vals[2]),
        xytext=(x[2] + width + 0.35, ml_vals[2] + 15),
        arrowprops=dict(arrowstyle="->", color="#2c3e50"),
        fontsize=10, color="#2c3e50",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_strategy_distribution_chart(ml_data: Dict, output_path: Path):
    """
    Two subplots: (left) overall pie of LLM vs template choices;
    (right) stacked bars per benchmark showing strategy proportion.

    Args:
        ml_data: the full ml_comparison.json dict
        output_path: where to save the PNG
    """
    ml_guided = ml_data.get("comparison", ml_data).get("ml_guided", {})
    decisions = ml_data.get("comparison", ml_data).get("per_function_decisions", [])

    llm_count = ml_guided.get("llm_chosen_count", 0)
    tmpl_count = ml_guided.get("template_chosen_count", 0)
    total = llm_count + tmpl_count

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left: overall pie ---
    if total > 0:
        sizes = [llm_count, tmpl_count]
        labels_pie = [f"LLM\n({llm_count} functions)", f"Template\n({tmpl_count} functions)"]
        colors_pie = ["#e74c3c", "#95a5a6"]
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=labels_pie, colors=colors_pie,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(12)
            at.set_fontweight("bold")
    ax1.set_title("Overall Strategy Distribution\n(ML-guided decisions)", fontsize=13, fontweight="bold")

    # --- Right: per-benchmark stacked bars ---
    if decisions:
        bench_counts: Dict[str, Dict[str, int]] = {}
        for d in decisions:
            key = f"{d['benchmark']}\n({d['language']})"
            bench_counts.setdefault(key, {"llm": 0, "template": 0})
            bench_counts[key][d["ml_strategy"]] += 1

        benches = list(bench_counts.keys())
        llm_vals = [bench_counts[b]["llm"] for b in benches]
        tmpl_vals = [bench_counts[b]["template"] for b in benches]
        y_pos = np.arange(len(benches))

        ax2.barh(y_pos, llm_vals, color="#e74c3c", alpha=0.85, label="LLM", edgecolor="white")
        ax2.barh(y_pos, tmpl_vals, left=llm_vals, color="#95a5a6", alpha=0.85, label="Template", edgecolor="white")

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(benches, fontsize=10)
        ax2.set_xlabel("Number of Functions", fontsize=11)
        ax2.set_title("Strategy Chosen per Benchmark", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)

        # Annotate totals
        for i, (l, t) in enumerate(zip(llm_vals, tmpl_vals)):
            ax2.text(l + t + 0.1, i, str(l + t), va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_ml_charts(ml_viz_data: Dict, output_dir: Path):
    """
    Generate all ML strategy selector visualizations.

    Args:
        ml_viz_data: dict with keys:
            - 'comparison': the ml_comparison.json comparison dict
            - 'feature_importances': list of {feature, importance} records
            - 'confusion_matrix_cv': [[TN,FP],[FN,TP]] aggregate from CV
            - 'cv_metrics': CV scoring dict
        output_dir: directory to save PNGs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ML visualizations...")

    feature_importances = ml_viz_data.get("feature_importances", [])
    cm = ml_viz_data.get("confusion_matrix_cv", [[0, 0], [0, 0]])
    comparison = ml_viz_data.get("comparison", {})

    create_feature_importance_chart(
        feature_importances,
        output_dir / "feature_importance.png",
    )
    create_ml_confusion_matrix(
        cm,
        output_dir / "confusion_matrix.png",
    )
    create_ml_vs_baselines_chart(
        comparison,
        output_dir / "ml_vs_baselines.png",
    )
    create_strategy_distribution_chart(
        {"comparison": comparison},
        output_dir / "strategy_distribution.png",
    )


def _collect_screened_functions(screening_data: Dict) -> List[Dict]:
    """Flatten all screened functions from the report structure."""
    if screening_data.get("functions"):
        return screening_data["functions"]

    functions: List[Dict] = []
    for file_entry in screening_data.get("files", []):
        for func in file_entry.get("functions", []):
            merged = dict(func)
            merged.setdefault("language", file_entry.get("language"))
            merged.setdefault("source_file", file_entry.get("source_file"))
            functions.append(merged)
    return functions


def create_llm_suitability_distribution_chart(screening_data: Dict, output_path: Path):
    """Create a histogram of predicted LLM suitability scores."""
    functions = _collect_screened_functions(screening_data)
    if not functions:
        return

    scores = [func["llm_suitability_score"] for func in functions]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=12, color="#1f77b4", edgecolor="white", alpha=0.9)
    ax.axvline(0.70, color="#2ca02c", linestyle="--", linewidth=2, label="Good candidate")
    ax.axvline(0.40, color="#ff7f0e", linestyle="--", linewidth=2, label="Borderline")
    ax.set_title("Distribution of LLM Suitability Scores", fontsize=14, fontweight="bold")
    ax.set_xlabel("LLM suitability score", fontsize=12)
    ax.set_ylabel("Function count", fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_candidate_breakdown_by_language_chart(screening_data: Dict, output_path: Path):
    """Create a stacked bar chart of screening buckets by language."""
    language_breakdown = screening_data.get("language_breakdown", {})
    if not language_breakdown:
        return

    language_order = ["c", "cpp", "rust"]
    labels = {"c": "C", "cpp": "C++", "rust": "Rust"}
    languages = [lang for lang in language_order if lang in language_breakdown]
    if not languages:
        return

    good = [language_breakdown[lang].get("good_candidate_count", 0) for lang in languages]
    borderline = [language_breakdown[lang].get("borderline_count", 0) for lang in languages]
    risky = [language_breakdown[lang].get("risky_count", 0) for lang in languages]
    x = np.arange(len(languages))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, good, color="#2ca02c", edgecolor="white", label="Good candidate")
    ax.bar(x, borderline, bottom=good, color="#ffbf00", edgecolor="white", label="Borderline")
    ax.bar(
        x,
        risky,
        bottom=np.array(good) + np.array(borderline),
        color="#d62728",
        edgecolor="white",
        label="Risky",
    )

    ax.set_title("Candidate Buckets by Language", fontsize=14, fontweight="bold")
    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Function count", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[lang] for lang in languages], fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path.name}")


def create_screening_charts(screening_data: Dict, output_dir: Path):
    """Generate the primary charts for the screening/testability narrative."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_llm_suitability_distribution_chart(
        screening_data,
        output_dir / "llm_suitability_distribution.png",
    )
    create_candidate_breakdown_by_language_chart(
        screening_data,
        output_dir / "llm_candidate_breakdown_by_language.png",
    )
