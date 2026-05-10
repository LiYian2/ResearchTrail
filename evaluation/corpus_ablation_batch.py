import argparse
import json
import os
import time
from pathlib import Path

from evaluation.benchmark_registry import BENCHMARK_TOPICS
from evaluation.corpus_ablation import VARIANTS, run_ablation


def run_batch(output_dir: str, max_papers: int, ids: list[str] | None = None, enrich: bool = True) -> dict[str, dict]:
    selected = _select_topics(ids)
    os.makedirs(output_dir, exist_ok=True)
    all_results: dict[str, dict] = {}

    for entry in selected:
        topic_id = entry["id"]
        topic_dir = Path(output_dir, topic_id)
        topic_dir.mkdir(parents=True, exist_ok=True)
        try:
            all_results[topic_id] = run_ablation(
                topic_or_id=topic_id,
                output_dir=str(topic_dir),
                max_papers=max_papers,
                user_level=entry.get("user_level", "intermediate"),
                enrich=enrich,
            )
        except Exception as exc:
            all_results[topic_id] = {"error": {"message": f"{type(exc).__name__}: {exc}"}}

        Path(output_dir, "corpus_ablation_batch_results.json").write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        Path(output_dir, "corpus_ablation_batch_summary.md").write_text(
            _summary_markdown(all_results),
            encoding="utf-8",
        )
        time.sleep(10)

    return all_results


def _select_topics(ids: list[str] | None) -> list[dict]:
    if ids:
        wanted = set(ids)
        return [entry for entry in BENCHMARK_TOPICS if entry["id"] in wanted]
    # Run the known low-corpus topic first so rate-limit failures late in the run
    # do not systematically disadvantage it.
    return sorted(BENCHMARK_TOPICS, key=lambda entry: 0 if entry["id"] == "protein_structure" else 1)


def _summary_markdown(all_results: dict[str, dict]) -> str:
    averages = _averages(all_results)
    lines = [
        "# Skill 1 Corpus Construction Ablation: Multi-Topic Benchmark",
        "",
        "This ablation compares corpus construction variants used by the Literature Retrieval Skill.",
        "",
        "## Average Metrics",
        "",
        "| Variant | Topics | Papers | Raw Records | Duplicates Removed | Abstract Coverage | Citation Metadata | Reference Coverage | Landmark Hit | Topic Precision | Graph Edges | Graph Edge Yield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        if variant in averages:
            lines.append(_average_row(variant, averages[variant]))

    lines += [
        "",
        "## Per-Topic Results",
        "",
        "| Topic | Variant | Papers | Raw Records | Duplicates Removed | Abstract Coverage | Citation Metadata | Reference Coverage | Landmark Hit | Topic Precision | Graph Edges | Graph Edge Yield |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topic_id, variants in all_results.items():
        if "error" in variants:
            lines.append(f"| {topic_id} | error | 0 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 0.00 |")
            continue
        for variant, metrics in variants.items():
            lines.append(
                "| {topic} | {variant} | {papers} | {raw} | {dupes} | {abstract:.1%} | {citation:.1%} | {refs:.1%} | {landmark:.1%} | {precision:.1%} | {edges} | {yield_:.2f} |".format(
                    topic=topic_id,
                    variant=variant,
                    papers=metrics["total_papers"],
                    raw=metrics["raw_records"],
                    dupes=metrics["duplicate_removal"],
                    abstract=metrics["abstract_coverage"],
                    citation=metrics["citation_metadata_coverage"],
                    refs=metrics["reference_coverage"],
                    landmark=metrics["landmark_hit_rate"],
                    precision=metrics["topic_precision"],
                    edges=metrics["graph_edges"],
                    yield_=metrics["graph_edge_yield"],
                )
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Higher paper count is useful only when topic precision remains acceptable.",
        "- Duplicate removal measures how much redundant API overlap was cleaned before graph construction.",
        "- Citation/reference coverage measures whether the corpus can support citation-aware downstream analysis.",
        "- Graph edge yield is the key downstream-readiness metric: it measures whether retrieved papers can form a usable graph rather than a disconnected list.",
        "- The verified-landmark variant is designed to improve recall of known milestones, not to maximize every metric simultaneously.",
    ]
    return "\n".join(lines) + "\n"


def _averages(all_results: dict[str, dict]) -> dict[str, dict[str, float]]:
    keys = [
        "total_papers",
        "raw_records",
        "duplicate_removal",
        "abstract_coverage",
        "citation_metadata_coverage",
        "reference_coverage",
        "landmark_hit_rate",
        "topic_precision",
        "graph_edges",
        "graph_edge_yield",
    ]
    out: dict[str, dict[str, float]] = {}
    for variant in VARIANTS:
        rows = [
            variants[variant]
            for variants in all_results.values()
            if variant in variants and "error" not in variants
        ]
        if not rows:
            continue
        out[variant] = {"topic_count": len(rows)}
        for key in keys:
            out[variant][key] = sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    return out


def _average_row(variant: str, metrics: dict[str, float]) -> str:
    return (
        "| {variant} | {topics:.0f} | {papers:.1f} | {raw:.1f} | {dupes:.1f} | {abstract:.1%} | {citation:.1%} | {refs:.1%} | {landmark:.1%} | {precision:.1%} | {edges:.1f} | {yield_:.2f} |"
    ).format(
        variant=variant,
        topics=metrics["topic_count"],
        papers=metrics["total_papers"],
        raw=metrics["raw_records"],
        dupes=metrics["duplicate_removal"],
        abstract=metrics["abstract_coverage"],
        citation=metrics["citation_metadata_coverage"],
        refs=metrics["reference_coverage"],
        landmark=metrics["landmark_hit_rate"],
        precision=metrics["topic_precision"],
        edges=metrics["graph_edges"],
        yield_=metrics["graph_edge_yield"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-topic Skill 1 corpus ablation.")
    parser.add_argument("--output-dir", default="outputs/corpus_ablation/full_12_topics")
    parser.add_argument("--max-papers", type=int, default=45)
    parser.add_argument("--ids", nargs="*", help="Optional benchmark topic IDs.")
    parser.add_argument("--no-enrich", action="store_true", help="Disable citation/reference enrichment.")
    args = parser.parse_args()
    results = run_batch(args.output_dir, args.max_papers, args.ids, enrich=not args.no_enrich)
    print(_summary_markdown(results))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
