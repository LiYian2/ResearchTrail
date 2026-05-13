import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

from evaluation.benchmark_registry import get_benchmark
from evaluation.graph_ablation import _collect_graph_metrics
from evaluation.reading_path_ablation import _load_state
from shared.data_layer import SharedDataLayer
from shared.types import Paper
from skill_graph.analysis import GraphAnalysisSkill
from skill_graph.graph_builder import GraphBuilder
from skill_reading_path.path_generator import ReadingPathSkill


DEFAULT_BACKENDS = ["tfidf", "lsa"]


def run_from_state_root(
    state_root: str,
    output_dir: str,
    limit_topics: list[str] | None = None,
    backends: list[str] | None = None,
) -> dict[str, dict]:
    root = Path(state_root)
    state_files = sorted(root.glob("*/state.json"))
    if limit_topics:
        wanted = set(limit_topics)
        state_files = [p for p in state_files if p.parent.name in wanted]

    backends = _resolve_backends(backends)
    os.makedirs(output_dir, exist_ok=True)

    all_results: dict[str, dict] = {}
    for state_path in state_files:
        topic_id = state_path.parent.name
        base_data = _load_state(state_path)
        profile = base_data.get_research_profile()
        benchmark = get_benchmark(topic_id) or get_benchmark(profile.topic if profile else topic_id)
        papers = base_data.get_all_papers()

        topic_results: dict[str, dict] = {}
        for backend in backends:
            data = SharedDataLayer()
            if profile:
                data.set_research_profile(profile)
            data.add_papers([Paper.from_dict(deepcopy(p.to_dict())) for p in papers])
            GraphBuilder(data).build(mode="hybrid", similarity_backend=backend)
            GraphAnalysisSkill(data).run()
            try:
                ReadingPathSkill(data).run()
            except Exception:
                pass
            metrics = _collect_graph_metrics(data, benchmark)
            metrics["requested_backend"] = backend
            metrics["actual_backend"] = data.get_metadata("similarity_backend")
            topic_results[backend] = metrics
        all_results[topic_id] = topic_results

    Path(output_dir, "similarity_backend_ablation_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(output_dir, "similarity_backend_ablation_summary.md").write_text(
        _markdown(all_results),
        encoding="utf-8",
    )
    return all_results


def _resolve_backends(backends: list[str] | None) -> list[str]:
    requested = backends or DEFAULT_BACKENDS
    available = set(GraphBuilder.available_similarity_backends())
    resolved = []
    for backend in requested:
        backend = GraphBuilder._resolve_similarity_backend(backend)
        if backend == "sentence-transformer" and backend not in available:
            continue
        if backend not in resolved:
            resolved.append(backend)
    return resolved or DEFAULT_BACKENDS


def _markdown(all_results: dict[str, dict]) -> str:
    averages = _average_by_backend(all_results)
    lines = [
        "# Skill 2 Similarity Backend Ablation",
        "",
        "This ablation rebuilds the production hybrid graph from the same saved corpora while changing only the semantic similarity backend.",
        "",
        "## Average Metrics",
        "",
        "| Backend | Topics | Edges | Similarity Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Top Bridge Plausibility | Foundation Landmark Hit | Graph Edge Yield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for backend, metrics in averages.items():
        lines.append(
            "| {backend} | {topics:.0f} | {edges:.1f} | {similarity:.1f} | {components:.1f} | {largest:.1%} | {modularity:.4f} | {communities:.1f} | {coverage:.1%} | {bridge:.1%} | {foundation:.1%} | {yield_:.2f} |".format(
                backend=backend,
                topics=metrics["topic_count"],
                edges=metrics["edge_count"],
                similarity=metrics["similarity_edges"],
                components=metrics["connected_components"],
                largest=metrics["largest_component_ratio"],
                modularity=metrics["modularity"],
                communities=metrics["community_count"],
                coverage=metrics["community_coverage_in_reading_path"],
                bridge=metrics["top_bridge_plausibility"],
                foundation=metrics["top_foundation_landmark_hit"],
                yield_=metrics["graph_edge_yield"],
            )
        )

    lines += [
        "",
        "## Per-Topic Results",
        "",
        "| Topic | Backend | Actual Backend | Nodes | Edges | Similarity Edges | Components | Largest Component | Modularity | Path Community Coverage | Foundation Landmark Hit |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topic_id, backends in all_results.items():
        for backend, metrics in backends.items():
            lines.append(
                "| {topic} | {backend} | {actual} | {nodes} | {edges} | {similarity} | {components} | {largest:.1%} | {modularity:.4f} | {coverage:.1%} | {foundation:.1%} |".format(
                    topic=topic_id,
                    backend=backend,
                    actual=metrics.get("actual_backend", backend),
                    nodes=metrics["node_count"],
                    edges=metrics["edge_count"],
                    similarity=metrics["similarity_edges"],
                    components=metrics["connected_components"],
                    largest=metrics["largest_component_ratio"],
                    modularity=metrics["modularity"],
                    coverage=metrics["community_coverage_in_reading_path"],
                    foundation=metrics["top_foundation_landmark_hit"],
                )
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- TF-IDF is lexical and stable; it is good when titles and abstracts share explicit terminology.",
        "- LSA is a local embedding-style backend built from TF-IDF plus dimensionality reduction; it can recover broader semantic neighborhoods while staying offline and reproducible.",
        "- Sentence-transformer can be enabled later if the package/model is installed, but it is intentionally optional so the project does not depend on a large runtime download.",
    ]
    return "\n".join(lines) + "\n"


def _average_by_backend(all_results: dict[str, dict]) -> dict[str, dict[str, float]]:
    averages: dict[str, dict[str, float]] = {}
    backend_names = sorted({backend for result in all_results.values() for backend in result})
    keys = [
        "edge_count",
        "similarity_edges",
        "connected_components",
        "largest_component_ratio",
        "modularity",
        "community_count",
        "community_coverage_in_reading_path",
        "top_bridge_plausibility",
        "top_foundation_landmark_hit",
        "graph_edge_yield",
    ]
    for backend in backend_names:
        rows = [result[backend] for result in all_results.values() if backend in result]
        averages[backend] = {"topic_count": len(rows)}
        for key in keys:
            averages[backend][key] = sum(float(row.get(key, 0.0)) for row in rows) / max(len(rows), 1)
    return averages


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Skill 2 similarity-backend ablation.")
    parser.add_argument("--state-root", required=True, help="Directory containing per-topic state.json files.")
    parser.add_argument("--ids", nargs="*", help="Optional topic IDs when using --state-root.")
    parser.add_argument("--backends", nargs="*", default=DEFAULT_BACKENDS)
    parser.add_argument("--output-dir", default="outputs/similarity_backend_ablation")
    args = parser.parse_args()

    results = run_from_state_root(args.state_root, args.output_dir, args.ids, args.backends)
    print(_markdown(results))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
