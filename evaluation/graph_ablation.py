import argparse
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from evaluation.benchmark_registry import get_benchmark
from evaluation.reading_path_ablation import _load_state
from shared.data_layer import SharedDataLayer
from shared.types import Paper, ResearchProfile, UserLevel
from skill_graph.analysis import GraphAnalysisSkill
from skill_graph.graph_builder import GraphBuilder
from skill_reading_path.path_generator import ReadingPathSkill
from skill_retrieval.retrieval import LiteratureRetrievalSkill


GRAPH_MODES = ["citation", "similarity", "hybrid"]


def run_graph_ablation(
    topic_or_id: str,
    output_dir: str,
    max_papers: int,
    user_level: str,
    demo: bool,
) -> dict[str, dict]:
    benchmark = get_benchmark(topic_or_id)
    topic = benchmark["topic"] if benchmark else topic_or_id
    profile = ResearchProfile(
        topic=topic,
        user_level=_user_level(user_level or (benchmark or {}).get("user_level", "intermediate")),
        max_papers=max_papers,
        preferred_length=12,
        goal="graph_mode_ablation",
    )

    base_data = SharedDataLayer()
    retrieval = LiteratureRetrievalSkill(base_data)
    retrieval.run(profile, query_plan=_query_plan(topic, benchmark), demo=demo)
    papers = base_data.get_all_papers()

    results: dict[str, dict] = {}
    for mode in GRAPH_MODES:
        data = SharedDataLayer()
        data.set_research_profile(profile)
        data.add_papers([Paper.from_dict(deepcopy(p.to_dict())) for p in papers])
        GraphBuilder(data).build(mode=mode)
        GraphAnalysisSkill(data).run()
        try:
            ReadingPathSkill(data).run()
        except Exception:
            pass
        results[mode] = _collect_graph_metrics(data, benchmark)

    os.makedirs(output_dir, exist_ok=True)
    Path(output_dir, "graph_ablation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(output_dir, "graph_ablation_summary.md").write_text(
        _markdown(topic, results),
        encoding="utf-8",
    )
    return results


def run_from_state_root(state_root: str, output_dir: str, limit_topics: list[str] | None = None) -> dict[str, dict]:
    root = Path(state_root)
    state_files = sorted(root.glob("*/state.json"))
    if limit_topics:
        wanted = set(limit_topics)
        state_files = [p for p in state_files if p.parent.name in wanted]

    os.makedirs(output_dir, exist_ok=True)
    all_results: dict[str, dict] = {}
    for state_path in state_files:
        topic_id = state_path.parent.name
        base_data = _load_state(state_path)
        profile = base_data.get_research_profile()
        topic = profile.topic if profile else topic_id
        benchmark = get_benchmark(topic) or get_benchmark(topic_id)
        papers = base_data.get_all_papers()

        topic_results: dict[str, dict] = {}
        for mode in GRAPH_MODES:
            data = SharedDataLayer()
            if profile:
                data.set_research_profile(profile)
            data.add_papers([Paper.from_dict(deepcopy(p.to_dict())) for p in papers])
            GraphBuilder(data).build(mode=mode)
            GraphAnalysisSkill(data).run()
            try:
                ReadingPathSkill(data).run()
            except Exception:
                pass
            topic_results[mode] = _collect_graph_metrics(data, benchmark)
        all_results[topic_id] = topic_results

    Path(output_dir, "graph_ablation_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(output_dir, "graph_ablation_summary.md").write_text(
        _batch_markdown(all_results),
        encoding="utf-8",
    )
    return all_results


def _collect_graph_metrics(data: SharedDataLayer, benchmark: dict | None) -> dict[str, Any]:
    graph_metrics = data.get_metadata("graph_metrics") or {}
    graph = data.get_graph_data()
    scores = data.get_all_scores()
    path = data.get_reading_path()
    n_nodes = len(graph.nodes) if graph else 0
    n_edges = len(graph.edges) if graph else 0
    top_bridge = _top_bridge(data)
    top_foundation = _top_foundation(data)
    return {
        "node_count": n_nodes,
        "edge_count": n_edges,
        "citation_edges": graph_metrics.get("citation_edges", 0),
        "similarity_edges": graph_metrics.get("similarity_edges", 0),
        "connected_components": graph_metrics.get("connected_components", 0),
        "largest_component_ratio": graph_metrics.get("largest_component_ratio", 0.0),
        "modularity": graph_metrics.get("modularity", 0.0),
        "community_count": graph_metrics.get("community_count", 0),
        "community_coverage_in_reading_path": _community_coverage(path, scores),
        "top_bridge_paper": top_bridge.get("title", ""),
        "top_bridge_plausibility": top_bridge.get("plausibility", 0.0),
        "top_foundation_paper": top_foundation.get("title", ""),
        "top_foundation_landmark_hit": _top_foundation_landmark_hit(data, benchmark),
        "graph_edge_yield": n_edges / max(n_nodes, 1),
    }


def _top_bridge(data: SharedDataLayer) -> dict[str, Any]:
    scores = data.get_all_scores()
    graph_data = data.get_graph_data()
    if not scores or not graph_data:
        return {"title": "", "plausibility": 0.0}
    pid, score = max(scores.items(), key=lambda item: item[1].bridge_score)
    paper = data.get_paper(pid)
    neighbors = set()
    for edge in graph_data.edges:
        if edge.source == pid:
            neighbors.add(edge.target)
        elif edge.target == pid:
            neighbors.add(edge.source)
    own = score.community
    cross = sum(1 for nb in neighbors if nb in scores and scores[nb].community != own)
    cross_ratio = cross / max(len(neighbors), 1)
    plausibility = min(1.0, 0.45 * min(1.0, score.betweenness * 5) + 0.35 * cross_ratio + 0.20 * min(1.0, len(neighbors) / 8))
    return {
        "title": paper.title if paper else pid,
        "plausibility": round(plausibility, 4),
    }


def _top_foundation(data: SharedDataLayer) -> dict[str, Any]:
    scores = data.get_all_scores()
    if not scores:
        return {"title": ""}
    pid, _ = max(scores.items(), key=lambda item: item[1].foundation_score)
    paper = data.get_paper(pid)
    return {"title": paper.title if paper else pid}


def _top_foundation_landmark_hit(data: SharedDataLayer, benchmark: dict | None) -> float:
    landmarks = (benchmark or {}).get("expected_landmarks", [])
    if not landmarks:
        return 0.0
    top_titles = [
        data.get_paper(pid).title
        for pid, _ in sorted(
            data.get_all_scores().items(),
            key=lambda item: item[1].foundation_score,
            reverse=True,
        )[:3]
        if data.get_paper(pid)
    ]
    normalized = [_normalize_title(title) for title in top_titles]
    for item in landmarks:
        if item.get("role") not in {"foundation", "core"}:
            continue
        target = _normalize_title(item.get("title", ""))
        if target and any(target in title or title in target for title in normalized):
            return 1.0
    return 0.0


def _community_coverage(path, scores: dict) -> float:
    all_communities = {score.community for score in scores.values() if score.community >= 0}
    if not all_communities or not path:
        return 0.0
    path_communities = set()
    for stage in path.stages:
        for paper in stage.get("papers", []):
            pid = paper.get("paper_id")
            if pid in scores and scores[pid].community >= 0:
                path_communities.add(scores[pid].community)
    return len(path_communities) / len(all_communities)


def _query_plan(topic: str, benchmark: dict | None) -> dict | None:
    if not benchmark:
        return None
    landmarks = [item["title"] for item in benchmark.get("expected_landmarks", []) if item.get("title")]
    topic_terms = [term for term in benchmark.get("topic_terms", []) if term]
    main_queries = [topic]
    main_queries.extend(f'"{title}"' for title in landmarks[:5])
    main_queries.extend(term for term in topic_terms if " " in term or "-" in term)
    return {
        "main_queries": _dedupe(main_queries)[:8],
        "prerequisite_queries": topic_terms[5:10],
        "exclude_terms": [],
        "expected_communities": topic_terms,
        "positive_terms": topic_terms,
        "alias_queries": topic_terms,
        "exact_title_queries": landmarks,
        "verified_landmark_candidates": benchmark.get("expected_landmarks", []),
    }


def _markdown(topic: str, results: dict[str, dict]) -> str:
    lines = [
        f"# Skill 2 Graph Mode Ablation: {topic}",
        "",
        "| Mode | Edges | Citation Edges | Similarity Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Top Bridge Plausibility | Foundation Landmark Hit | Graph Edge Yield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in results.items():
        lines.append(
            "| {mode} | {edges} | {citation} | {similarity} | {components} | {largest:.1%} | {modularity:.4f} | {communities} | {coverage:.1%} | {bridge:.1%} | {foundation:.1%} | {yield_:.2f} |".format(
                mode=mode,
                edges=metrics["edge_count"],
                citation=metrics["citation_edges"],
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
        "## Interpretation Guide",
        "",
        "- Citation-only is most faithful to scholarly dependency, but can be sparse when references are missing.",
        "- Similarity-only improves connectivity and community coverage, but can over-connect papers with similar vocabulary.",
        "- Hybrid is expected to balance citation evidence with semantic coverage.",
        "- Top bridge plausibility is a deterministic proxy based on betweenness, cross-community neighbor ratio, and degree.",
        "- Foundation landmark hit checks whether top foundation-score papers include known foundational landmarks.",
    ]
    return "\n".join(lines) + "\n"


def _batch_markdown(all_results: dict[str, dict]) -> str:
    averages = _average_by_mode(all_results)
    averages_without_low_corpus = _average_by_mode({
        topic: modes for topic, modes in all_results.items()
        if topic != "protein_structure"
    })
    lines = [
        "# Skill 2 Graph Mode Ablation: Multi-Topic Benchmark",
        "",
        "This ablation rebuilds citation-only, similarity-only, and hybrid graphs from the same saved corpora used in the full-agent benchmark.",
        "",
        "## Average Metrics",
        "",
        "| Mode | Topics | Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Top Bridge Plausibility | Foundation Landmark Hit | Graph Edge Yield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in averages.items():
        lines.append(_average_row(mode, metrics))

    lines += [
        "",
        "## Average Metrics Excluding Protein Structure",
        "",
        "Protein structure had a low-paper corpus in the saved full-agent benchmark, so this secondary average reports the same metrics without that low-confidence topic.",
        "",
        "| Mode | Topics | Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Top Bridge Plausibility | Foundation Landmark Hit | Graph Edge Yield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in averages_without_low_corpus.items():
        lines.append(_average_row(mode, metrics))

    lines += [
        "",
        "## Per-Topic Results",
        "",
        "| Topic | Mode | Nodes | Edges | Citation Edges | Similarity Edges | Components | Largest Component | Modularity | Communities | Path Community Coverage | Foundation Landmark Hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topic_id, modes in all_results.items():
        for mode, metrics in modes.items():
            lines.append(
                "| {topic} | {mode} | {nodes} | {edges} | {citation} | {similarity} | {components} | {largest:.1%} | {modularity:.4f} | {communities} | {coverage:.1%} | {foundation:.1%} |".format(
                    topic=topic_id,
                    mode=mode,
                    nodes=metrics["node_count"],
                    edges=metrics["edge_count"],
                    citation=metrics["citation_edges"],
                    similarity=metrics["similarity_edges"],
                    components=metrics["connected_components"],
                    largest=metrics["largest_component_ratio"],
                    modularity=metrics["modularity"],
                    communities=metrics["community_count"],
                    coverage=metrics["community_coverage_in_reading_path"],
                    foundation=metrics["top_foundation_landmark_hit"],
                )
            )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Citation-only is the cleanest dependency graph, but sparse metadata can fragment the graph.",
        "- Similarity-only usually increases connectivity and graph edge yield, but its edges are semantic association rather than scholarly dependency.",
        "- Hybrid graph is the production choice because it preserves citation edges and uses semantic edges to recover structure when references are incomplete.",
    ]
    return "\n".join(lines) + "\n"


def _average_by_mode(all_results: dict[str, dict]) -> dict[str, dict[str, float]]:
    averages: dict[str, dict[str, float]] = {}
    for mode in GRAPH_MODES:
        rows = [modes[mode] for modes in all_results.values() if mode in modes]
        if not rows:
            continue
        keys = [
            "edge_count",
            "connected_components",
            "largest_component_ratio",
            "modularity",
            "community_count",
            "community_coverage_in_reading_path",
            "top_bridge_plausibility",
            "top_foundation_landmark_hit",
            "graph_edge_yield",
        ]
        averages[mode] = {"topic_count": len(rows)}
        for key in keys:
            averages[mode][key] = sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
    return averages


def _average_row(mode: str, metrics: dict[str, float]) -> str:
    return (
        "| {mode} | {topics:.0f} | {edges:.1f} | {components:.1f} | {largest:.1%} | {modularity:.4f} | {communities:.1f} | {coverage:.1%} | {bridge:.1%} | {foundation:.1%} | {yield_:.2f} |"
    ).format(
        mode=mode,
        topics=metrics["topic_count"],
        edges=metrics["edge_count"],
        components=metrics["connected_components"],
        largest=metrics["largest_component_ratio"],
        modularity=metrics["modularity"],
        communities=metrics["community_count"],
        coverage=metrics["community_coverage_in_reading_path"],
        bridge=metrics["top_bridge_plausibility"],
        foundation=metrics["top_foundation_landmark_hit"],
        yield_=metrics["graph_edge_yield"],
    )


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _user_level(value: str) -> UserLevel:
    return {
        "beginner": UserLevel.BEGINNER,
        "advanced": UserLevel.ADVANCED,
        "intermediate": UserLevel.INTERMEDIATE,
    }.get(str(value).lower(), UserLevel.INTERMEDIATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Skill 2 graph-mode ablation.")
    parser.add_argument("--topic", help="Topic string or benchmark ID.")
    parser.add_argument("--state-root", help="Directory containing per-topic state.json files.")
    parser.add_argument("--ids", nargs="*", help="Optional topic IDs when using --state-root.")
    parser.add_argument("--output-dir", default="outputs/graph_ablation")
    parser.add_argument("--max-papers", type=int, default=45)
    parser.add_argument("--user-level", choices=["beginner", "intermediate", "advanced"], default="intermediate")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.state_root:
        results = run_from_state_root(args.state_root, args.output_dir, args.ids)
        print(_batch_markdown(results))
        print(f"Saved to {args.output_dir}")
        return

    if not args.topic:
        parser.error("--topic is required unless --state-root is provided")

    benchmark = get_benchmark(args.topic)
    topic_name = benchmark["topic"] if benchmark else args.topic
    results = run_graph_ablation(
        topic_or_id=args.topic,
        output_dir=args.output_dir,
        max_papers=args.max_papers,
        user_level=args.user_level,
        demo=args.demo,
    )
    print(_markdown(topic_name, results))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
