import json
import os
from typing import Any

from evaluation.metrics import markdown_table


def write_evaluation_reports(topic: str, results: dict[str, dict], output_dir: str) -> None:
    state = _load_state(os.path.join(output_dir, "llm_assisted_agent", "state.json"))
    _write(os.path.join(output_dir, "group_eval_summary.md"), _group_summary(topic, results, state))
    _write(os.path.join(output_dir, "retrieval_eval.md"), _retrieval_report(topic, results, state))
    _write(os.path.join(output_dir, "graph_eval.md"), _graph_report(topic, results, state))
    _write(os.path.join(output_dir, "reading_path_eval.md"), _reading_path_report(topic, results, state))
    _write(os.path.join(output_dir, "manual_rating_template.md"), _manual_rating_template(topic, state))


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _load_state(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _group_summary(topic: str, results: dict[str, dict], state: dict[str, Any]) -> str:
    metadata = state.get("metadata", {})
    lines = [
        f"# Group Agent Evaluation: {topic}",
        "",
        "## Main Result Table",
        "",
        markdown_table(results),
        "",
        "## Reading-Path Quality Metrics",
        "",
        _quality_table(results),
        "",
        "## Agent Behavior",
        "",
        f"- Planner mode: `{metadata.get('planner_mode', 'unknown')}`",
        f"- LLM query plan available: `{bool(metadata.get('query_plan'))}`",
        f"- LLM community labels available: `{bool(metadata.get('community_labels'))}`",
        f"- Evidence packets available: `{bool(metadata.get('evidence_packets'))}`",
        "",
        "## Interpretation",
        "",
        "- The Agent composes retrieval, graph analysis, and reading-path generation into one reproducible workflow.",
        "- The deterministic graph metrics provide the structural basis for selecting foundation, bridge, and frontier papers.",
        "- LLM calls are used for task parsing, query planning, community naming, and evidence-grounded why-read text; centrality and community computation remain deterministic.",
        "- Demo-mode outputs are only for pipeline reliability checks. Report-quality claims should use live-mode arXiv/OpenAlex/Semantic Scholar results.",
        "- Curated or landmark records are transparent source categories. They improve recall for known foundational papers but are reported separately from retrieved API records.",
    ]
    return "\n".join(lines)


def _retrieval_report(topic: str, results: dict[str, dict], state: dict[str, Any]) -> str:
    lines = [
        f"# Literature Retrieval Skill Report: {topic}",
        "",
        "## Functionality",
        "",
        "This Skill turns a natural-language research topic into a structured academic corpus. It combines LLM or rule-based query planning with deterministic arXiv/OpenAlex retrieval, title deduplication, topic-specific filtering, and metadata enrichment.",
        "",
        "## Inputs and Outputs",
        "",
        "- Inputs: topic, user level, time range, maximum papers, optional query plan.",
        "- Outputs: paper corpus, source URLs, abstracts, citation counts with source labels, reference lists, corpus quality metrics, and query-plan metadata.",
        "",
        "## Quantitative Results",
        "",
        "| Run | Planner | Papers | Abstract Coverage | Citation/Reference Coverage | Year Range | Dedup Removed | Main Queries |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for name in ["rule_only_agent", "llm_assisted_agent"]:
        metrics = results.get(name, {})
        retrieval = metrics.get("retrieval", {})
        query_plan = metrics.get("query_plan", {})
        lines.append(
            "| {name} | {planner} | {papers} | {abstract:.1%} | {citation:.1%} | {year_range} | {dedup} | {queries} |".format(
                name=name,
                planner=metrics.get("planner_mode") or "none",
                papers=retrieval.get("total_papers", 0),
                abstract=retrieval.get("has_abstract_ratio", 0),
                citation=retrieval.get("has_citation_ratio", 0),
                year_range=retrieval.get("year_range", "N/A"),
                dedup=retrieval.get("deduplication_removed", 0),
                queries=len(query_plan.get("main_queries", [])),
            )
        )

    metadata = state.get("metadata", {})
    query_plan = metadata.get("query_plan", {})
    lines += [
        "",
        "## Source Mix",
        "",
        "| Run | Source Mix |",
        "|---|---|",
    ]
    for name in ["rule_only_agent", "llm_assisted_agent"]:
        source_mix = results.get(name, {}).get("quality", {}).get("source_mix", {})
        lines.append(f"| {name} | {json.dumps(source_mix, ensure_ascii=False)} |")

    lines += [
        "",
        "## LLM Query Plan",
        "",
        f"- Main queries: {', '.join(query_plan.get('main_queries', [])) or 'N/A'}",
        f"- Prerequisite queries: {', '.join(query_plan.get('prerequisite_queries', [])) or 'N/A'}",
        f"- Exclude terms: {', '.join(query_plan.get('exclude_terms', [])) or 'N/A'}",
        f"- Expected communities: {', '.join(query_plan.get('expected_communities', [])) or 'N/A'}",
        "",
        "## Analysis",
        "",
        "- The key measurable outputs are corpus size, abstract coverage, citation/reference coverage, and year range.",
        "- Topic-specific filtering is important for ambiguous terms and short acronyms. CFR retrieval filters out unrelated uses of `regret` and `counterfactual`; Vision Transformer retrieval uses word-boundary matching for acronyms such as `ViT`, `DeiT`, `DINO`, `MAE`, and `BEiT` to avoid unrelated high-citation OpenAlex records.",
        "- Citation counts prefer Semantic Scholar `citationCount` when `S2_API_KEY` is available, then fall back to OpenAlex `cited_by_count` or curated landmark metadata. They may still differ from Google Scholar snapshots.",
        "- Source mix is reported to distinguish API-retrieved papers from curated landmarks and synthetic demo records.",
    ]
    return "\n".join(lines)


def _graph_report(topic: str, results: dict[str, dict], state: dict[str, Any]) -> str:
    lines = [
        f"# Research Graph Analysis Skill Report: {topic}",
        "",
        "## Functionality",
        "",
        "This Skill builds paper networks and computes deterministic SNA metrics. Citation edges capture explicit references when available; similarity edges use TF-IDF cosine similarity over abstracts to reduce sparsity.",
        "",
        "## Graph Ablation Results",
        "",
        "| Graph Mode | Nodes | Edges | Citation Edges | Similarity Edges | Components | Largest Component | Communities | Modularity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["citation_graph", "similarity_graph", "hybrid_graph"]:
        graph = results.get(name, {}).get("graph", {})
        lines.append(
            "| {mode} | {nodes} | {edges} | {cite} | {sim} | {comp} | {largest:.1%} | {communities} | {modularity:.4f} |".format(
                mode=name.replace("_graph", ""),
                nodes=graph.get("node_count", 0),
                edges=graph.get("edge_count", 0),
                cite=graph.get("citation_edges", 0),
                sim=graph.get("similarity_edges", 0),
                comp=graph.get("connected_components", 0),
                largest=graph.get("largest_component_ratio", 0),
                communities=graph.get("community_count", 0),
                modularity=graph.get("modularity", 0),
            )
        )

    labels = state.get("metadata", {}).get("community_labels", {})
    lines += [
        "",
        "## LLM-Labeled Communities",
        "",
    ]
    for cid, label in sorted(labels.items(), key=lambda x: int(x[0])):
        lines.append(f"- **Community {cid}: {label.get('label', 'Unlabeled')}** — {label.get('description', '')}")
    if not labels:
        lines.append("- No community labels recorded.")

    lines += [
        "",
        "## Analysis",
        "",
        "- Citation-only graphs are precise but sparse when references are missing from APIs.",
        "- Similarity-only graphs improve connectivity but can over-cluster papers by language rather than citation structure.",
        "- Hybrid graphs are the default because they preserve citation evidence while adding enough semantic edges for stable community detection.",
        "- Citation PageRank is computed on a directed citation graph where citing papers point to cited papers. Community detection uses the undirected semantic/hybrid projection.",
        "- Betweenness centrality uses `distance = 1 / weight` so stronger similarity means shorter graph distance.",
        "- PageRank, betweenness, Louvain communities, and role scores are deterministic and are not computed by the LLM.",
    ]
    return "\n".join(lines)


def _reading_path_report(topic: str, results: dict[str, dict], state: dict[str, Any]) -> str:
    path = state.get("reading_path", {}).get("reading_path", [])
    papers = state.get("papers", {})
    sorted_by_citation = _unique_by_title(
        sorted(papers.values(), key=lambda p: p.get("citation_count", 0), reverse=True)
    )[:12]

    lines = [
        f"# Reading Path and Report Skill Report: {topic}",
        "",
        "## Functionality",
        "",
        "This Skill converts graph scores and community labels into a staged reading path. It generates evidence packets and asks the LLM for concise why-read explanations grounded in title, abstract, URL, stage, community, and graph scores.",
        "",
        "## Reading Path Metrics",
        "",
        "| Run | Stages | Unique Papers | Explanation Coverage | Role Counts |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ["rule_only_agent", "llm_assisted_agent"]:
        metrics = results.get(name, {}).get("reading_path", {})
        lines.append(
            "| {name} | {stages} | {papers} | {coverage:.1%} | {roles} |".format(
                name=name,
                stages=metrics.get("stage_count", 0),
                papers=metrics.get("unique_paper_count", 0),
                coverage=metrics.get("explanation_availability_ratio", 0),
                roles=json.dumps(metrics.get("role_counts", {}), ensure_ascii=False),
            )
        )

    lines += [
        "",
        "## Path Quality Metrics",
        "",
        "| Run | Landmark Hit Rate | Topic Precision | Ordering Quality | Community Coverage | Stage Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["rule_only_agent", "llm_assisted_agent"]:
        quality = results.get(name, {}).get("quality", {})
        lines.append(
            "| {name} | {landmark:.1%} | {precision:.1%} | {ordering:.1%} | {community:.1%} | {stage:.1%} |".format(
                name=name,
                landmark=quality.get("landmark_hit_rate", 0),
                precision=quality.get("topic_precision", 0),
                ordering=quality.get("ordering_quality", 0),
                community=quality.get("community_coverage", 0),
                stage=quality.get("path_stage_coverage", 0),
            )
        )

    lines += [
        "",
        "## Network-Aware Reading Path",
        "",
    ]
    for stage in path:
        lines.append(f"### {stage.get('stage', 'Stage')}")
        for paper in stage.get("papers", []):
            lines.append(f"- {paper.get('read_order')}. {paper.get('title')} — {paper.get('reason')}")
        lines.append("")

    lines += [
        "## Citation-Count Baseline",
        "",
        "This baseline ranks papers only by citation count and ignores learning stage, community coverage, and bridge/frontier roles.",
        "",
    ]
    for idx, paper in enumerate(sorted_by_citation, 1):
        source = _citation_source_label(paper.get("citation_source", ""))
        lines.append(f"{idx}. {paper.get('title')} ({paper.get('year')}, {source}: {paper.get('citation_count', 0)})")

    lines += [
        "",
        "## Analysis",
        "",
        "- The network-aware path separates prerequisites/foundations, core methods, key developments, bridge papers, and frontier papers.",
        "- The citation-count baseline often over-emphasizes old or broadly cited papers and does not guarantee a coherent learning order.",
        "- Evidence-grounded why-read text is generated from structured packets, limiting LLM freedom to invent unsupported claims.",
        "- Landmark hit rate, ordering quality, and community coverage directly evaluate whether the Agent creates a useful reading path rather than only a connected graph.",
    ]
    return "\n".join(lines)


def _quality_table(results: dict[str, dict]) -> str:
    lines = [
        "| Run | Landmark Hit Rate | Topic Precision | Ordering Quality | Community Coverage | Stage Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        quality = metrics.get("quality", {})
        lines.append(
            "| {name} | {landmark:.1%} | {precision:.1%} | {ordering:.1%} | {community:.1%} | {stage:.1%} |".format(
                name=name,
                landmark=quality.get("landmark_hit_rate", 0),
                precision=quality.get("topic_precision", 0),
                ordering=quality.get("ordering_quality", 0),
                community=quality.get("community_coverage", 0),
                stage=quality.get("path_stage_coverage", 0),
            )
        )
    return "\n".join(lines)


def _unique_by_title(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for paper in papers:
        title = " ".join((paper.get("title") or "").lower().split())
        if not title or title in seen:
            continue
        seen.add(title)
        unique.append(paper)
    return unique


def _citation_source_label(source: str) -> str:
    if source == "semantic_scholar":
        return "Semantic Scholar citations"
    if source == "openalex":
        return "OpenAlex citations"
    if source == "curated":
        return "curated citations"
    return "citations"


def _manual_rating_template(topic: str, state: dict[str, Any]) -> str:
    return "\n".join([
        f"# Manual Reading Path Rating Template: {topic}",
        "",
        "Use this table for human evaluation in the group report. Score each criterion from 1 to 5.",
        "",
        "| Method | Coherence | Coverage | Beginner/Intermediate Friendliness | Explanation Usefulness | Notes |",
        "|---|---:|---:|---:|---:|---|",
        "| Citation-count baseline |  |  |  |  |  |",
        "| Network-aware ResearchTrail |  |  |  |  |  |",
        "| LLM-assisted ResearchTrail |  |  |  |  |  |",
    ])
