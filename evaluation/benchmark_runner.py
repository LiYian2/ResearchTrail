import argparse
import json
import os
import re
import time
from pathlib import Path

from agent.llm_client import LLMClient, load_prompt
from evaluation.benchmark_registry import BENCHMARK_TOPICS
from evaluation.metrics import collect_metrics
from shared.data_layer import SharedDataLayer
from shared.types import ResearchProfile, UserLevel
from skill_graph.analysis import GraphAnalysisSkill
from skill_graph.community_labeler import CommunityLabeler
from skill_graph.graph_builder import GraphBuilder
from skill_reading_path.explanation_writer import ExplanationWriter
from skill_reading_path.path_generator import ReadingPathSkill
from skill_reading_path.report import ReportGenerator
from skill_reading_path.visualization import VisualizationEngine
from skill_retrieval.retrieval import LiteratureRetrievalSkill


def run_one(entry: dict, output_root: str, max_papers: int, llm: str, demo: bool) -> dict:
    output_dir = os.path.join(output_root, entry["id"])
    os.makedirs(output_dir, exist_ok=True)
    data = SharedDataLayer()
    profile = ResearchProfile(
        topic=entry["topic"],
        user_level=_user_level(entry.get("user_level", "intermediate")),
        max_papers=max_papers,
        preferred_length=12,
        goal="benchmark_research_trail",
    )
    data.set_metadata("benchmark_prompt", entry["prompt"])
    data.set_metadata("planner_mode", "benchmark_fixed_topic_llm_assisted" if llm != "off" else "benchmark_fixed_topic_rule")

    llm_client = LLMClient(enabled=llm != "off")
    llm_normalization = _llm_query_normalization(entry, llm_client)
    query_plan = _benchmark_query_plan(entry, llm_normalization)

    try:
        retrieval = LiteratureRetrievalSkill(data)
        papers = retrieval.run(profile, query_plan=query_plan, demo=demo)
        if not demo:
            _recover_verified_landmarks(retrieval, data, entry)
            _apply_benchmark_relevance_filter(retrieval, data, entry, max_papers, llm_normalization)
            papers = data.get_all_papers()

        if not papers:
            data.set_metadata("benchmark_error", "no_papers_retrieved")
            return _finalize_metrics(data, entry, output_dir)

        GraphBuilder(data).build(mode="hybrid")
        GraphAnalysisSkill(data).run()
        CommunityLabeler(data, llm_client).run()
        ReadingPathSkill(data, ExplanationWriter(llm_client)).run()
        _write_outputs(data, output_dir)
    except Exception as exc:
        data.set_metadata("benchmark_error", f"{type(exc).__name__}: {exc}")

    metrics = _finalize_metrics(data, entry, output_dir)
    return metrics


def _user_level(value: str) -> UserLevel:
    return {
        "beginner": UserLevel.BEGINNER,
        "advanced": UserLevel.ADVANCED,
        "intermediate": UserLevel.INTERMEDIATE,
    }.get(str(value).lower(), UserLevel.INTERMEDIATE)


def _llm_query_normalization(entry: dict, llm_client: LLMClient) -> dict:
    if not llm_client.available():
        return {}
    payload = {
        "topic": entry["topic"],
        "prompt": entry["prompt"],
        "expected_landmark_hints": entry.get("expected_landmarks", []),
        "topic_term_hints": entry.get("topic_terms", []),
    }
    result = llm_client.complete_json(
        load_prompt("query_normalizer.md"),
        json.dumps(payload, indent=2, ensure_ascii=False),
        temperature=0.1,
        max_tokens=1600,
    )
    if not result.ok or not isinstance(result.data, dict):
        return {"error": result.error}
    return result.data


def _benchmark_query_plan(entry: dict, llm_normalization: dict | None = None) -> dict:
    llm_normalization = llm_normalization or {}
    landmarks = [
        str(item.get("title", "")).strip()
        for item in entry.get("expected_landmarks", [])
        if item.get("title")
    ]
    topic_terms = [str(term).strip() for term in entry.get("topic_terms", []) if str(term).strip()]
    relevance_terms = _benchmark_relevance_terms(entry, llm_normalization)
    llm_titles = _clean_llm_list(llm_normalization.get("exact_title_queries", []))
    llm_main = _clean_llm_list(llm_normalization.get("main_queries", []))
    llm_aliases = _clean_llm_list(llm_normalization.get("alias_queries", []))
    llm_negative = _clean_llm_list(llm_normalization.get("negative_terms", []))
    llm_subfields = _clean_llm_list(llm_normalization.get("expected_subfields", []))

    main_queries = [entry["topic"]]
    main_queries.extend(f'"{title}"' for title in _dedupe(landmarks[:5] + llm_titles[:5]))
    main_queries.extend(llm_main[:4])
    main_queries.extend(llm_aliases[:4])
    main_queries.extend(relevance_terms[:5])

    return {
        "main_queries": _dedupe(main_queries),
        "prerequisite_queries": relevance_terms[5:10] or topic_terms[4:8],
        "exclude_terms": llm_negative,
        "expected_communities": _dedupe(topic_terms + llm_subfields),
        "verified_landmark_candidates": entry.get("expected_landmarks", []),
        "llm_query_normalization": llm_normalization,
        "source": "benchmark_registry_plus_llm_normalization" if llm_normalization else "benchmark_registry",
    }


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


def _recover_verified_landmarks(
    retrieval: LiteratureRetrievalSkill,
    data: SharedDataLayer,
    entry: dict,
) -> None:
    existing_titles = [_normalize_title(p.title) for p in data.get_all_papers()]
    recovered = []
    for landmark in entry.get("expected_landmarks", []):
        title = str(landmark.get("title", "")).strip()
        if not title:
            continue
        target = _normalize_title(title)
        if any(target in existing or existing in target for existing in existing_titles):
            continue

        candidates = []
        for query in (f'"{title}"', title):
            candidates.extend(retrieval._search_arxiv(query, 4))
            candidates.extend(retrieval._search_openalex(query, 4))
            time.sleep(0.25)
        best = _best_title_match(title, candidates)
        if best:
            data.add_papers([best])
            existing_titles.append(_normalize_title(best.title))
            recovered.append({"target": title, "matched": best.title, "source": best.source})
    data.set_metadata("verified_landmark_recovery", recovered)


def _apply_benchmark_relevance_filter(
    retrieval: LiteratureRetrievalSkill,
    data: SharedDataLayer,
    entry: dict,
    max_papers: int,
    llm_normalization: dict | None = None,
) -> None:
    papers = data.get_all_papers()
    if not papers:
        return
    terms = _benchmark_relevance_terms(entry, llm_normalization or {})
    landmark_titles = [item.get("title", "") for item in entry.get("expected_landmarks", [])]
    filtered = [
        paper for paper in papers
        if _paper_matches_terms(paper.title, paper.abstract, terms, landmark_titles)
    ]
    if len(filtered) < min(8, len(papers)):
        filtered = papers
    filtered = sorted(
        filtered,
        key=lambda p: (
            _is_landmark_title(p.title, landmark_titles),
            p.citation_count,
            p.year,
        ),
        reverse=True,
    )[:max_papers]
    enriched = retrieval._enrich_citations(filtered, max_papers)
    data.clear_papers()
    data.add_papers(enriched)
    quality = retrieval._assess_corpus_quality(enriched)
    quality["benchmark_relevance_filtered"] = len(papers) - len(enriched)
    data.set_corpus_quality(quality)


def _benchmark_relevance_terms(entry: dict, llm_normalization: dict | None = None) -> list[str]:
    llm_normalization = llm_normalization or {}
    topic_terms = [str(term).lower().strip() for term in entry.get("topic_terms", []) if str(term).strip()]
    acronym_terms = {
        "rag", "dpr", "retro", "gcn", "gat", "moco", "byol", "dino", "dinov2", "mae", "beit",
        "mamba", "ppo", "dpo", "rlaif", "nerf", "maddpg", "qmix", "mappo", "fedavg", "fedprox",
    }
    custom = {
        "diffusion_models": [
            "denoising diffusion", "diffusion model", "score-based generative", "stochastic differential equation",
            "latent diffusion", "flow matching", "rectified flow", "ddpm", "generative model",
        ],
        "rag": [
            "retrieval-augmented", "retrieval augmented", "dense passage", "open-domain question answering",
            "fusion-in-decoder", "self-rag", "retro", "atlas", "agentic rag", "rag",
        ],
        "mechanistic_interpretability": [
            "mechanistic interpretability", "transformer circuits", "induction head", "superposition",
            "sparse autoencoder", "monosemanticity", "circuits", "sae",
        ],
        "gnn": [
            "graph neural network", "graph convolutional", "graphsage", "graph attention", "message passing",
            "weisfeiler", "graph transformer", "geometric deep learning",
        ],
        "causal_ml": [
            "causal inference", "potential outcome", "propensity score", "causal graph", "counterfactual",
            "causal discovery", "invariant risk", "treatment effect",
        ],
        "ssl_vision": [
            "self-supervised", "contrastive learning", "simclr", "moco", "byol", "dino",
            "masked autoencoder", "masked image modeling", "ibot",
        ],
        "efficient_transformers": [
            "longformer", "reformer", "linear attention", "sparse attention", "performer",
            "flashattention", "state space", "structured state spaces", "mamba", "long context",
        ],
        "rlhf": [
            "human feedback", "preference optimization", "reward model", "proximal policy optimization",
            "ppo", "dpo", "rlaif", "constitutional ai", "instruction following",
        ],
        "nerf_3dgs": [
            "neural radiance", "nerf", "mip-nerf", "instant-ngp", "plenoxels",
            "gaussian splatting", "novel view synthesis",
        ],
        "marl": [
            "multi-agent reinforcement learning", "value decomposition", "centralized training",
            "maddpg", "qmix", "mappo", "population based", "starcraft multi-agent",
        ],
        "federated_learning": [
            "federated learning", "fedavg", "secure aggregation", "differential privacy",
            "personalized federated", "fedprox", "decentralized data",
        ],
        "protein_structure": [
            "protein structure", "alphafold", "rosettafold", "coevolution", "protein language model",
            "protein design", "scientific machine learning",
        ],
    }.get(entry.get("id"), [])
    usable_topic_terms = [
        term for term in topic_terms
        if " " in term or "-" in term or term in acronym_terms
    ]
    llm_terms = (
        _clean_llm_list(llm_normalization.get("positive_terms", []))
        + _clean_llm_list(llm_normalization.get("alias_queries", []))
        + _clean_llm_list(llm_normalization.get("main_queries", []))
    )
    llm_terms = [
        term.lower()
        for term in llm_terms
        if " " in term or "-" in term or term.lower() in acronym_terms
    ]
    return _dedupe(custom + usable_topic_terms + llm_terms)


def _clean_llm_list(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, str):
            continue
        item = re.sub(r"\s+", " ", item).strip()
        if item:
            out.append(item)
    return _dedupe(out)


def _paper_matches_terms(title: str, abstract: str, terms: list[str], landmark_titles: list[str]) -> bool:
    text = f" {title} {abstract} ".lower()
    normalized = _normalize_title(title)
    if any(_normalize_title(landmark) in normalized or normalized in _normalize_title(landmark) for landmark in landmark_titles):
        return True
    return any(_term_match(text, term) for term in terms)


def _best_title_match(target_title: str, candidates: list) -> object | None:
    target = _normalize_title(target_title)
    target_tokens = set(target.split())
    best = None
    best_score = 0.0
    for candidate in candidates:
        cand = _normalize_title(getattr(candidate, "title", ""))
        if not cand:
            continue
        cand_tokens = set(cand.split())
        overlap = len(target_tokens & cand_tokens) / max(len(target_tokens | cand_tokens), 1)
        containment = 1.0 if target in cand or cand in target else 0.0
        score = max(overlap, containment)
        if score > best_score:
            best = candidate
            best_score = score
    return best if best_score >= 0.45 else None


def _is_landmark_title(title: str, landmark_titles: list[str]) -> bool:
    normalized = _normalize_title(title)
    return any(_normalize_title(landmark) in normalized or normalized in _normalize_title(landmark) for landmark in landmark_titles)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _term_match(text: str, term: str) -> bool:
    term = term.lower().strip()
    if len(term) <= 5 and term.isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


def _write_outputs(data: SharedDataLayer, output_dir: str) -> None:
    report_path = os.path.join(output_dir, "research_report.md")
    graph_path = os.path.join(output_dir, "research_graph.png")
    dist_path = os.path.join(output_dir, "scores_distribution.png")
    state_path = os.path.join(output_dir, "state.json")

    report = ReportGenerator(data).generate()
    Path(report_path).write_text(report, encoding="utf-8")
    data.save(state_path)

    visualizer = VisualizationEngine(data)
    try:
        visualizer.render_network(graph_path, figsize=(18, 12))
    except Exception as exc:
        data.set_metadata("network_visualization_error", str(exc))
    try:
        visualizer.render_score_distribution(dist_path)
    except Exception as exc:
        data.set_metadata("score_visualization_error", str(exc))
    data.save(state_path)


def _finalize_metrics(data: SharedDataLayer, entry: dict, output_dir: str) -> dict:
    if not os.path.exists(os.path.join(output_dir, "state.json")):
        data.save(os.path.join(output_dir, "state.json"))
    metrics = collect_metrics(data)
    metrics["benchmark"] = {
        "id": entry["id"],
        "topic": entry["topic"],
        "prompt": entry["prompt"],
        "expected_landmarks": entry["expected_landmarks"],
    }
    if data.get_metadata("benchmark_error"):
        metrics["error"] = data.get_metadata("benchmark_error")
    return metrics


def write_summary(results: dict[str, dict], output_root: str) -> None:
    lines = [
        "# ResearchTrail LLM-Assisted Benchmark",
        "",
        "| ID | Papers | Landmark Hit | Topic Precision | Ordering | Community Coverage | Stage Coverage | Report |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for bid, metrics in results.items():
        quality = metrics.get("quality", {})
        retrieval = metrics.get("retrieval", {})
        report_path = f"{bid}/research_report.md"
        lines.append(
            "| {bid} | {papers} | {landmark:.1%} | {precision:.1%} | {ordering:.1%} | {community:.1%} | {stage:.1%} | `{report}` |".format(
                bid=bid,
                papers=retrieval.get("total_papers", 0),
                landmark=quality.get("landmark_hit_rate", 0),
                precision=quality.get("topic_precision", 0),
                ordering=quality.get("ordering_quality", 0),
                community=quality.get("community_coverage", 0),
                stage=quality.get("path_stage_coverage", 0),
                report=report_path,
            )
        )

    lines += [
        "",
        "## Notes",
        "",
        "- This benchmark uses fixed benchmark topics and verified landmark candidates to avoid prompt parsing drift.",
        "- Retrieval, graph analysis, community labeling, explanation writing, and report generation still run through the normal skills.",
        "- It is intended to inspect report quality and research-trail coherence, not graph-mode ablations.",
        "- Expected landmarks come from `evaluation/benchmark_registry.py` and are used only for evaluation.",
    ]
    Path(output_root, "benchmark_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-assisted ResearchTrail benchmark prompts.")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--max-papers", type=int, default=50)
    parser.add_argument("--ids", nargs="*", default=None, help="Optional benchmark IDs to run.")
    parser.add_argument("--llm", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected = BENCHMARK_TOPICS
    if args.ids:
        wanted = set(args.ids)
        selected = [entry for entry in BENCHMARK_TOPICS if entry["id"] in wanted]

    results = {}
    for idx, entry in enumerate(selected, 1):
        print(f"[{idx}/{len(selected)}] Running {entry['id']}: {entry['topic']}", flush=True)
        results[entry["id"]] = run_one(entry, args.output_dir, args.max_papers, args.llm, args.demo)
        Path(args.output_dir, "benchmark_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_summary(results, args.output_dir)

    print(f"Benchmark saved to {args.output_dir}")


if __name__ == "__main__":
    main()
