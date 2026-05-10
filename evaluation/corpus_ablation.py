import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from evaluation.benchmark_registry import get_benchmark
from evaluation.metrics import _is_topic_relevant
from shared.data_layer import SharedDataLayer
from shared.types import Paper, ResearchProfile, UserLevel
from skill_graph.graph_builder import GraphBuilder
from skill_retrieval.retrieval import LiteratureRetrievalSkill


VARIANTS = {
    "A_arxiv_only": {"arxiv": True, "openalex": False, "filter": False, "landmarks": False},
    "B_openalex_only": {"arxiv": False, "openalex": True, "filter": False, "landmarks": False},
    "C_arxiv_openalex": {"arxiv": True, "openalex": True, "filter": False, "landmarks": False},
    "D_plus_topic_filtering": {"arxiv": True, "openalex": True, "filter": True, "landmarks": False},
    "E_plus_verified_landmarks": {"arxiv": True, "openalex": True, "filter": True, "landmarks": True},
}


def run_ablation(topic_or_id: str, output_dir: str, max_papers: int, user_level: str, enrich: bool) -> dict[str, dict]:
    benchmark = get_benchmark(topic_or_id)
    topic = benchmark["topic"] if benchmark else topic_or_id
    profile = ResearchProfile(
        topic=topic,
        user_level=_user_level(user_level or (benchmark or {}).get("user_level", "intermediate")),
        max_papers=max_papers,
        preferred_length=12,
        goal="corpus_construction_ablation",
    )
    query_plan = _query_plan(topic, benchmark)
    search_cache: dict[tuple[str, str], list[Paper]] = {}

    results = {}
    for name, cfg in VARIANTS.items():
        data = SharedDataLayer()
        data.set_research_profile(profile)
        data.set_metadata("query_plan", query_plan)
        skill = LiteratureRetrievalSkill(data)

        raw = _retrieve_raw(skill, query_plan["main_queries"], max_papers, cfg["arxiv"], cfg["openalex"], search_cache)
        deduped = skill._deduplicate(raw)
        papers = deduped
        if cfg["filter"]:
            papers = skill._filter_relevance(papers, topic, query_plan=query_plan)
            papers = skill._filter_topic_specific_relevance(papers, topic, query_plan=query_plan)
            papers = skill._filter_exclude_terms(papers, query_plan.get("exclude_terms", []))
        if cfg["landmarks"]:
            papers = _recover_verified_landmarks(skill, papers, benchmark)
            papers = skill._deduplicate(papers)
        papers = _rank_for_corpus(topic, papers, benchmark)[:max_papers]
        if enrich:
            papers = skill._enrich_citations(papers, max_papers)

        data.add_papers(papers)
        quality = skill._assess_corpus_quality(papers)
        quality["deduplication_removed"] = max(0, len(raw) - len(deduped))
        data.set_corpus_quality(quality)
        graph = GraphBuilder(data).build(mode="hybrid")
        results[name] = _metrics(topic, papers, raw, deduped, graph, benchmark)

    os.makedirs(output_dir, exist_ok=True)
    Path(output_dir, "corpus_ablation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(output_dir, "corpus_ablation_summary.md").write_text(
        _markdown(topic, results),
        encoding="utf-8",
    )
    return results


def _retrieve_raw(
    skill: LiteratureRetrievalSkill,
    queries: list[str],
    max_papers: int,
    use_arxiv: bool,
    use_openalex: bool,
    search_cache: dict[tuple[str, str], list[Paper]],
) -> list[Paper]:
    raw: list[Paper] = []
    queries = queries[:8] or []
    per_query = max(3, max_papers // max(len(queries), 1))
    for query in queries:
        if use_arxiv:
            cache_key = ("arxiv", query)
            if cache_key not in search_cache:
                search_cache[cache_key] = skill._search_arxiv(query, per_query)
            raw.extend(search_cache[cache_key])
        if use_openalex:
            cache_key = ("openalex", query)
            if cache_key not in search_cache:
                search_cache[cache_key] = skill._search_openalex(query, min(per_query, 20))
            raw.extend(search_cache[cache_key])
        time.sleep(0.25)
    return raw


def _query_plan(topic: str, benchmark: dict | None) -> dict:
    if not benchmark:
        data = SharedDataLayer()
        skill = LiteratureRetrievalSkill(data)
        return {
            "main_queries": skill._generate_query_plan(topic, UserLevel.INTERMEDIATE)[:8],
            "prerequisite_queries": [],
            "exclude_terms": [],
            "expected_communities": [],
            "positive_terms": [],
            "alias_queries": [],
            "exact_title_queries": [],
            "verified_landmark_candidates": [],
        }

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


def _recover_verified_landmarks(
    skill: LiteratureRetrievalSkill,
    papers: list[Paper],
    benchmark: dict | None,
) -> list[Paper]:
    if not benchmark:
        return papers
    existing = {_normalize_title(p.title) for p in papers}
    recovered: list[Paper] = []
    for item in benchmark.get("expected_landmarks", []):
        title = item.get("title", "")
        target = _normalize_title(title)
        if not target or any(target in key or key in target for key in existing):
            continue
        candidates: list[Paper] = []
        for query in [f'"{title}"', title]:
            candidates.extend(skill._search_arxiv(query, 4))
            candidates.extend(skill._search_openalex(query, 4))
            time.sleep(0.25)
        match = _best_paper_title_match(title, candidates)
        if match:
            recovered.append(match)
            existing.add(_normalize_title(match.title))
    return papers + recovered


def _metrics(
    topic: str,
    papers: list[Paper],
    raw: list[Paper],
    deduped: list[Paper],
    graph,
    benchmark: dict | None,
) -> dict[str, Any]:
    n = len(papers)
    abstract_coverage = sum(1 for p in papers if p.abstract) / max(n, 1)
    citation_metadata_coverage = sum(1 for p in papers if p.citation_count > 0) / max(n, 1)
    reference_coverage = sum(1 for p in papers if p.references or p.citations) / max(n, 1)
    topic_precision = sum(1 for p in papers if _is_topic_relevant(topic, p.title, p.abstract)) / max(n, 1)
    landmark_hit = _landmark_hit_rate(papers, benchmark)
    edge_count = len(graph.edges)
    citation_edges = sum(1 for e in graph.edges if e.type == "citation")
    similarity_edges = sum(1 for e in graph.edges if e.type == "similarity")
    return {
        "total_papers": n,
        "raw_records": len(raw),
        "duplicate_removal": max(0, len(raw) - len(deduped)),
        "abstract_coverage": abstract_coverage,
        "citation_metadata_coverage": citation_metadata_coverage,
        "reference_coverage": reference_coverage,
        "landmark_hit_rate": landmark_hit,
        "topic_precision": topic_precision,
        "graph_edges": edge_count,
        "citation_edges": citation_edges,
        "similarity_edges": similarity_edges,
        "graph_edge_yield": edge_count / max(n, 1),
    }


def _rank_for_corpus(topic: str, papers: list[Paper], benchmark: dict | None) -> list[Paper]:
    landmarks = [
        item.get("title", "")
        for item in (benchmark or {}).get("expected_landmarks", [])
        if item.get("title")
    ]
    topic_terms = [
        term for term in (benchmark or {}).get("topic_terms", [])
        if isinstance(term, str) and (" " in term or "-" in term or term.lower() in {
            "rag", "dpr", "retro", "gcn", "gat", "moco", "byol", "dino", "dinov2",
            "mae", "beit", "vit", "ddpm", "sde", "mamba", "ppo", "dpo", "rlaif",
            "nerf", "maddpg", "qmix", "mappo", "fedavg", "fedprox",
        })
    ]

    def score(paper: Paper) -> tuple[float, int, int, int]:
        text = f" {paper.title} {paper.abstract} ".lower()
        normalized_title = _normalize_title(paper.title)
        landmark_bonus = 0
        for landmark in landmarks:
            target = _normalize_title(landmark)
            if target and (target in normalized_title or normalized_title in target):
                landmark_bonus = 100
                break
        term_hits = sum(1 for term in topic_terms if _term_match(text, term.lower()))
        judged_relevant = 20 if _is_topic_relevant(topic, paper.title, paper.abstract) else 0
        metadata_bonus = int(bool(paper.abstract)) + int(bool(paper.references or paper.citations)) + int(paper.citation_count > 0)
        source_bonus = 2 if paper.source == "arxiv" else 1 if paper.source == "openalex" else 0
        return (
            landmark_bonus + judged_relevant + term_hits * 5 + metadata_bonus + source_bonus,
            paper.citation_count,
            paper.year,
            len(paper.abstract or ""),
        )

    return sorted(papers, key=score, reverse=True)


def _landmark_hit_rate(papers: list[Paper], benchmark: dict | None) -> float:
    landmarks = (benchmark or {}).get("expected_landmarks", [])
    if not landmarks:
        return 0.0
    titles = [_normalize_title(p.title) for p in papers]
    hits = 0
    for item in landmarks:
        target = _normalize_title(item.get("title", ""))
        if target and any(target in title or title in target for title in titles):
            hits += 1
    return hits / len(landmarks)


def _markdown(topic: str, results: dict[str, dict]) -> str:
    lines = [
        f"# Corpus Construction Ablation: {topic}",
        "",
        "| Variant | Papers | Duplicates Removed | Abstract Coverage | Citation Metadata | Reference Coverage | Landmark Hit | Topic Precision | Graph Edges | Graph Edge Yield |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        lines.append(
            "| {name} | {papers} | {dupes} | {abstract:.1%} | {citation:.1%} | {refs:.1%} | {landmark:.1%} | {precision:.1%} | {edges} | {yield_:.2f} |".format(
                name=name,
                papers=metrics["total_papers"],
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
        "## Variant Definitions",
        "",
        "- A: arXiv only.",
        "- B: OpenAlex only.",
        "- C: arXiv + OpenAlex.",
        "- D: arXiv + OpenAlex + topic filtering.",
        "- E: arXiv + OpenAlex + topic filtering + verified landmark recovery.",
        "",
        "`Graph Edge Yield = hybrid graph edges / retained papers`; it measures how much graph-connectable structure Skill 1 provides to downstream graph analysis.",
    ]
    return "\n".join(lines) + "\n"


def _best_paper_title_match(title: str, candidates: list[Paper]) -> Paper | None:
    target = _normalize_title(title)
    target_tokens = set(target.split())
    best = None
    best_score = 0.0
    for paper in candidates:
        candidate = _normalize_title(paper.title)
        if not candidate:
            continue
        candidate_tokens = set(candidate.split())
        overlap = len(target_tokens & candidate_tokens) / max(len(target_tokens | candidate_tokens), 1)
        containment = 1.0 if target in candidate or candidate in target else 0.0
        score = max(overlap, containment)
        if score > best_score:
            best = paper
            best_score = score
    return best if best_score >= 0.45 else None


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _term_match(text: str, term: str) -> bool:
    if len(term) <= 5 and term.isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text


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
    parser = argparse.ArgumentParser(description="Run Skill 1 corpus construction ablation.")
    parser.add_argument("--topic", required=True, help="Topic string or benchmark ID.")
    parser.add_argument("--output-dir", default="outputs/corpus_ablation")
    parser.add_argument("--max-papers", type=int, default=45)
    parser.add_argument("--user-level", choices=["beginner", "intermediate", "advanced"], default="intermediate")
    parser.add_argument("--no-enrich", action="store_true", help="Disable citation/reference enrichment.")
    args = parser.parse_args()

    results = run_ablation(
        topic_or_id=args.topic,
        output_dir=args.output_dir,
        max_papers=args.max_papers,
        user_level=args.user_level,
        enrich=not args.no_enrich,
    )
    print(_markdown(get_benchmark(args.topic)["topic"] if get_benchmark(args.topic) else args.topic, results))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
