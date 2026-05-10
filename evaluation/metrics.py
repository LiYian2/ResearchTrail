import re
from typing import Any

from shared.data_layer import SharedDataLayer
from evaluation.benchmark_registry import get_benchmark


def collect_metrics(data: SharedDataLayer) -> dict:
    quality = collect_quality_metrics(data)
    return {
        "retrieval": data.get_corpus_quality(),
        "graph": data.get_metadata("graph_metrics") or {},
        "reading_path": data.get_metadata("reading_path_metrics") or {},
        "quality": quality,
        "planner_mode": data.get_metadata("planner_mode"),
        "query_plan": data.get_metadata("query_plan") or {},
        "community_labels": data.get_metadata("community_labels") or {},
    }


def collect_quality_metrics(data: SharedDataLayer) -> dict[str, Any]:
    profile = data.get_research_profile()
    topic = profile.topic if profile else ""
    papers = data.get_all_papers()
    path = data.get_reading_path()
    scores = data.get_all_scores()

    path_titles = []
    path_paper_ids = []
    stage_positions: dict[str, int] = {}
    if path:
        for stage_idx, stage in enumerate(path.stages):
            for paper in stage.get("papers", []):
                title = paper.get("title", "")
                path_titles.append(title)
                pid = paper.get("paper_id") or _find_paper_id_by_title(data, title)
                if pid:
                    path_paper_ids.append(pid)
                stage_positions[title] = stage_idx

    landmarks = _landmark_specs(topic)
    landmark_hits = _landmark_hits(path_titles, landmarks)
    topic_precision = _topic_precision(topic, papers)
    ordering_quality = _ordering_quality(stage_positions, landmarks)
    community_coverage = _community_coverage(path_paper_ids, scores)
    path_stage_coverage = _path_stage_coverage(path)
    source_mix = _source_mix(papers)

    return {
        "landmark_hit_rate": landmark_hits["hit_rate"],
        "landmark_hits": landmark_hits["hits"],
        "landmark_total": landmark_hits["total"],
        "topic_precision": topic_precision,
        "ordering_quality": ordering_quality,
        "community_coverage": community_coverage,
        "path_stage_coverage": path_stage_coverage,
        "source_mix": source_mix,
    }


def markdown_table(results: dict[str, dict]) -> str:
    lines = [
        "| Run | Papers | Edges | Components | Largest Component | Communities | Modularity | Stages | Path Papers | Landmark Hit | Topic Precision | Ordering | Community Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        retrieval = metrics.get("retrieval", {})
        graph = metrics.get("graph", {})
        path = metrics.get("reading_path", {})
        quality = metrics.get("quality", {})
        lines.append(
            "| {name} | {papers} | {edges} | {components} | {largest:.1%} | {communities} | {modularity:.4f} | {stages} | {path_papers} | {landmark:.1%} | {precision:.1%} | {ordering:.1%} | {coverage:.1%} |".format(
                name=name,
                papers=retrieval.get("total_papers", 0),
                edges=graph.get("edge_count", 0),
                components=graph.get("connected_components", 0),
                largest=graph.get("largest_component_ratio", 0),
                communities=graph.get("community_count", 0),
                modularity=graph.get("modularity", 0),
                stages=path.get("stage_count", 0),
                path_papers=path.get("unique_paper_count", 0),
                landmark=quality.get("landmark_hit_rate", 0),
                precision=quality.get("topic_precision", 0),
                ordering=quality.get("ordering_quality", 0),
                coverage=quality.get("community_coverage", 0),
            )
        )
    return "\n".join(lines)


def _landmark_specs(topic: str) -> list[dict[str, str]]:
    benchmark = get_benchmark(topic)
    if benchmark:
        return benchmark.get("expected_landmarks", [])
    t = topic.lower()
    if "counterfactual" in t and ("regret" in t or "cfr" in t):
        return [
            {"title": "Regret Minimization in Games with Incomplete Information", "role": "foundation"},
            {"title": "Monte Carlo Sampling for Regret Minimization in Extensive Games", "role": "foundation"},
            {"title": "Deep Counterfactual Regret Minimization", "role": "development"},
        ]
    if ("vision" in t or "image" in t or "vit" in t) and "transformer" in t:
        return [
            {"title": "An Image is Worth 16x16 Words", "role": "foundation"},
            {"title": "Training data-efficient image transformers", "role": "development"},
            {"title": "Swin Transformer", "role": "development"},
            {"title": "Emerging Properties in Self-Supervised Vision Transformers", "role": "development"},
            {"title": "Masked Autoencoders Are Scalable Vision Learners", "role": "development"},
        ]
    return []


def _landmark_hits(path_titles: list[str], landmarks: list[dict[str, str]]) -> dict[str, Any]:
    if not landmarks:
        return {"hit_rate": 0.0, "hits": [], "total": 0}
    hits = []
    normalized_titles = [_normalize_title(title) for title in path_titles]
    for landmark in landmarks:
        target = _normalize_title(landmark["title"])
        if any(target in title or title in target for title in normalized_titles):
            hits.append(landmark["title"])
    return {
        "hit_rate": len(hits) / len(landmarks),
        "hits": hits,
        "total": len(landmarks),
    }


def _topic_precision(topic: str, papers: list) -> float:
    if not papers:
        return 0.0
    relevant = sum(1 for paper in papers if _is_topic_relevant(topic, paper.title, paper.abstract))
    return relevant / len(papers)


def _is_topic_relevant(topic: str, title: str, abstract: str) -> bool:
    t = topic.lower()
    text = f" {title} {abstract} ".lower()
    benchmark = get_benchmark(topic)
    if benchmark:
        acronyms = {
            "rag", "dpr", "retro", "gcn", "gat", "moco", "byol", "dino", "dinov2", "mae", "beit",
            "mamba", "ppo", "dpo", "rlaif", "nerf", "maddpg", "qmix", "mappo", "fedavg", "fedprox",
        }
        terms = [
            str(term).lower()
            for term in benchmark.get("topic_terms", [])
            if " " in str(term) or "-" in str(term) or str(term).lower() in acronyms
        ]
        hits = sum(1 for term in terms if _term_match(text, term))
        if hits:
            return True
        landmark_titles = [
            str(item.get("title", "")).lower()
            for item in benchmark.get("expected_landmarks", [])
            if item.get("title")
        ]
        return any(_normalize_title(title) in _normalize_title(text) for title in landmark_titles)
    if "counterfactual" in t and ("regret" in t or "cfr" in t):
        required = [
            "counterfactual regret",
            " cfr",
            "cfr ",
            "regret minimization",
            "imperfect information",
            "imperfect-information",
            "extensive form",
            "extensive-form",
            "poker",
        ]
        return any(term in text for term in required)
    if ("vision" in t or "image" in t or "vit" in t) and "transformer" in t:
        vision = ["vision", "visual", "image", "segmentation", "classification", "detection"]
        transformer = ["vision transformer", "transformer", "vit", "deit", "swin", "dino", "mae", "beit"]
        return any(v in text for v in vision) and any(_term_match(text, term) for term in transformer)
    topic_terms = [term for term in re.findall(r"[a-z]{4,}", t) if term not in {"research", "field", "understand"}]
    return bool(topic_terms) and sum(1 for term in topic_terms if term in text) >= max(1, min(2, len(topic_terms)))


def _ordering_quality(stage_positions: dict[str, int], landmarks: list[dict[str, str]]) -> float:
    if not landmarks or not stage_positions:
        return 0.0
    role_rank = {"prerequisite": 0, "foundation": 1, "core": 2, "development": 3, "bridge": 4, "frontier": 5}
    observed = []
    for landmark in landmarks:
        target = _normalize_title(landmark["title"])
        for title, stage_idx in stage_positions.items():
            title_key = _normalize_title(title)
            if target in title_key or title_key in target:
                observed.append((role_rank.get(landmark.get("role", "development"), 3), stage_idx))
                break
    if len(observed) < 2:
        return 1.0 if observed else 0.0
    comparisons = 0
    correct = 0
    for i in range(len(observed)):
        for j in range(i + 1, len(observed)):
            if observed[i][0] == observed[j][0]:
                continue
            comparisons += 1
            earlier_role, earlier_stage = observed[i]
            later_role, later_stage = observed[j]
            if (earlier_role < later_role and earlier_stage <= later_stage) or (
                earlier_role > later_role and earlier_stage >= later_stage
            ):
                correct += 1
    return correct / comparisons if comparisons else 1.0


def _community_coverage(path_paper_ids: list[str], scores: dict) -> float:
    all_communities = {score.community for score in scores.values() if score.community >= 0}
    if not all_communities:
        return 0.0
    path_communities = {
        scores[pid].community
        for pid in path_paper_ids
        if pid in scores and scores[pid].community >= 0
    }
    return len(path_communities) / len(all_communities)


def _path_stage_coverage(path) -> float:
    if not path or not path.stages:
        return 0.0
    stage_names = " ".join(stage.get("stage", "").lower() for stage in path.stages)
    expected = ["foundation", "core", "development", "frontier"]
    hits = sum(1 for term in expected if term in stage_names or ("recent" in stage_names and term == "frontier"))
    return hits / len(expected)


def _source_mix(papers: list) -> dict[str, int]:
    counts: dict[str, int] = {}
    for paper in papers:
        source = getattr(paper, "source", "") or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _find_paper_id_by_title(data: SharedDataLayer, title: str) -> str:
    target = _normalize_title(title)
    for paper in data.get_all_papers():
        if _normalize_title(paper.title) == target:
            return paper.paper_id
    return ""


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _term_match(text: str, term: str) -> bool:
    if len(term) <= 5 and term.isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    return term in text
