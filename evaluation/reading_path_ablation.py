import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

from evaluation.benchmark_registry import get_benchmark
from evaluation.metrics import _is_topic_relevant
from shared.data_layer import SharedDataLayer
from shared.types import GraphData, NodeScores, Paper, ReadingPath, ReadingPathItem, ResearchProfile, UserLevel
from skill_reading_path.path_generator import ReadingPathSkill


VARIANTS = ["random_order", "citation_count_order", "pagerank_order", "researchtrail_staged"]


def run_from_state_root(state_root: str, output_dir: str, limit_topics: list[str] | None = None) -> dict[str, dict]:
    root = Path(state_root)
    state_files = sorted(root.glob("*/state.json"))
    if limit_topics:
        wanted = set(limit_topics)
        state_files = [p for p in state_files if p.parent.name in wanted]

    os.makedirs(output_dir, exist_ok=True)
    all_results: dict[str, dict] = {}
    path_dir = Path(output_dir, "paths_for_human_eval")
    path_dir.mkdir(parents=True, exist_ok=True)

    for state_path in state_files:
        topic_id = state_path.parent.name
        data = _load_state(state_path)
        profile = data.get_research_profile()
        topic = profile.topic if profile else topic_id
        benchmark = get_benchmark(topic) or get_benchmark(topic_id)

        network_path = ReadingPathSkill(data).run()
        path_len = max(5, len(network_path.items))
        variant_paths = {
            "random_order": _flat_path(data, "Random Order", _rank_random(data, topic_id), path_len),
            "citation_count_order": _flat_path(data, "Citation Count Ranking", _rank_citation(data), path_len),
            "pagerank_order": _flat_path(data, "PageRank Ranking", _rank_pagerank(data), path_len),
            "researchtrail_staged": network_path,
        }

        topic_results = {}
        for variant, path in variant_paths.items():
            topic_results[variant] = _path_metrics(data, path, benchmark)
        all_results[topic_id] = topic_results
        Path(path_dir, f"{topic_id}.md").write_text(
            _human_eval_markdown(topic_id, topic, variant_paths, topic_results),
            encoding="utf-8",
        )

    Path(output_dir, "reading_path_ablation_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(output_dir, "reading_path_ablation_summary.md").write_text(
        _summary_markdown(all_results),
        encoding="utf-8",
    )
    Path(output_dir, "manual_eval_template.md").write_text(_manual_eval_template(), encoding="utf-8")
    return all_results


def _load_state(path: Path) -> SharedDataLayer:
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = SharedDataLayer()
    profile_raw = raw.get("research_profile") or {}
    level = {
        "beginner": UserLevel.BEGINNER,
        "intermediate": UserLevel.INTERMEDIATE,
        "advanced": UserLevel.ADVANCED,
    }.get(profile_raw.get("user_level", "intermediate"), UserLevel.INTERMEDIATE)
    data.set_research_profile(ResearchProfile(
        topic=profile_raw.get("topic", ""),
        seed_papers=profile_raw.get("seed_papers", []),
        time_range=profile_raw.get("time_range", "all"),
        max_papers=profile_raw.get("max_papers", 100),
        user_level=level,
        preferred_length=profile_raw.get("preferred_length", 12),
        goal=profile_raw.get("goal", "reading_path_ablation"),
    ))
    data.add_papers([Paper.from_dict(p) for p in (raw.get("papers") or {}).values()])
    data.set_node_scores({
        pid: NodeScores.from_dict(score)
        for pid, score in (raw.get("node_scores") or {}).items()
    })
    if raw.get("graph_data"):
        data.set_graph_data(GraphData.from_dict(raw["graph_data"]))
    data.set_corpus_quality(raw.get("corpus_quality") or {})
    for key, value in (raw.get("metadata") or {}).items():
        data.set_metadata(key, value)
    return data


def _flat_path(data: SharedDataLayer, stage_name: str, ranked_ids: list[str], path_len: int) -> ReadingPath:
    papers = []
    items = []
    order = 0
    for pid in ranked_ids[:path_len]:
        paper = data.get_paper(pid)
        if not paper:
            continue
        order += 1
        reason = _baseline_reason(stage_name, paper, data.get_node_scores(pid))
        info = {
            "paper_id": pid,
            "title": paper.title,
            "reason": reason,
            "read_order": order,
            "year": paper.year,
            "citation_count": paper.citation_count,
        }
        papers.append(info)
        items.append(ReadingPathItem(pid, paper.title, reason, order, stage_name))
    return ReadingPath(stages=[{"stage": stage_name, "papers": papers}], items=items)


def _rank_random(data: SharedDataLayer, seed_text: str) -> list[str]:
    ids = [p.paper_id for p in data.get_all_papers()]
    rng = random.Random(seed_text)
    rng.shuffle(ids)
    return ids


def _rank_citation(data: SharedDataLayer) -> list[str]:
    return [
        p.paper_id
        for p in sorted(data.get_all_papers(), key=lambda p: (p.citation_count, p.year), reverse=True)
    ]


def _rank_pagerank(data: SharedDataLayer) -> list[str]:
    scores = data.get_all_scores()
    return [
        pid for pid, _ in sorted(scores.items(), key=lambda item: item[1].pagerank, reverse=True)
    ]


def _baseline_reason(stage_name: str, paper: Paper, score: NodeScores | None) -> str:
    if stage_name == "Citation Count Ranking":
        return f"Selected because it has high citation count ({paper.citation_count}) among the retrieved corpus."
    if stage_name == "PageRank Ranking" and score:
        return f"Selected because it has high PageRank ({score.pagerank:.4f}) in the paper graph."
    return "Selected as a baseline comparison item without network-aware staging."


def _path_metrics(data: SharedDataLayer, path: ReadingPath, benchmark: dict | None) -> dict[str, Any]:
    profile = data.get_research_profile()
    topic = profile.topic if profile else ""
    paper_ids = [item.paper_id for item in path.items]
    papers = [data.get_paper(pid) for pid in paper_ids if data.get_paper(pid)]
    scores = data.get_all_scores()
    return {
        "path_papers": len(paper_ids),
        "landmark_hit_rate": _landmark_hit_rate(papers, benchmark),
        "ordering_quality": _ordering_quality(path.items, benchmark),
        "community_coverage": _community_coverage(paper_ids, scores),
        "stage_coverage": _stage_coverage(path),
        "topic_precision": sum(1 for p in papers if _is_topic_relevant(topic, p.title, p.abstract)) / max(len(papers), 1),
        "explanation_coverage": sum(1 for item in path.items if item.reason) / max(len(path.items), 1),
        "unique_stage_count": len(path.stages),
    }


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


def _ordering_quality(items: list[ReadingPathItem], benchmark: dict | None) -> float:
    landmarks = (benchmark or {}).get("expected_landmarks", [])
    if not landmarks:
        return 0.0
    title_to_pos = {_normalize_title(item.title): idx for idx, item in enumerate(items)}
    role_rank = {"prerequisite": 0, "foundation": 1, "core": 2, "development": 3, "bridge": 4, "frontier": 5}
    observed = []
    for landmark in landmarks:
        target = _normalize_title(landmark.get("title", ""))
        for title, pos in title_to_pos.items():
            if target and (target in title or title in target):
                observed.append((role_rank.get(landmark.get("role", "development"), 3), pos))
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
            role_i, pos_i = observed[i]
            role_j, pos_j = observed[j]
            if (role_i < role_j and pos_i <= pos_j) or (role_i > role_j and pos_i >= pos_j):
                correct += 1
    return correct / comparisons if comparisons else 1.0


def _community_coverage(paper_ids: list[str], scores: dict[str, NodeScores]) -> float:
    all_communities = {score.community for score in scores.values() if score.community >= 0}
    if not all_communities:
        return 0.0
    path_communities = {
        scores[pid].community
        for pid in paper_ids
        if pid in scores and scores[pid].community >= 0
    }
    return len(path_communities) / len(all_communities)


def _stage_coverage(path: ReadingPath) -> float:
    names = " ".join(stage.get("stage", "").lower() for stage in path.stages)
    expected = ["foundation", "core", "development", "frontier"]
    hits = sum(1 for term in expected if term in names or ("recent" in names and term == "frontier"))
    return hits / len(expected)


def _summary_markdown(results: dict[str, dict]) -> str:
    lines = [
        "# Skill 3 Reading Path Ablation",
        "",
        "| Topic | Variant | Papers | Landmark Hit | Ordering | Community Coverage | Stage Coverage | Topic Precision | Explanation Coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topic_id, topic_results in results.items():
        for variant, metrics in topic_results.items():
            lines.append(
                "| {topic} | {variant} | {papers} | {landmark:.1%} | {ordering:.1%} | {community:.1%} | {stage:.1%} | {precision:.1%} | {explain:.1%} |".format(
                    topic=topic_id,
                    variant=variant,
                    papers=metrics["path_papers"],
                    landmark=metrics["landmark_hit_rate"],
                    ordering=metrics["ordering_quality"],
                    community=metrics["community_coverage"],
                    stage=metrics["stage_coverage"],
                    precision=metrics["topic_precision"],
                    explain=metrics["explanation_coverage"],
                )
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Random order is a lower-bound sanity baseline.",
        "- Citation-count order tests whether popularity alone creates a useful reading path.",
        "- PageRank order tests graph centrality without staged pedagogy.",
        "- ResearchTrail staged path tests whether graph roles can be converted into a structured learning sequence.",
    ]
    return "\n".join(lines) + "\n"


def _human_eval_markdown(
    topic_id: str,
    topic: str,
    paths: dict[str, ReadingPath],
    metrics: dict[str, dict],
) -> str:
    lines = [f"# Human Evaluation Paths: {topic_id}", "", f"Topic: {topic}", ""]
    for variant, path in paths.items():
        m = metrics[variant]
        lines += [
            f"## {variant}",
            "",
            f"Auto metrics: landmark={m['landmark_hit_rate']:.1%}, ordering={m['ordering_quality']:.1%}, community={m['community_coverage']:.1%}, stage={m['stage_coverage']:.1%}",
            "",
        ]
        for stage in path.stages:
            lines.append(f"### {stage.get('stage', 'Stage')}")
            for paper in stage.get("papers", []):
                lines.append(f"{paper.get('read_order')}. {paper.get('title')} ({paper.get('year', 'N/A')})")
                lines.append(f"   - Why read: {paper.get('reason', '')}")
            lines.append("")
    return "\n".join(lines)


def _manual_eval_template() -> str:
    return """# Manual Reading Path Evaluation Template

Score each variant from 1-5.

Fields:
- Coherence: does the sequence form a meaningful learning trail?
- Coverage: does it include important papers and method families?
- Ordering: are foundations before developments/frontiers?
- User fit: is it appropriate for the intended level?
- Explanation usefulness: are why-read notes actionable?
- Overall usefulness: would you use this path?

```text
Topic:
Variant:
Coherence:
Coverage:
Ordering:
User fit:
Explanation usefulness:
Overall usefulness:
Major omissions:
Ordering problems:
Notes:
```
"""


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Skill 3 reading path ablation from existing state files.")
    parser.add_argument("--state-root", default="outputs/benchmark_full_llm_normalized")
    parser.add_argument("--output-dir", default="outputs/reading_path_ablation")
    parser.add_argument("--topics", nargs="*", default=None, help="Optional topic IDs to include.")
    args = parser.parse_args()

    results = run_from_state_root(args.state_root, args.output_dir, args.topics)
    print(_summary_markdown(results))
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
