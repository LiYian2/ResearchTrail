import os
import re
import time
from pathlib import Path
from typing import Any

import gradio as gr

from agent.llm_client import DEFAULT_MODEL
from agent.planner import AgentPlanner


OUTPUT_ROOT = Path("outputs/gradio_demo")


EXAMPLE_TOPICS = [
    "Vision Transformer",
    "Counterfactual regret minimization",
    "Diffusion Models / Score-Based Generative Models",
    "Retrieval-Augmented Generation",
    "Mechanistic Interpretability of Transformers",
    "Graph Neural Networks / Geometric Deep Learning",
    "Self-Supervised Learning in Vision",
    "Efficient Transformers / Long-Context Modeling",
    "RLHF and Preference Optimization",
    "Neural Radiance Fields and 3D Gaussian Splatting",
]


def run_researchtrail(
    topic: str,
    user_level: str,
    goal: str,
    max_papers: int,
    llm_mode: str,
    similarity_backend: str,
    demo_mode: bool,
) -> tuple[str, str, str, Any, str, str | None, str | None, dict]:
    topic = (topic or "").strip()
    if not topic:
        return (
            "Enter a research field first.",
            "",
            "",
            gr.update(choices=[], value=[]),
            "",
            None,
            None,
            {},
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = f"{_slugify(topic)}-{int(time.time())}"
    output_dir = OUTPUT_ROOT / run_id

    agent = AgentPlanner(
        llm_mode=llm_mode,
        llm_provider="siliconflow",
        llm_model=DEFAULT_MODEL,
        output_dir=str(output_dir),
        max_papers_override=int(max_papers),
        similarity_backend=similarity_backend,
    )
    agent.demo = bool(demo_mode)

    prompt = _build_prompt(topic, user_level, goal)
    try:
        transcript = agent.process(prompt)
    except Exception as exc:
        return (
            f"Run failed: {type(exc).__name__}: {exc}",
            "",
            "",
            gr.update(choices=[], value=[]),
            "",
            None,
            None,
            {},
        )

    state = {
        "output_dir": str(output_dir),
        "papers": _paper_lookup(agent),
        "stages": _stages(agent),
        "metrics": _metrics(agent),
        "communities": agent.data.get_metadata("community_labels") or {},
        "transcript": transcript,
    }
    choices = _checkbox_choices(state)
    report_path = output_dir / "research_report.md"
    graph_path = output_dir / "research_graph.png"
    dist_path = output_dir / "scores_distribution.png"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""

    return (
        _status_markdown(agent, output_dir),
        _metrics_html(state),
        _reading_list_html(state, read_ids=[]),
        gr.update(choices=choices, value=[]),
        report,
        str(graph_path) if graph_path.exists() else None,
        str(dist_path) if dist_path.exists() else None,
        state,
    )


def update_reading_list(read_ids: list[str], state: dict) -> str:
    return _reading_list_html(state or {}, read_ids or [])


def load_example(topic: str) -> str:
    return topic


def _build_prompt(topic: str, user_level: str, goal: str) -> str:
    level = (user_level or "intermediate").lower()
    goal = (goal or "build a reading path and identify bridge papers").strip()
    return f"I am a {level} student. I want to understand {topic}. {goal}"


def _paper_lookup(agent: AgentPlanner) -> dict[str, dict[str, Any]]:
    papers = {}
    scores = agent.data.get_all_scores()
    for paper in agent.data.get_all_papers():
        score = scores.get(paper.paper_id)
        papers[paper.paper_id] = {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "abstract": paper.abstract,
            "url": paper.url,
            "citation_count": paper.citation_count,
            "citation_source": paper.citation_source,
            "source": paper.source,
            "pagerank": score.pagerank if score else 0.0,
            "betweenness": score.betweenness if score else 0.0,
            "foundation_score": score.foundation_score if score else 0.0,
            "bridge_score": score.bridge_score if score else 0.0,
            "frontier_score": score.frontier_score if score else 0.0,
            "community": score.community if score else -1,
        }
    return papers


def _stages(agent: AgentPlanner) -> list[dict[str, Any]]:
    path = agent.data.get_reading_path()
    return path.stages if path else []


def _metrics(agent: AgentPlanner) -> dict[str, Any]:
    return {
        "corpus": agent.data.get_corpus_quality() or {},
        "graph": agent.data.get_metadata("graph_metrics") or {},
        "path": agent.data.get_metadata("reading_path_metrics") or {},
        "planner_mode": agent.data.get_metadata("planner_mode") or "unknown",
    }


def _checkbox_choices(state: dict) -> list[tuple[str, str]]:
    choices = []
    for stage in state.get("stages", []):
        for item in stage.get("papers", []):
            pid = item.get("paper_id", "")
            title = item.get("title", "")
            order = item.get("read_order", "")
            if pid and title:
                choices.append((f"{order}. {title}", pid))
    return choices


def _status_markdown(agent: AgentPlanner, output_dir: Path) -> str:
    quality = agent.data.get_corpus_quality() or {}
    graph = agent.data.get_metadata("graph_metrics") or {}
    path = agent.data.get_metadata("reading_path_metrics") or {}
    return "\n".join([
        "### Run Complete",
        f"- Output directory: `{output_dir}`",
        f"- Papers retrieved: **{quality.get('total_papers', 0)}**",
        f"- Communities: **{graph.get('community_count', 0)}**",
        f"- Similarity backend: **{graph.get('similarity_backend', 'tfidf')}**",
        f"- Reading path papers: **{path.get('unique_paper_count', 0)}**",
        f"- Planner mode: **{agent.data.get_metadata('planner_mode') or 'unknown'}**",
    ])


def _metrics_html(state: dict) -> str:
    metrics = state.get("metrics", {})
    corpus = metrics.get("corpus", {})
    graph = metrics.get("graph", {})
    path = metrics.get("path", {})
    communities = state.get("communities", {})

    cards = [
        ("Papers", corpus.get("total_papers", 0)),
        ("Abstract Coverage", _pct(corpus.get("has_abstract_ratio", 0))),
        ("Citation Coverage", _pct(corpus.get("has_citation_ratio", 0))),
        ("Year Range", corpus.get("year_range", "N/A")),
        ("Edges", graph.get("edge_count", 0)),
        ("Citation Edges", graph.get("citation_edges", 0)),
        ("Similarity Edges", graph.get("similarity_edges", 0)),
        ("Similarity Backend", graph.get("similarity_backend", "tfidf")),
        ("Communities", graph.get("community_count", 0)),
        ("Path Stages", path.get("stage_count", 0)),
        ("Path Papers", path.get("unique_paper_count", 0)),
        ("Explanation Coverage", _pct(path.get("explanation_availability_ratio", 0))),
    ]
    card_html = "".join(
        f"<div class='metric-card'><span>{label}</span><strong>{value}</strong></div>"
        for label, value in cards
    )

    community_rows = []
    for cid, label in sorted(communities.items(), key=lambda x: int(x[0])):
        community_rows.append(
            "<tr>"
            f"<td>{cid}</td>"
            f"<td>{_escape(label.get('label', 'Unlabeled'))}</td>"
            f"<td>{_escape(label.get('description', ''))}</td>"
            "</tr>"
        )
    community_table = (
        "<table class='community-table'><thead><tr><th>ID</th><th>Label</th><th>Description</th></tr></thead>"
        f"<tbody>{''.join(community_rows)}</tbody></table>"
        if community_rows else "<p>No community labels available.</p>"
    )

    return f"<div class='metric-grid'>{card_html}</div><h3>Detected Communities</h3>{community_table}"


def _reading_list_html(state: dict, read_ids: list[str]) -> str:
    read = set(read_ids or [])
    papers = state.get("papers", {})
    stages = state.get("stages", [])
    if not stages:
        return "<p>No reading path generated yet.</p>"

    blocks = []
    total = 0
    done = 0
    for stage in stages:
        cards = []
        for item in stage.get("papers", []):
            total += 1
            pid = item.get("paper_id", "")
            paper = papers.get(pid, {})
            if pid in read:
                done += 1
                cards.append(
                    "<details class='paper-card read-card'>"
                    f"<summary>Read: {item.get('read_order')}. {_escape(item.get('title', ''))}</summary>"
                    f"<p>{_escape(item.get('reason', ''))}</p>"
                    "</details>"
                )
                continue
            cards.append(_paper_card(item, paper))
        blocks.append(
            f"<section class='stage-block'><h3>{_escape(stage.get('stage', 'Stage'))}</h3>{''.join(cards)}</section>"
        )
    progress = f"<div class='progress-note'>Progress: {done}/{total} papers marked as read.</div>"
    return progress + "".join(blocks)


def _paper_card(item: dict, paper: dict) -> str:
    authors = ", ".join((paper.get("authors") or [])[:3])
    if len(paper.get("authors") or []) > 3:
        authors += " et al."
    citation_label = paper.get("citation_source") or "citations"
    url = paper.get("url") or ""
    url_html = f"<a href='{_escape(url)}' target='_blank'>Open paper</a>" if url else "No URL"
    abstract = _shorten(paper.get("abstract", ""), 900)
    scores = (
        f"PR {paper.get('pagerank', 0):.4f} · "
        f"Bridge {paper.get('bridge_score', 0):.3f} · "
        f"Frontier {paper.get('frontier_score', 0):.3f}"
    )
    return (
        "<article class='paper-card'>"
        f"<div class='paper-title'>{item.get('read_order')}. {_escape(item.get('title', ''))}</div>"
        f"<div class='paper-meta'>{_escape(authors)} · {paper.get('year', 'N/A')} · "
        f"{_escape(str(citation_label))}: {paper.get('citation_count', 0)} · {url_html}</div>"
        f"<div class='score-line'>{_escape(scores)}</div>"
        f"<p><strong>Why read:</strong> {_escape(item.get('reason', ''))}</p>"
        f"<details><summary>Abstract</summary><p>{_escape(abstract)}</p></details>"
        "</article>"
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:60] or "researchtrail"


def _pct(value: float) -> str:
    try:
        return f"{float(value):.1%}"
    except Exception:
        return "0.0%"


def _shorten(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _escape(value: Any) -> str:
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


CSS = """
.gradio-container { max-width: 1440px !important; }
.hero {
  border-bottom: 1px solid #d7dde5;
  padding: 10px 0 16px 0;
  margin-bottom: 12px;
}
.hero h1 { font-size: 30px; margin: 0 0 4px 0; letter-spacing: 0; }
.hero p { margin: 0; color: #4c5563; }
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}
.metric-card {
  border: 1px solid #d8dee7;
  border-radius: 8px;
  padding: 10px 12px;
  background: #fbfcfe;
}
.metric-card span {
  display: block;
  color: #606b7b;
  font-size: 12px;
}
.metric-card strong {
  display: block;
  margin-top: 4px;
  font-size: 18px;
}
.community-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}
.community-table th, .community-table td {
  border: 1px solid #d8dee7;
  padding: 8px;
  vertical-align: top;
}
.progress-note {
  margin: 6px 0 12px 0;
  padding: 8px 10px;
  border: 1px solid #d8dee7;
  border-radius: 8px;
  background: #f7faf9;
}
.stage-block { margin-bottom: 18px; }
.stage-block h3 {
  border-bottom: 1px solid #d8dee7;
  padding-bottom: 6px;
}
.paper-card {
  border: 1px solid #d8dee7;
  border-radius: 8px;
  padding: 12px;
  margin: 10px 0;
  background: #ffffff;
}
.read-card {
  background: #f2f5f7;
  color: #566171;
}
.paper-title {
  font-weight: 700;
  font-size: 16px;
  line-height: 1.35;
}
.paper-meta, .score-line {
  color: #5b6572;
  font-size: 13px;
  margin-top: 4px;
}
.paper-card p { margin: 8px 0 0 0; }
"""


with gr.Blocks(title="ResearchTrail Demo") as demo:
    state = gr.State({})

    gr.HTML(
        "<div class='hero'><h1>ResearchTrail</h1>"
        "<p>LLM-assisted, code-backed reading path generation over citation and similarity networks.</p></div>"
    )

    with gr.Row():
        with gr.Column(scale=4):
            topic = gr.Textbox(
                label="Research field",
                placeholder="e.g. Diffusion Models / Score-Based Generative Models",
                lines=2,
            )
        with gr.Column(scale=2):
            example = gr.Dropdown(EXAMPLE_TOPICS, label="Demo examples", value=None)
            example.change(load_example, inputs=example, outputs=topic)

    with gr.Row():
        user_level = gr.Radio(
            ["beginner", "intermediate", "advanced"],
            value="intermediate",
            label="User background",
        )
        max_papers = gr.Slider(20, 120, value=45, step=5, label="Max papers")
        llm_mode = gr.Radio(["auto", "off"], value="auto", label="LLM mode")
        similarity_backend = gr.Radio(["tfidf", "lsa"], value="tfidf", label="Similarity backend")
        demo_mode = gr.Checkbox(value=False, label="Demo mode")

    goal = gr.Textbox(
        label="Goal",
        value="Build a reading path, identify bridge papers, and explain the main conceptual shifts.",
        lines=2,
    )

    run_button = gr.Button("Generate Reading Path", variant="primary")
    status = gr.Markdown()

    with gr.Tab("Reading List"):
        read_selector = gr.CheckboxGroup(label="Mark papers as read")
        reading_html = gr.HTML()

    with gr.Tab("Analysis"):
        metrics_html = gr.HTML()
        with gr.Row():
            graph_image = gr.Image(label="Research Network", type="filepath")
            score_image = gr.Image(label="Score Distributions", type="filepath")

    with gr.Tab("Report"):
        report_md = gr.Markdown()

    run_button.click(
        run_researchtrail,
        inputs=[topic, user_level, goal, max_papers, llm_mode, similarity_backend, demo_mode],
        outputs=[status, metrics_html, reading_html, read_selector, report_md, graph_image, score_image, state],
    )
    read_selector.change(update_reading_list, inputs=[read_selector, state], outputs=reading_html)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860, css=CSS)
