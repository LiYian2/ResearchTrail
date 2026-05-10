from shared.data_layer import SharedDataLayer
from shared.types import Paper, NodeScores, ReadingPath


class ReportGenerator:
    def __init__(self, data_layer: SharedDataLayer, abstract_max_chars: int = 1600):
        self.data = data_layer
        self.abstract_max_chars = abstract_max_chars

    def generate(self) -> str:
        path = self.data.get_reading_path()
        if not path:
            return "No reading path available."

        lines: list[str] = []
        profile = self.data.get_research_profile()

        lines.append("# ResearchTrail: Literature Reading Path Report\n")
        if profile:
            lines.append(f"**Topic:** {profile.topic}")
            lines.append(f"**Level:** {profile.user_level.value}")
            lines.append(f"**Goal:** {profile.goal}")
            lines.append(f"**Preferred Length:** {profile.preferred_length} papers\n")

        lines.append("---\n")
        self._append_summary_sections(lines)

        for stage in path.stages:
            lines.append(f"## {stage['stage']}")
            lines.append("")
            for paper in stage["papers"]:
                lines.append(f"### {paper['read_order']}. {paper['title']}")
                p = self.data.get_paper(paper["paper_id"])
                if p:
                    lines.append(f"**Authors:** {', '.join(p.authors[:3])}{' et al.' if len(p.authors) > 3 else ''}")
                    citation_label = self._citation_label(p)
                    lines.append(f"**Year:** {p.year} | **{citation_label}:** {p.citation_count}")
                    if p.source == "demo":
                        lines.append("**Source:** Synthetic demo record")
                    elif p.url:
                        lines.append(f"**URL:** {p.url}")
                    if p.abstract:
                        lines.append(f"**Abstract:** {self._shorten_text(p.abstract, self.abstract_max_chars)}")
                lines.append(f"**Why read:** {paper['reason']}")
                lines.append("")

        lines.append("---\n")

        scores = self.data.get_all_scores()
        if scores:
            lines.append("## Research Graph Statistics\n")
            community_info = self._get_community_summary(scores)
            lines.append(f"- **Communities detected:** {community_info['num_communities']}")
            lines.append(f"- **Community sizes:** {community_info['sizes']}")
            top_by_pr = sorted(scores.items(), key=lambda x: x[1].pagerank, reverse=True)[:3]
            lines.append("\n**Top papers by PageRank:**")
            for pid, sc in top_by_pr:
                p = self.data.get_paper(pid)
                if p:
                    lines.append(f"- {p.title} (PR: {sc.pagerank:.4f}, {p.year})")

            top_by_bc = sorted(scores.items(), key=lambda x: x[1].betweenness, reverse=True)[:3]
            lines.append("\n**Top bridge papers:**")
            for pid, sc in top_by_bc:
                p = self.data.get_paper(pid)
                if p:
                    lines.append(f"- {p.title} (BC: {sc.betweenness:.4f})")

        lines.append("\n---\n")
        lines.append("## Follow-up Questions You Can Ask\n")
        for q in (path.followup_suggestions or []):
            lines.append(f"- \"{q}\"")

        return "\n".join(lines)

    def _append_summary_sections(self, lines: list[str]) -> None:
        quality = self.data.get_corpus_quality()
        graph_metrics = self.data.get_metadata("graph_metrics") or {}
        path_metrics = self.data.get_metadata("reading_path_metrics") or {}
        community_labels = self.data.get_metadata("community_labels") or {}

        if quality:
            lines.append("## Corpus Summary\n")
            if quality.get("demo_mode") or self.data.get_metadata("demo_mode"):
                lines.append("> **Demo mode:** this corpus contains synthetic demonstration records. Use live mode without `--demo` for real papers and external URLs.\n")
            lines.append(f"- Papers: {quality.get('total_papers', 0)}")
            lines.append(f"- Abstract coverage: {quality.get('has_abstract_ratio', 0):.1%}")
            lines.append(f"- Citation/reference coverage: {quality.get('has_citation_ratio', 0):.1%}")
            lines.append("- Citation counts prefer Semantic Scholar `citationCount` when `S2_API_KEY` is available, then fall back to OpenAlex `cited_by_count` or curated landmark metadata.")
            lines.append(f"- Year range: {quality.get('year_range', 'N/A')}")
            lines.append(f"- Deduplicated records: {quality.get('deduplication_removed', 0)}")
            lines.append("")

        if graph_metrics:
            lines.append("## Field Map Metrics\n")
            lines.append(f"- Nodes: {graph_metrics.get('node_count', 0)}")
            lines.append(f"- Edges: {graph_metrics.get('edge_count', 0)}")
            lines.append(f"- Citation edges: {graph_metrics.get('citation_edges', 0)}")
            lines.append(f"- Similarity edges: {graph_metrics.get('similarity_edges', 0)}")
            lines.append(f"- Communities: {graph_metrics.get('community_count', 0)}")
            lines.append(f"- Modularity: {graph_metrics.get('modularity', 0)}")
            lines.append(f"- Largest component ratio: {graph_metrics.get('largest_component_ratio', 0):.1%}")
            lines.append("")

        if community_labels:
            lines.append("## Detected Research Communities\n")
            for cid, label in sorted(community_labels.items(), key=lambda x: int(x[0])):
                lines.append(f"- **Community {cid}: {label.get('label', 'Unlabeled')}** — {label.get('description', '')}")
            lines.append("")

        if path_metrics:
            lines.append("## Reading Path Metrics\n")
            lines.append(f"- Stages: {path_metrics.get('stage_count', 0)}")
            lines.append(f"- Unique papers: {path_metrics.get('unique_paper_count', 0)}")
            lines.append(f"- Explanation coverage: {path_metrics.get('explanation_availability_ratio', 0):.1%}")
            lines.append("")

    def _get_community_summary(self, scores: dict[str, NodeScores]) -> dict:
        comms: dict[int, int] = {}
        for sc in scores.values():
            comms[sc.community] = comms.get(sc.community, 0) + 1
        return {
            "num_communities": len(comms),
            "sizes": list(comms.values()),
        }

    def explain_paper_role(self, paper_id: str) -> str:
        paper = self.data.get_paper(paper_id)
        scores = self.data.get_node_scores(paper_id)

        if not paper:
            return f"No paper found with ID: {paper_id}"

        lines = [
            f"# Paper Role Analysis: {paper.title}\n",
            f"**Authors:** {', '.join(paper.authors)}",
            f"**Year:** {paper.year}",
            f"**{self._citation_label(paper)}:** {paper.citation_count}\n",
        ]

        if scores:
            role = self._classify_paper_role(scores)
            lines.append(f"**Primary Role:** {role}\n")
            lines.append(f"- PageRank: {scores.pagerank:.4f}")
            lines.append(f"- Betweenness Centrality: {scores.betweenness:.4f}")
            lines.append(f"- Community: {scores.community}")
            lines.append(f"- Foundation Score: {scores.foundation_score:.4f}")
            lines.append(f"- Bridge Score: {scores.bridge_score:.4f}")
            lines.append(f"- Frontier Score: {scores.frontier_score:.4f}")

        return "\n".join(lines)

    def _classify_paper_role(self, scores: NodeScores) -> str:
        if scores.foundation_score > max(scores.bridge_score, scores.frontier_score):
            return "Foundation — Establishes core concepts for the field"
        elif scores.bridge_score > max(scores.foundation_score, scores.frontier_score):
            return "Bridge — Connects multiple research communities"
        elif scores.frontier_score > max(scores.foundation_score, scores.bridge_score):
            return "Frontier — At the cutting edge of recent research"
        else:
            return "Core — Central methodological contribution"

    @staticmethod
    def _shorten_text(text: str, max_chars: int) -> str:
        import re

        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars].rstrip() + "... [truncated in report]"

    @staticmethod
    def _citation_label(paper: Paper) -> str:
        if paper.source == "demo":
            return "Synthetic citations"
        if paper.citation_source == "semantic_scholar":
            return "Citations (Semantic Scholar)"
        if paper.citation_source == "openalex":
            return "Citations (OpenAlex)"
        if paper.citation_source == "curated":
            return "Citations (curated metadata)"
        return "Citations"

    def generate_bridge_report(self) -> str:
        scores = self.data.get_all_scores()
        if not scores:
            return "No graph scores available."

        sorted_by_bc = sorted(scores.items(), key=lambda x: x[1].bridge_score, reverse=True)[:10]
        lines = ["# Bridge Papers Report\n"]
        lines.append("Papers that connect multiple research communities:\n")

        for pid, sc in sorted_by_bc:
            paper = self.data.get_paper(pid)
            if paper:
                lines.append(f"### {paper.title} ({paper.year})")
                lines.append(f"- Bridge Score: {sc.bridge_score:.4f}")
                lines.append(f"- Betweenness: {sc.betweenness:.4f}")
                lines.append(f"- Community: {sc.community}")
                lines.append(f"- Citations: {paper.citation_count}")
                lines.append("")

        return "\n".join(lines)

    def generate_seven_day_plan(self) -> str:
        path = self.data.get_reading_path()
        if not path:
            return "No reading path available."

        items = path.items
        if not items:
            return "No papers in the reading path."

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        lines = ["# 7-Day Reading Plan\n"]
        per_day = max(1, len(items) // 7)

        for i, day in enumerate(days):
            start = i * per_day
            end = start + per_day if i < 6 else len(items)
            day_items = items[start:end]
            lines.append(f"## {day}")
            for item in day_items:
                lines.append(f"- [{item.stage}] {item.title}" + (f" ({item.read_order})" if item.read_order else ""))
            lines.append("")

        return "\n".join(lines)
