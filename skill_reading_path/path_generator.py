from typing import Optional
import re

from shared.data_layer import SharedDataLayer
from shared.types import Paper, NodeScores, ReadingPath, ReadingPathItem, ResearchProfile, UserLevel
from shared.types import IntentType
from skill_reading_path.explanation_writer import ExplanationWriter


class ReadingPathSkill:
    def __init__(self, data_layer: SharedDataLayer, explanation_writer: Optional[ExplanationWriter] = None):
        self.data = data_layer
        self.explanation_writer = explanation_writer or ExplanationWriter()

    def run(self) -> ReadingPath:
        papers = self.data.get_all_papers()
        scores = self.data.get_all_scores()
        profile = self.data.get_research_profile()

        if not papers or not scores:
            raise ValueError("Papers or scores not available. Run retrieval and graph analysis first.")

        is_beginner = profile and profile.user_level == UserLevel.BEGINNER

        # Classify papers into stages
        foundation = self._select_by_score_type(scores, "foundation_score", 3)
        core = self._select_by_score_type(scores, "pagerank", 3)
        bridges = self._select_by_score_type(scores, "bridge_score", 2)
        frontier = self._select_by_score_type(scores, "frontier_score", 3)
        prerequisites = self._select_prerequisites(papers, scores, profile) if is_beginner else []
        key_developments = self._select_key_developments(papers, scores, profile, 6)
        self._apply_must_read_overrides(papers, profile, foundation, core, key_developments)

        # Remove duplicates across stages via paper_id set
        used_ids: set[str] = set()
        stages = self._build_stages(foundation, core, key_developments, bridges, frontier, prerequisites, papers, scores, used_ids, profile)
        evidence_packets = self._build_evidence_packets(stages, scores)
        llm_reasons = self.explanation_writer.generate(evidence_packets)
        if llm_reasons:
            self._apply_llm_reasons(stages, llm_reasons)
            for packet in evidence_packets:
                if packet["paper_id"] in llm_reasons:
                    packet["llm_reason"] = llm_reasons[packet["paper_id"]]
        self.data.set_metadata("evidence_packets", evidence_packets)
        self.data.set_metadata("reading_path_metrics", self._compute_path_metrics(stages, evidence_packets))

        followup = self._generate_followup_suggestions(profile)
        report = self._generate_report_markdown(stages, profile)

        all_items: list[ReadingPathItem] = []
        for stage in stages:
            for p in stage["papers"]:
                all_items.append(ReadingPathItem(
                    paper_id=p["paper_id"],
                    title=p["title"],
                    reason=p["reason"],
                    read_order=p["read_order"],
                    stage=stage["stage"],
                ))

        path = ReadingPath(
            stages=stages,
            items=all_items,
            report_markdown=report,
            followup_suggestions=followup,
        )
        self.data.set_reading_path(path)
        return path

    def _build_evidence_packets(self, stages: list[dict], scores: dict[str, NodeScores]) -> list[dict]:
        community_labels = self.data.get_metadata("community_labels") or {}
        packets = []
        for stage in stages:
            for paper_info in stage["papers"]:
                pid = paper_info["paper_id"]
                paper = self.data.get_paper(pid)
                score = scores.get(pid)
                if not paper or not score:
                    continue
                role = self._classify_stage_role(stage["stage"])
                comm_label = community_labels.get(score.community, {})
                packets.append({
                    "paper_id": pid,
                    "title": paper.title,
                    "abstract": self._shorten_text(paper.abstract, 900),
                    "url": paper.url,
                    "stage": stage["stage"],
                    "role": role,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                    "source": paper.source,
                    "pagerank": score.pagerank,
                    "betweenness": score.betweenness,
                    "foundation_score": score.foundation_score,
                    "bridge_score": score.bridge_score,
                    "frontier_score": score.frontier_score,
                    "community_id": score.community,
                    "community_label": comm_label.get("label", f"Community {score.community}"),
                    "template_reason": paper_info["reason"],
                })
        return packets

    @staticmethod
    def _shorten_text(text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _apply_llm_reasons(stages: list[dict], reasons: dict[str, str]) -> None:
        for stage in stages:
            for paper_info in stage["papers"]:
                reason = reasons.get(paper_info["paper_id"])
                if reason:
                    paper_info["reason"] = reason

    @staticmethod
    def _classify_stage_role(stage: str) -> str:
        lower = stage.lower()
        if "prereq" in lower:
            return "prerequisite"
        if "found" in lower:
            return "foundation"
        if "bridge" in lower or "gap" in lower:
            return "bridge"
        if "development" in lower:
            return "development"
        if "frontier" in lower or "recent" in lower or "state of the art" in lower:
            return "frontier"
        return "core"

    def _compute_path_metrics(self, stages: list[dict], evidence_packets: list[dict]) -> dict:
        roles = {}
        for packet in evidence_packets:
            roles[packet["role"]] = roles.get(packet["role"], 0) + 1
        total = sum(len(stage["papers"]) for stage in stages)
        with_reason = sum(1 for stage in stages for p in stage["papers"] if p.get("reason"))
        return {
            "stage_count": len(stages),
            "unique_paper_count": total,
            "role_counts": roles,
            "explanation_availability_ratio": with_reason / max(total, 1),
        }

    def _select_prerequisites(
        self, papers: list[Paper], scores: dict[str, NodeScores], profile
    ) -> list[str]:
        topic = profile.topic.lower() if profile else ""
        if ("vision" in topic or "image" in topic or "vit" in topic) and "transformer" in topic:
            prereq_titles = [
                "Attention Is All You Need",
                "Deep Residual Learning for Image Recognition",
                "ImageNet Classification with Deep Convolutional Neural Networks",
            ]
            selected = self._matching_titles_in_order(papers, prereq_titles)
            if selected:
                return selected[:3]

        STOP_WORDS = {
            "want", "learn", "study", "enter", "field", "would", "like",
            "with", "from", "that", "this", "have", "been", "their",
            "about", "into", "over", "after", "into", "more", "some",
        }
        topic_words = {w for w in re.findall(r"[a-z]{4,}", topic) if w not in STOP_WORDS}

        scored: list[tuple[str, float]] = []
        for p in papers:
            p_text = (p.title + " " + p.abstract).lower()
            topic_matches = sum(1 for w in topic_words if w in p_text)
            if topic_matches >= 2:
                continue  # Already well-covered by main topic results
            sc = scores.get(p.paper_id)
            if not sc:
                continue
            prereq_score = (p.citation_count / max(1, 2026 - max(p.year, 1990))) * (0.5 + sc.pagerank * 3) + sc.bridge_score * 2
            scored.append((p.paper_id, prereq_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored[:3]]

    @staticmethod
    def _matching_titles_in_order(papers: list[Paper], title_patterns: list[str]) -> list[str]:
        selected = []
        lowered_patterns = [p.lower() for p in title_patterns]
        for pattern in lowered_patterns:
            for paper in papers:
                if pattern in paper.title.lower() and paper.paper_id not in selected:
                    selected.append(paper.paper_id)
                    break
        return selected

    def _select_key_developments(
        self,
        papers: list[Paper],
        scores: dict[str, NodeScores],
        profile: Optional[ResearchProfile],
        top_n: int,
    ) -> list[str]:
        topic = (profile.topic if profile else "").lower()
        topic_words = {w for w in re.findall(r"[a-z]{3,}", topic) if w not in {"the", "and", "for", "with"}}
        max_citations = max((p.citation_count for p in papers), default=1)
        candidates: list[tuple[str, float]] = []
        for p in papers:
            if p.year < 2020:
                continue
            text = f"{p.title} {p.abstract}".lower()
            if topic_words and not any(word in text for word in topic_words):
                continue
            sc = scores.get(p.paper_id)
            if not sc:
                continue
            citation_factor = p.citation_count / max(max_citations, 1)
            source_bonus = 0.25 if p.source in {"landmark_demo", "curated_landmark"} else 0.0
            lower_title = p.title.lower()
            vit_followup_bonus = 0.0
            if p.source in {"landmark_demo", "curated_landmark"} and any(
                key in lower_title
                for key in [
                    "swin transformer",
                    "data-efficient image transformers",
                    "self-supervised vision transformers",
                    "masked autoencoders",
                    "image is worth 16x16",
                ]
            ):
                vit_followup_bonus = 0.2
            development_score = (
                source_bonus
                + vit_followup_bonus
                + 0.35 * citation_factor
                + 0.25 * sc.pagerank
                + 0.25 * sc.bridge_score
                + 0.15 * sc.frontier_score
            )
            candidates.append((p.paper_id, development_score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in candidates[:top_n]]

    def _apply_must_read_overrides(
        self,
        papers: list[Paper],
        profile: Optional[ResearchProfile],
        foundation_ids: list[str],
        core_ids: list[str],
        key_development_ids: list[str],
    ) -> None:
        topic = (profile.topic if profile else "").lower()
        if "counterfactual" in topic and ("regret" in topic or "cfr" in topic):
            self._prepend_matching_title(
                papers,
                foundation_ids,
                ["Regret Minimization in Games with Incomplete Information"],
                max_len=3,
            )
            self._prepend_matching_title(
                papers,
                core_ids,
                ["Monte Carlo Sampling for Regret Minimization in Extensive Games"],
                max_len=3,
            )
            self._prepend_matching_title(
                papers,
                key_development_ids,
                ["Deep Counterfactual Regret Minimization"],
                max_len=6,
            )
        if ("vision" in topic or "image" in topic or "vit" in topic) and "transformer" in topic:
            self._prepend_matching_title(
                papers,
                foundation_ids,
                ["An Image is Worth 16x16 Words"],
                max_len=3,
            )
            self._prepend_matching_title(
                papers,
                core_ids,
                ["Training data-efficient image transformers", "Swin Transformer"],
                max_len=3,
            )
            for title in [
                "Masked Autoencoders Are Scalable Vision Learners",
                "Emerging Properties in Self-Supervised Vision Transformers",
                "BEiT: BERT Pre-Training of Image Transformers",
            ]:
                self._prepend_matching_title(papers, key_development_ids, [title], max_len=6)

    @staticmethod
    def _prepend_matching_title(
        papers: list[Paper],
        target_ids: list[str],
        title_patterns: list[str],
        max_len: int,
    ) -> None:
        lowered_patterns = [p.lower() for p in title_patterns]
        for paper in papers:
            title = paper.title.lower()
            if any(pattern in title for pattern in lowered_patterns):
                if paper.paper_id in target_ids:
                    target_ids.remove(paper.paper_id)
                target_ids.insert(0, paper.paper_id)
                del target_ids[max_len:]
                return

    def _select_by_score_type(
        self, scores: dict[str, NodeScores], score_field: str, top_n: int
    ) -> list[str]:
        scored = [(pid, getattr(s, score_field, 0.0)) for pid, s in scores.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in scored[:top_n]]

    def _build_stages(
        self,
        foundation_ids: list[str],
        core_ids: list[str],
        key_development_ids: list[str],
        bridge_ids: list[str],
        frontier_ids: list[str],
        prereq_ids: list[str],
        papers: list[Paper],
        scores: dict[str, NodeScores],
        used_ids: set[str],
        profile: Optional[ResearchProfile],
    ) -> list[dict]:
        paper_map = {p.paper_id: p for p in papers}
        score_map = scores

        is_beginner = profile and profile.user_level == UserLevel.BEGINNER

        stage_defs: list[tuple[str, list[str], str]] = []

        if is_beginner and prereq_ids:
            stage_defs.append((
                "Prerequisites",
                prereq_ids,
                "Background knowledge needed before diving into the specific topic.",
            ))

        if is_beginner:
            stage_defs += [
                ("Conceptual Foundations", foundation_ids, "Explains the fundamental ideas and terminology."),
                ("Core Methods", core_ids, "Introduces the primary methods and techniques."),
                ("Key Developments", key_development_ids, "Covers the major follow-up branches that shaped the field after the core method."),
                ("Bridge Papers", bridge_ids, "Links different ideas together for a broader understanding."),
                ("Recent Advances", frontier_ids, "Shows where the field is heading."),
            ]
        elif profile and profile.user_level == UserLevel.ADVANCED:
            stage_defs = [
                ("Foundations", foundation_ids, "Key prior work that established the field."),
                ("Core Methods", core_ids, "Central algorithmic contributions."),
                ("Key Developments", key_development_ids, "Major follow-up work that reshaped the field."),
                ("Research Gaps", bridge_ids, "Papers highlighting open problems and cross-field connections."),
                ("State of the Art", frontier_ids, "Most recent and impactful work."),
            ]
        else:
            stage_defs = [
                ("Foundations", foundation_ids, "Introduces foundational concepts for this research area."),
                ("Core Algorithms", core_ids, "Presents key algorithmic contributions and methods."),
                ("Key Developments", key_development_ids, "Major follow-up work after the core foundations."),
                ("Bridge Papers", bridge_ids, "Connects different research communities or subfields."),
                ("Recent Frontier", frontier_ids, "Represents the latest advances and open problems."),
            ]

        stages: list[dict] = []
        order = 0
        for stage_name, ids, default_reason in stage_defs:
            stage_papers: list[dict] = []
            for pid in ids:
                if pid in used_ids:
                    continue
                paper = paper_map.get(pid)
                if not paper:
                    continue
                used_ids.add(pid)
                order += 1
                sc = score_map.get(pid)
                reason = self._generate_paper_reason(paper, sc, stage_name)
                stage_papers.append({
                    "paper_id": pid,
                    "title": paper.title,
                    "reason": reason or default_reason,
                    "read_order": order,
                    "year": paper.year,
                    "citation_count": paper.citation_count,
                })

            if stage_papers:
                stages.append({"stage": stage_name, "papers": stage_papers})

        return stages

    def _generate_paper_reason(
        self, paper: Paper, scores: Optional[NodeScores], stage: str
    ) -> str:
        if not scores:
            return ""

        if "Found" in stage:
            if scores.pagerank > 0.05:
                return f"Read this early because it is structurally central in the graph and heavily cited ({paper.citation_count} citations), making it a foundation for later work."
            hint = self._topic_hint(paper)
            if hint == "the field":
                return "Read this early to establish the core vocabulary and assumptions before moving to later specialized papers."
            return f"Read this early to establish the core vocabulary and assumptions before moving to later papers on {hint}."
        elif "Core" in stage:
            return f"Read this as a core method paper because it has high graph centrality (PageRank {scores.pagerank:.3f}) and anchors later developments in the path."
        elif "Key Development" in stage:
            hint = self._topic_hint(paper)
            branch = hint if hint != "the field" else f"{paper.title.split(':', 1)[0].lower()} research"
            return f"Read this after the foundation papers because it represents a major follow-up branch ({branch}) with strong impact ({paper.citation_count} citations) and bridge score {scores.bridge_score:.3f}."
        elif "Bridge" in stage:
            if scores.bridge_score > 0.5:
                return f"Read this to understand how separate research communities connect; its high betweenness ({scores.betweenness:.3f}) marks it as a bridge paper."
            return f"Read this for cross-community context because it links {self._topic_hint(paper)} with neighboring directions in the graph."
        elif "Frontier" in stage or "Recent" in stage:
            return f"Read this near the end to see recent directions from {paper.year}; its frontier score ({scores.frontier_score:.3f}) indicates current research momentum."
        return ""

    @staticmethod
    def _topic_hint(paper: Paper) -> str:
        text = f"{paper.title} {paper.abstract}".lower()
        hints = [
            ("self-supervised", "self-supervised learning"),
            ("masked", "masked pretraining"),
            ("hierarchical", "hierarchical backbones"),
            ("window", "window-based attention"),
            ("distillation", "data-efficient training"),
            ("language", "vision-language learning"),
            ("dense prediction", "dense prediction"),
            ("attention", "attention mechanisms"),
        ]
        for key, label in hints:
            if key in text:
                return label
        return "the field"

    def _generate_report_markdown(self, stages: list[dict], profile: Optional[ResearchProfile]) -> str:
        lines = []
        topic = profile.topic if profile else "your research topic"
        level = profile.user_level.value if profile else "beginner"

        lines.append(f"# ResearchTrail Reading Path\n")
        lines.append(f"**Topic:** {topic}\n")
        lines.append(f"**Level:** {level}\n")
        lines.append(f"**Generated:** {self._timestamp()}\n")
        lines.append("---\n")

        for stage in stages:
            lines.append(f"## {stage['stage']}\n")
            for p in stage["papers"]:
                lines.append(f"### {p['read_order']}. {p['title']} ({p.get('year', 'N/A')})")
                lines.append(f"*{p['reason']}*")
                if "citation_count" in p and p["citation_count"] > 0:
                    lines.append(f"Citations: {p['citation_count']}")
                lines.append("")

        lines.append("---\n")
        total = sum(len(s["papers"]) for s in stages)
        lines.append(f"*Total: {total} papers in {len(stages)} stages.*\n")

        return "\n".join(lines)

    def _generate_followup_suggestions(self, profile: Optional[ResearchProfile]) -> list[str]:
        suggestions = [
            "Show bridge papers between key communities",
            "Generate a 7-day reading plan",
            "Show only papers after 2020",
            "Which authors should I follow?",
            "Replace theoretical papers with more applied papers",
            "Explain why these papers are foundational",
            "Show me the citation network visualization",
        ]
        if profile and profile.user_level == UserLevel.BEGINNER:
            suggestions.insert(0, "Add more survey and tutorial papers")
        if profile and profile.user_level == UserLevel.ADVANCED:
            suggestions.insert(0, "Show open problems and research gaps")
        return suggestions

    def filter_by_year(self, min_year: int) -> ReadingPath:
        papers = self.data.get_all_papers()
        filtered = [p for p in papers if p.year >= min_year]
        old_papers = {p.paper_id for p in self.data.get_all_papers()}
        self.data.clear_papers()
        self.data.add_papers(filtered)
        path = self.run()
        return path

    def regenerate_path(self) -> ReadingPath:
        return self.run()

    @staticmethod
    def _timestamp() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
