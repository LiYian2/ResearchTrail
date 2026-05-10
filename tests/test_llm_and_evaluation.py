from agent.llm_client import LLMClient
from evaluation.metrics import collect_metrics
from shared.data_layer import SharedDataLayer
from shared.types import Paper, ResearchProfile, UserLevel
from skill_graph.analysis import GraphAnalysisSkill
from skill_graph.graph_builder import GraphBuilder
from skill_reading_path.path_generator import ReadingPathSkill
from skill_reading_path.report import ReportGenerator
from skill_retrieval.retrieval import LiteratureRetrievalSkill


def test_llm_extract_json_from_fenced_block():
    data = LLMClient._extract_json('```json\n{"intent": "build_reading_path"}\n```')
    assert data["intent"] == "build_reading_path"


def test_graph_builder_modes_export_metrics():
    data = SharedDataLayer()
    papers = [
        Paper("p1", "Graph Learning", ["A"], 2020, abstract="graph learning representation node edge"),
        Paper("p2", "Graph Neural Networks", ["B"], 2021, abstract="graph neural network representation edge node"),
        Paper("p3", "Vision Transformer", ["C"], 2022, abstract="vision transformer image attention model"),
    ]
    papers[1].references = ["p1"]
    data.add_papers(papers)
    graph = GraphBuilder(data).build(mode="citation")
    assert all(edge.type == "citation" for edge in graph.edges)
    metrics = data.get_metadata("graph_metrics")
    assert metrics["node_count"] == 3
    assert metrics["citation_edges"] == 1


def test_evaluation_metrics_smoke_demo_pipeline():
    data = SharedDataLayer()
    profile = ResearchProfile(topic="Vision Transformer", user_level=UserLevel.BEGINNER, max_papers=20)
    LiteratureRetrievalSkill(data).run(profile, demo=True)
    GraphBuilder(data).build()
    GraphAnalysisSkill(data).run()
    ReadingPathSkill(data).run()
    metrics = collect_metrics(data)
    assert metrics["retrieval"]["total_papers"] >= 20
    assert metrics["graph"]["node_count"] >= 20
    assert metrics["reading_path"]["unique_paper_count"] > 0


def test_vit_demo_injects_major_followup_landmarks():
    data = SharedDataLayer()
    profile = ResearchProfile(topic="Vision Transformer", user_level=UserLevel.BEGINNER, max_papers=20)
    papers = LiteratureRetrievalSkill(data).run(profile, demo=True)
    titles = {p.title for p in papers}
    assert "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" in titles
    assert "Training data-efficient image transformers and distillation through attention" in titles
    assert "Emerging Properties in Self-Supervised Vision Transformers" in titles
    assert "Masked Autoencoders Are Scalable Vision Learners" in titles


def test_vit_rule_query_plan_includes_followup_branches():
    data = SharedDataLayer()
    skill = LiteratureRetrievalSkill(data)
    queries = skill._generate_query_plan("Vision Transformer", UserLevel.BEGINNER)
    query_text = " ".join(queries).lower()
    assert "swin transformer" in query_text
    assert "dino" in query_text
    assert "masked autoencoders" in query_text


def test_cfr_rule_query_plan_includes_game_specific_terms():
    data = SharedDataLayer()
    skill = LiteratureRetrievalSkill(data)
    queries = skill._generate_query_plan("Counterfactual regret minimization", UserLevel.INTERMEDIATE)
    query_text = " ".join(queries).lower()
    assert "deep counterfactual regret minimization" in query_text
    assert "imperfect information games" in query_text
    assert "extensive-form games" in query_text


def test_report_includes_abstract_and_url_for_reading_items():
    data = SharedDataLayer()
    profile = ResearchProfile(topic="Vision Transformer", user_level=UserLevel.BEGINNER, max_papers=20)
    LiteratureRetrievalSkill(data).run(profile, demo=True)
    GraphBuilder(data).build()
    GraphAnalysisSkill(data).run()
    ReadingPathSkill(data).run()
    report = ReportGenerator(data).generate()
    assert "**Abstract:**" in report
    assert "**URL:**" in report
    assert "**Why read:**" in report


def test_cfr_topic_filter_removes_generic_regret_papers():
    data = SharedDataLayer()
    skill = LiteratureRetrievalSkill(data)
    papers = [
        Paper("p1", "The Involvement of the Orbitofrontal Cortex in the Experience of Regret", ["A"], 2004, abstract="This paper studies regret and counterfactual thinking in neuroscience."),
        Paper("p2", "Deep Counterfactual Regret Minimization", ["B"], 2018, abstract="Deep CFR solves imperfect-information games with counterfactual regret minimization."),
        Paper("p3", "Efficient Nash equilibrium approximation through Monte Carlo counterfactual regret minimization", ["C"], 2012, abstract="Monte Carlo CFR computes Nash equilibrium approximations in extensive-form games."),
        Paper("p4", "Regret Minimization in Games with Incomplete Information", ["D"], 2007, abstract="Counterfactual regret minimization is applied to poker and extensive games."),
        Paper("p5", "Solving Imperfect-Information Games via Discounted Regret Minimization", ["E"], 2019, abstract="Discounted regret minimization solves imperfect-information games."),
        Paper("p6", "Optimistic Regret Minimization for Extensive-Form Games", ["F"], 2022, abstract="An extensive-form game algorithm for regret minimization."),
    ]
    filtered = skill._filter_topic_specific_relevance(papers, "Counterfactual regret minimization")
    titles = {p.title for p in filtered}
    assert "The Involvement of the Orbitofrontal Cortex in the Experience of Regret" not in titles
    assert "Deep Counterfactual Regret Minimization" in titles


def test_cfr_must_read_deep_cfr_enters_reading_path():
    data = SharedDataLayer()
    profile = ResearchProfile(topic="Counterfactual regret minimization", user_level=UserLevel.INTERMEDIATE, max_papers=20)
    data.set_research_profile(profile)
    papers = [
        Paper("orig", "Regret Minimization in Games with Incomplete Information", ["A"], 2007, abstract="Counterfactual regret minimization in poker and extensive-form games.", citation_count=2500),
        Paper("mccfr", "Monte Carlo Sampling for Regret Minimization in Extensive Games", ["B"], 2009, abstract="Monte Carlo CFR for extensive games.", citation_count=1500),
        Paper("deep", "Deep Counterfactual Regret Minimization", ["C"], 2018, abstract="Deep CFR uses neural networks for imperfect-information games.", citation_count=1000),
        Paper("new1", "Recent CFR Variant", ["D"], 2024, abstract="Counterfactual regret minimization for imperfect-information games.", citation_count=10),
        Paper("new2", "Another CFR Variant", ["E"], 2025, abstract="Counterfactual regret minimization for poker.", citation_count=10),
        Paper("new3", "CFR Theory", ["F"], 2023, abstract="Extensive-form games and counterfactual regret.", citation_count=10),
    ]
    data.add_papers(papers)
    GraphBuilder(data).build()
    GraphAnalysisSkill(data).run()
    path = ReadingPathSkill(data).run()
    titles = [item.title for item in path.items]
    assert "Deep Counterfactual Regret Minimization" in titles


def test_title_based_citation_enrichment_for_arxiv_paper(monkeypatch):
    data = SharedDataLayer()
    skill = LiteratureRetrievalSkill(data)
    paper = Paper(
        "1811.00164v3",
        "Deep Counterfactual Regret Minimization",
        ["Noam Brown"],
        2018,
        abstract="Deep CFR uses neural networks for imperfect-information games.",
        url="https://arxiv.org/abs/1811.00164v3",
        citation_count=0,
        source="arxiv",
    )

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {
                        "id": "https://openalex.org/W123",
                        "title": "Deep Counterfactual Regret Minimization",
                        "cited_by_count": 357,
                        "referenced_works": ["https://openalex.org/W1"],
                    }
                ]
            }

    def fake_get(*args, **kwargs):
        return DummyResponse()

    import skill_retrieval.retrieval as retrieval_module

    monkeypatch.setattr(retrieval_module.requests, "get", fake_get)
    skill._enrich_citations([paper], 10)
    assert paper.citation_count == 357
    assert paper.references == ["W1"]
