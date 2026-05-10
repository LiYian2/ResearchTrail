import argparse
import json
import os

from agent.planner import AgentPlanner
from evaluation.metrics import collect_metrics, markdown_table
from evaluation.reporting import write_evaluation_reports
from shared.types import ResearchProfile, UserLevel
from skill_graph.analysis import GraphAnalysisSkill
from skill_graph.community_labeler import CommunityLabeler
from skill_graph.graph_builder import GraphBuilder
from skill_reading_path.path_generator import ReadingPathSkill
from skill_retrieval.retrieval import LiteratureRetrievalSkill


def run_agent_case(topic: str, output_dir: str, llm: str, demo: bool, max_papers: int, user_level: str) -> dict:
    agent = AgentPlanner(llm_mode=llm, output_dir=output_dir, max_papers_override=max_papers)
    agent.demo = demo
    if user_level == "beginner":
        query = f"I am a beginner and want to deeply understand {topic}"
    elif user_level == "advanced":
        query = f"I am an advanced student and want to understand recent advances and open problems in {topic}"
    else:
        query = f"I am an intermediate student and want to understand {topic}"
    agent.process(query)
    return collect_metrics(agent.data)


def run_graph_ablation(topic: str, mode: str, demo: bool, max_papers: int, user_level: str) -> dict:
    from shared.data_layer import SharedDataLayer

    level = {
        "beginner": UserLevel.BEGINNER,
        "intermediate": UserLevel.INTERMEDIATE,
        "advanced": UserLevel.ADVANCED,
    }.get(user_level, UserLevel.INTERMEDIATE)
    data = SharedDataLayer()
    profile = ResearchProfile(
        topic=topic,
        user_level=level,
        max_papers=max_papers,
        preferred_length=10,
        goal="deep_understanding",
    )
    retrieval = LiteratureRetrievalSkill(data)
    retrieval.run(profile, demo=demo)
    builder = GraphBuilder(data)
    builder.build(mode=mode)
    analysis = GraphAnalysisSkill(data)
    analysis.run()
    CommunityLabeler(data).run()
    ReadingPathSkill(data).run()
    return collect_metrics(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResearchTrail evaluation cases.")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--max-papers", type=int, default=80)
    parser.add_argument("--user-level", choices=["beginner", "intermediate", "advanced"], default="intermediate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "rule_only_agent": run_agent_case(
            args.topic,
            os.path.join(args.output_dir, "rule_only_agent"),
            llm="off",
            demo=args.demo,
            max_papers=args.max_papers,
            user_level=args.user_level,
        ),
        "llm_assisted_agent": run_agent_case(
            args.topic,
            os.path.join(args.output_dir, "llm_assisted_agent"),
            llm="auto",
            demo=args.demo,
            max_papers=args.max_papers,
            user_level=args.user_level,
        ),
        "citation_graph": run_graph_ablation(args.topic, "citation", args.demo, args.max_papers, args.user_level),
        "similarity_graph": run_graph_ablation(args.topic, "similarity", args.demo, args.max_papers, args.user_level),
        "hybrid_graph": run_graph_ablation(args.topic, "hybrid", args.demo, args.max_papers, args.user_level),
    }

    json_path = os.path.join(args.output_dir, "evaluation_results.json")
    md_path = os.path.join(args.output_dir, "evaluation_summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# ResearchTrail Evaluation: {args.topic}\n\n")
        f.write(markdown_table(results))
        f.write("\n")
    write_evaluation_reports(args.topic, results, args.output_dir)

    print(f"Evaluation saved to {json_path}")
    print(f"Summary saved to {md_path}")
    print(f"Skill reports saved under {args.output_dir}")


if __name__ == "__main__":
    main()
