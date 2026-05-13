---
name: reading-path-report
description: "Generate personalized research reading paths and evidence-grounded reports from graph analysis results."
author: researchtrail-team
version: 1.0.0
tags:
  - social-network-analysis
  - reading-path
  - report-generation
  - evidence-grounded-explanation
  - visualization
---

# Reading Path and Report Skill

## When to Use
Use this Skill after graph analysis has produced paper scores and communities, when the user needs an actionable reading path, report, visualization, or follow-up explanation.

## Inputs
- User profile from the shared data layer.
- Paper corpus from the shared data layer.
- Graph scores and community labels from the Research Graph Analysis Skill.
- Optional LLM explanation mode: `llm`, `rule`, or `off`.

## Procedure
1. Select prerequisite, foundation, core, bridge, and frontier papers by deterministic graph scores.
2. Arrange papers into stages according to user level and learning goal.
3. Create evidence packets for each recommended paper.
4. Optionally ask the LLM to turn evidence packets into concise, grounded explanations.
5. Generate markdown report, graph visualization, score distribution, follow-up suggestions, and evaluation metrics.

## Outputs
- Personalized reading path with stages and paper reasons.
- Evidence packets for recommended papers.
- Markdown report.
- Network and score visualizations.
- Follow-up answers such as bridge-paper explanations and 7-day reading plans.

## Evaluation Protocol
- Compare network-aware reading paths against a citation-count baseline.
- Record stage count, unique paper count, role counts, explanation coverage, landmark hit rate, topic precision, ordering quality, community coverage, and stage coverage.
- For human evaluation, rate coherence, coverage, level appropriateness, and explanation usefulness from 1 to 5.
- Verify that each recommended paper includes authors, year, source URL when available, abstract, and an evidence-grounded why-read explanation.

## Current Evaluation Artifacts
- `outputs/evaluation/skill_reading_path_report.md`
- `outputs/reading_path_ablation/full_11_topics/reading_path_ablation_summary.md`
- `outputs/reading_path_ablation/full_11_topics/reading_path_ablation_results.json`
- `outputs/reading_path_ablation/full_11_topics/paths_for_human_eval/`
- `outputs/report_materials/final_evaluation_and_implementation_summary.md`

## Known Limitations
- Why-read explanations are only as reliable as the retrieved metadata and abstract.
- Citation-count baselines use labeled citation sources: Semantic Scholar, OpenAlex, or curated metadata. Counts may differ from Google Scholar.
- Topic-specific landmark support is transparent in source mix and quality metrics. It should be evaluated with ablations when strict generalization is the main claim.

## Failure Handling
- If LLM explanation fails, use deterministic template explanations.
- If no graph scores exist, ask the Agent to run graph analysis first.
- If visualization fails, still return the report and reading path.

## Do Not
- Do not retrieve papers.
- Do not recompute graph centrality.
- Do not invent facts not present in the evidence packet.
