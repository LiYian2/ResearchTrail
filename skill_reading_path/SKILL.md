---
name: reading-path-report
description: "Generate personalized research reading paths, evidence-grounded explanations, Markdown reports, and visualizations from graph analysis results."
author: LiYian2
version: 1.0.0
tags:
  - reading-path
  - report-generation
  - evidence-grounded-explanation
  - visualization
  - research-agent
metadata:
  openclaw:
    requires:
      env:
        - SILICON_FLOW_API
      bins:
        - python
---

# Reading Path and Report Skill

You are helping the ResearchTrail agent turn graph analysis results into an actionable reading path and human-readable report.

## When to Use

Use this skill after papers have been retrieved and graph scores have been computed, especially when the user asks for:

- a reading list or reading path
- a 7-day study plan
- bridge paper explanations
- field map or community summary
- a Markdown report with figures
- follow-up explanations grounded in retrieved evidence

Do not use this skill to retrieve papers or recompute graph centrality.

## How to Run

Preferred full-agent command:

```bash
python main.py "<user research goal>" --llm auto --max-papers 45 --output-dir outputs/<run_name>
```

Reading-path ablation from saved full-agent states:

```bash
python -m evaluation.reading_path_ablation --state-root outputs/benchmark_full_llm_normalized --output-dir outputs/reading_path_ablation/full_11_topics
```

GUI demo:

```bash
python app.py
```

## Inputs

- User profile.
- Paper corpus.
- Graph scores and community labels.
- Optional LLM explanation writer.

## Procedure

1. Select prerequisite, foundation, core, bridge, and frontier papers using graph scores and user level.
2. Arrange papers into staged reading order.
3. Create evidence packets for each paper.
4. If LLM is available, generate concise why-read explanations from evidence only.
5. Generate `research_report.md`, `research_graph.png`, `scores_distribution.png`, and `state.json`.
6. Add follow-up suggestions such as bridge explanations and study-plan requests.

## Outputs

- Staged reading path.
- Paper-level abstract, URL, role scores, and why-read explanation.
- Markdown report.
- Network and score visualizations.
- Follow-up suggestions.

## Error Handling

- If LLM explanation fails, use deterministic template explanations.
- If graph scores are missing, ask the agent to run Research Graph Analysis first.
- If visualization fails, still return the Markdown report and state file.
- If abstracts are missing, still include URL/title/authors/year and explain that metadata was incomplete.

## Evaluation Evidence

See `docs/evaluation.md`. The reading-path ablation compares random order, citation-count order, PageRank order, and ResearchTrail staged paths.

## Do Not

- Do not invent claims that are absent from the evidence packet.
- Do not cite papers that are not in the corpus.
- Do not overwrite graph scores.
