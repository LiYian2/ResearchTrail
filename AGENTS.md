---
name: researchtrail
description: "A multi-skill citation-network research navigation agent. Use when a user wants to enter, survey, or understand a research field through paper retrieval, graph analysis, and personalized reading paths."
author: LiYian2
version: 1.0.0
tags:
  - research-agent
  - social-network-analysis
  - citation-network
  - reading-path
  - literature-retrieval
metadata:
  openclaw:
    requires:
      env:
        - SILICON_FLOW_API
        - S2_API_KEY
      bins:
        - python
    primaryEnv: SILICON_FLOW_API
---

# ResearchTrail Agent

You are operating ResearchTrail, a code-backed, LLM-assisted research navigation agent.

Use this agent when the user asks to learn, enter, survey, map, compare, or build a reading path for a research field.

## Child Skills

Call the skills in this order for a full workflow:

1. `skill_retrieval`: build a paper corpus.
2. `skill_graph`: construct citation/similarity networks and compute paper roles.
3. `skill_reading_path`: generate a staged reading path, report, and figures.

## Normal Command

From the repository root:

```bash
python main.py "<user research goal>" --llm auto --llm-provider siliconflow --llm-model Pro/zai-org/GLM-4.7 --max-papers 45 --output-dir outputs/<run_name>
```

Stable offline demo:

```bash
python main.py "I am a beginner and want to understand Vision Transformer" --demo --llm off --max-papers 40 --output-dir outputs/demo_vit
```

GUI:

```bash
python app.py
```

Optional graph similarity backend:

```bash
python main.py "<user research goal>" --similarity-backend lsa --max-papers 45 --output-dir outputs/<run_name>
```

Use `tfidf` for the most lightweight deterministic backend and `lsa` for an offline embedding-style semantic backend. The code also accepts `sentence-transformer`, but that backend requires the optional package/model and falls back when unavailable.

## Required Outputs

After a successful full run, check that the output directory contains:

- `research_report.md`
- `research_graph.png`
- `scores_distribution.png`
- `state.json`

## Error Handling

- If `SILICON_FLOW_API` is missing or LLM calls fail, rerun with `--llm off` or `--demo --llm off`.
- If arXiv/OpenAlex/Semantic Scholar fail, use demo mode for the presentation and report the live API failure.
- If graph construction yields too few citation edges, prefer hybrid graph mode because it adds semantic similarity edges.
- If the user wants a more semantic graph, use `--similarity-backend lsa`; keep TF-IDF when reproducibility and lexical precision matter more.
- If report generation succeeds but visualization fails, return the Markdown report and state file, then mention the visualization failure.

## Personalization Layer

If `MEMORY.md` exists and an external agent is polishing a generated report, read it before rewriting narrative text. Use it only to adjust wording, explanation depth, examples, and study-plan framing. Preserve factual sections from `research_report.md` and `state.json`, including paper metadata, URLs, abstracts, citation source labels, graph scores, communities, and limitations.

For detailed report-polishing rules, see `AGENT.md`.

## Do Not

- Do not let the LLM invent papers or citation counts.
- Do not use LLM output as final evidence unless it is grounded in retrieved paper metadata.
- Do not submit generated `outputs/` directories as source code; keep sample artifacts in `outputs_submission/`.
- Do not remove abstracts, URLs, or citation source labels when personalizing reports.
