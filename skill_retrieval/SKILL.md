---
name: literature-retrieval
description: "Retrieve, enrich, deduplicate, filter, and quality-check academic papers for a user-specified research topic. Use when the agent needs a graph-ready paper corpus."
author: LiYian2
version: 1.0.0
tags:
  - literature-retrieval
  - arxiv
  - openalex
  - semantic-scholar
  - social-network-analysis
metadata:
  openclaw:
    requires:
      env:
        - S2_API_KEY
      bins:
        - python
---

# Literature Retrieval Skill

You are helping the ResearchTrail agent build a reliable paper corpus for downstream citation and similarity graph analysis.

## When to Use

Use this skill when the user asks to:

- enter, learn, survey, or explore a research field
- build a reading path for a topic
- find foundational papers or recent developments
- prepare a paper corpus before graph/network analysis

Do not use this skill for graph centrality, community detection, or final report writing.

## How to Run

Preferred full-agent invocation from the repository root:

```bash
python main.py "<user research goal>" --llm auto --llm-provider siliconflow --llm-model Pro/zai-org/GLM-4.7 --max-papers 45 --output-dir outputs/<run_name>
```

Skill-specific corpus ablation/evaluation:

```bash
python -m evaluation.corpus_ablation --topic rag --max-papers 30 --output-dir outputs/corpus_ablation/rag
```

Multi-topic corpus ablation:

```bash
python -m evaluation.corpus_ablation_batch --max-papers 45 --output-dir outputs/corpus_ablation/full_12_topics
```

Offline fallback demo:

```bash
python main.py "I am a beginner and want to understand Vision Transformer" --demo --llm off --max-papers 40 --output-dir outputs/demo_vit
```

## Inputs

- `topic`: research topic string.
- `user_level`: `beginner`, `intermediate`, or `advanced`.
- `max_papers`: maximum paper count.
- Optional `query_plan`: structured query plan with `main_queries`, `prerequisite_queries`, `exclude_terms`, `expected_communities`, `positive_terms`, `alias_queries`, and verified landmark candidates.

## Procedure

1. Normalize or generate a query plan.
2. Search arXiv and OpenAlex.
3. Deduplicate papers by normalized title.
4. Filter by topic relevance, aliases, exact titles, expected communities, and exclude terms.
5. Recover verified landmarks only when API metadata confirms the paper.
6. Enrich citation counts and references using Semantic Scholar when `S2_API_KEY` is available; fall back to OpenAlex title matching.
7. Save paper records and corpus quality metrics to the shared data layer.

## Outputs

- Paper records with title, authors, year, abstract, URL, citation count, citation source, source label, references, and citations.
- Corpus quality metrics: total papers, abstract coverage, citation metadata coverage, reference coverage, deduplication count, year range.
- Query metadata for reproducibility and evaluation.

## Error Handling

- If `S2_API_KEY` is missing, continue with OpenAlex and arXiv metadata; report weaker citation/reference coverage.
- If arXiv returns rate-limit errors, wait and retry; if it still fails, continue with OpenAlex or demo mode.
- If OpenAlex fails, keep arXiv results and mark metadata coverage as partial.
- If fewer than 8 papers are found, broaden the query or ask the agent to rerun with a more general topic.

## Evaluation Evidence

See:

- `docs/evaluation.md`
- `outputs_submission/sample_research_report.md`

The main corpus ablation compares arXiv-only, OpenAlex-only, combined retrieval, topic filtering, and verified landmark recovery.
