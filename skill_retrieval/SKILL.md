---
name: literature-retrieval
description: "Retrieve, enrich, and quality-check academic papers for a user-specified research topic."
author: researchtrail-team
version: 1.0.0
tags:
  - social-network-analysis
  - literature-retrieval
  - arxiv
  - openalex
  - semantic-scholar
  - query-expansion
---

# Literature Retrieval Skill

## When to Use
Use this Skill when the user wants to learn, enter, survey, or explore a research field and the Agent needs a structured paper corpus for downstream network analysis.

## Inputs
- `topic`: research topic string.
- `user_level`: `beginner`, `intermediate`, or `advanced`.
- `goal`: learning goal, such as `deep_understanding` or `survey`.
- `time_range`: `all`, `last_year`, `last_3_years`, or `last_5_years`.
- `max_papers`: maximum corpus size.
- `query_plan` optional: structured LLM-generated plan with `main_queries`, `prerequisite_queries`, `exclude_terms`, and `expected_communities`.

## Procedure
1. Build or receive a structured query plan.
2. Search arXiv and OpenAlex deterministically.
3. Deduplicate records by normalized title and preserve richer metadata.
4. Filter by topic relevance, time range, and optional exclude terms. Topic filtering uses exact landmark titles, LLM/rule query aliases, positive terms, expected communities, and acronym-aware matching instead of broad single-keyword matching.
5. Enrich records with citation counts and references when available. Prefer Semantic Scholar `citationCount` when `S2_API_KEY` is configured, respecting the 1 request/second API limit; otherwise fall back to OpenAlex and curated landmark metadata.
6. Save papers and corpus quality metrics to the shared data layer.

## Outputs
- Paper corpus in shared data layer.
- Corpus quality metrics: paper count, abstract coverage, citation/reference coverage, year range, and deduplication count.
- Query metadata for evaluation and report generation.

## Evaluation Protocol
- Run live retrieval on at least two topics: `Vision Transformer` and `Counterfactual regret minimization`.
- Compare `rule_only_agent` and `llm_assisted_agent`.
- Record paper count, abstract coverage, citation/reference coverage, year range, deduplication count, and number of main queries.
- Record topic precision and source mix so retrieved API papers, curated landmarks, and synthetic demo records are distinguishable.
- Run the corpus construction ablation with `python -m evaluation.corpus_ablation --topic <benchmark_id_or_topic> --max-papers 45 --output-dir outputs/corpus_ablation/<topic>`.
- Corpus ablation variants: arXiv only, OpenAlex only, arXiv + OpenAlex, plus topic filtering, plus verified landmark recovery.
- Corpus ablation metrics: total papers, duplicate removal, abstract coverage, citation metadata coverage, reference coverage, landmark hit rate, topic precision, graph edges, and graph edge yield.
- Inspect top retrieved papers for topical relevance, especially for ambiguous terms such as `regret` and `counterfactual`.
- Treat demo mode as a pipeline reliability check only; report-quality experiments must use live arXiv/OpenAlex/Semantic Scholar mode.

## Current Evaluation Artifacts
- `outputs/evaluation/skill_retrieval_report.md`
- `outputs/corpus_ablation/full_12_topics/corpus_ablation_batch_summary.md`
- `outputs/corpus_ablation/full_12_topics/corpus_ablation_batch_results.json`
- `outputs/report_materials/final_evaluation_and_implementation_summary.md`

## Known Limitations
- Citation counts prefer Semantic Scholar when `S2_API_KEY` is available, then OpenAlex or curated metadata. Counts may still differ from Google Scholar or Google Search snippets.
- arXiv can rate-limit live experiments; OpenAlex usually remains available but reference coverage may vary by topic.
- LLM query expansion improves flexibility but needs deterministic relevance filters to avoid off-topic retrieval.
- Curated landmarks improve recall for known foundational papers. Evaluation reports expose them in source mix; use a no-curated rerun for strict generalization ablation if required.

## Failure Handling
- If LLM query planning is unavailable, use rule-based query expansion.
- If external APIs fail, continue with partial results or demo-mode generated papers.
- If too few papers are found, ask the Agent to broaden retrieval.

## Do Not
- Do not compute graph centrality or communities.
- Do not generate the final reading path.
- Do not let the LLM directly decide the final accepted paper set.
