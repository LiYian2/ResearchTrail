# ResearchTrail

**ResearchTrail: A citation- and similarity-network agent for personalized research reading paths**

ResearchTrail helps a user enter a new research field from a natural-language goal. It retrieves relevant papers, constructs citation and semantic-similarity networks, identifies foundational, bridge, and frontier papers, and generates a personalized reading path with abstracts, URLs, evidence-grounded explanations, and visualizations.

Repository for StudyClawHub submission:

- GitHub: `https://github.com/LiYian2/ResearchTrail`
- Submit type: `Agent`
- Agent path: `.`
- Child skill paths:
  - `skill_retrieval`
  - `skill_graph`
  - `skill_reading_path`

## Why This Is Not Just an arXiv Summarizer

Most paper briefing agents stop at search and abstract summaries. ResearchTrail instead models a research field as a paper network:

```text
User learning goal
  -> LLM/rule planner
  -> Literature Retrieval Skill
  -> Citation + Similarity Graph Skill
  -> Reading Path + Report Skill
  -> Report, figures, state file, and follow-up suggestions
```

LLMs are used only for flexible semantic work: intent parsing, query normalization, acronym expansion, community labels, and evidence-grounded why-read explanations. Deterministic code handles retrieval, deduplication, graph construction, PageRank, betweenness, community detection, scoring, and evaluation.

The repository also includes a report-personalization layer for external coding/writing agents. `AGENT.md` explains how a tool such as Claude Code, OpenClaw, or Codex may use `MEMORY.md` to polish generated reports for a specific user while preserving paper metadata, graph scores, URLs, abstracts, and limitations.

## Skills

| Skill | Folder | When the agent should use it | Backend |
|---|---|---|---|
| Literature Retrieval | `skill_retrieval/` | User asks to enter, survey, learn, or explore a research topic | arXiv/OpenAlex/Semantic Scholar retrieval, topic filtering, verified landmarks, corpus quality checks |
| Research Graph Analysis | `skill_graph/` | A paper corpus exists and the agent needs field structure or paper roles | directed citation graph, TF-IDF/LSA similarity graph, PageRank, betweenness, communities |
| Reading Path and Report | `skill_reading_path/` | Graph scores exist and the user needs a reading list, report, figures, or follow-up explanation | staged path generation, evidence packets, Markdown report, visualizations, GUI-ready state |

Each skill has a StudyClawHub-compatible `SKILL.md` that tells an agent when to use the skill, how to invoke the code, and how to handle failures.

## Quick Start

Use the conda environment described below.

```bash
conda env create -f environment.yml
conda activate network
python -m pytest tests -q
```

Run a stable offline demo:

```bash
python main.py "I am a beginner and want to understand Vision Transformer" \
  --demo \
  --llm off \
  --max-papers 40 \
  --similarity-backend lsa \
  --output-dir outputs/demo_vit
```

Run a live LLM-assisted workflow:

```bash
export SILICON_FLOW_API="your_siliconflow_key"
export S2_API_KEY="your_semantic_scholar_key"  # optional but recommended

python main.py "I want to understand the research trail of diffusion models, from DDPM to score-based modeling and latent diffusion" \
  --llm auto \
  --llm-provider siliconflow \
  --llm-model Pro/zai-org/GLM-4.7 \
  --max-papers 45 \
  --similarity-backend lsa \
  --output-dir outputs/diffusion_live
```

Run the Gradio demo interface:

```bash
python app.py
```

## Outputs

Each full run writes:

- `research_report.md`: final reading-path report
- `research_graph.png`: paper network visualization
- `scores_distribution.png`: role score visualization
- `state.json`: machine-readable papers, graph, scores, communities, and path metrics

Sample output files are included in `outputs_submission/`:

- `sample_research_report.md`
- `sample_research_graph.png`
- `sample_scores_distribution.png`

## Evaluation Summary

Full evaluation details are in `docs/evaluation.md`.

### Overall Agent Benchmark

| Scope | Topics | Papers | Edges | Communities | Landmark Hit | Topic Precision | Ordering | Community Coverage | Stage Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benchmark topics | 11 | 22.7 | 131.7 | 4.0 | 67.6% | 93.3% | 74.6% | 90.6% | 97.7% |

### Skill Ablations

| Component | Main ablation | Key result |
|---|---|---|
| Skill 1 retrieval | arXiv only vs OpenAlex only vs combined vs filtering vs verified landmarks | Combined retrieval + filtering improves graph edge yield from 5.01 to 6.00 and topic precision from 61.4% to 67.2%. |
| Skill 2 graph | citation-only vs similarity-only vs hybrid | Citation-only is sparse; hybrid reaches 97.7% largest component ratio and 90.6% path community coverage. |
| Skill 2 similarity backend | TF-IDF vs LSA under the same hybrid graph mode | LSA improves largest component ratio from 92.1% to 94.3%, path community coverage from 86.0% to 88.8%, and foundation landmark hit from 72.7% to 81.8%. |
| Skill 3 reading path | random vs citation count vs PageRank vs ResearchTrail staged | ResearchTrail is the only variant with high stage coverage, 97.7%, while keeping 95.3% topic precision. |

## Related Work

ResearchTrail combines ideas from four areas:

- Literature discovery systems such as arXiv search, Semantic Scholar, OpenAlex, and survey-oriented paper recommendation.
- Social network analysis methods, especially PageRank, betweenness centrality, community detection, and bridge-node interpretation.
- Citation-network and bibliometric analysis, where papers are treated as nodes and citation/reference links encode scholarly dependency.
- LLM-assisted agent workflows, where LLMs handle semantic planning and report writing while deterministic tools perform verifiable computation.

Compared with daily arXiv briefing projects, ResearchTrail focuses on **field structure and reading order** rather than only recent paper summaries.

## Reproducibility

Create the environment:

```bash
conda env create -f environment.yml
conda activate network
```

If you already have the course environment:

```bash
conda activate network
pip install -r requirements.txt
```

Optional API keys:

```bash
export SILICON_FLOW_API="..."  # LLM planning, community labels, explanations
export S2_API_KEY="..."        # Semantic Scholar citation/reference enrichment
```

Run tests:

```bash
python -m pytest tests -q
```

Run evaluation scripts:

```bash
python -m evaluation.corpus_ablation_batch --max-papers 45 --output-dir outputs/corpus_ablation/full_12_topics
python -m evaluation.graph_ablation --topic diffusion_models --max-papers 30 --output-dir outputs/graph_ablation/diffusion
python -m evaluation.similarity_backend_ablation --state-root outputs/benchmark_full_llm_normalized --output-dir outputs/similarity_backend_ablation/full_12_topics
python -m evaluation.benchmark_runner --max-papers 45 --output-dir outputs/benchmark_full_llm_normalized
```

## Repository Structure

```text
.
├── AGENTS.md
├── AGENT.md
├── MEMORY.md
├── README.md
├── STUDYCLAWHUB_SUBMISSION.md
├── main.py
├── app.py
├── agent/
├── skill_retrieval/
├── skill_graph/
├── skill_reading_path/
├── shared/
├── evaluation/
├── docs/
├── outputs_submission/
└── tests/
```

## License

MIT License. See `LICENSE`.
