# StudyClawHub Submission Pack

## Submit Type

- Submit as: `Agent`
- Agent name: `researchtrail`
- GitHub repository: `https://github.com/LiYian2/ResearchTrail`
- Path to Agent folder: `.`
- GitHub username: `LiYian2`

## Child Skills to Register

| Skill | Path | Description |
|---|---|---|
| `literature-retrieval` | `skill_retrieval` | Retrieve, enrich, deduplicate, filter, and quality-check academic papers for a research topic. |
| `research-graph-analysis` | `skill_graph` | Construct citation/similarity networks and identify foundational, bridge, and frontier papers. |
| `reading-path-report` | `skill_reading_path` | Generate personalized reading paths, grounded explanations, reports, and visualizations. |

## Suggested StudyClawHub Description

```text
ResearchTrail is a code-backed, LLM-assisted research navigation agent. It retrieves papers, builds citation/similarity networks, identifies foundation/bridge/frontier papers with SNA metrics, and generates personalized reading paths with reports and visualizations.
```

## Demo Command

```bash
python main.py "I am a beginner and want to understand Vision Transformer" --demo --llm off --max-papers 40 --output-dir outputs/demo_vit
```

## Live Command

```bash
export SILICON_FLOW_API="your_key"
export S2_API_KEY="your_key"
python main.py "I want to understand retrieval-augmented generation from dense retrieval to agentic RAG" --llm auto --max-papers 45 --output-dir outputs/rag_live
```

## Evidence Files

- Agent instructions: `AGENTS.md`
- Skill instructions: `skill_retrieval/SKILL.md`, `skill_graph/SKILL.md`, `skill_reading_path/SKILL.md`
- Sample output: `outputs_submission/`
- Evaluation summary: `docs/evaluation.md`
- Reproducibility: `environment.yml`, `requirements.txt`
- Tests: `tests/`

## Final Checklist

- Confirm the GitHub repository is public or accessible to graders.
- Submit the top-level agent first with path `.`.
- Register the three child skills with the paths listed above.
- Do not register `outputs_submission/` as a skill; it is only sample evidence.
