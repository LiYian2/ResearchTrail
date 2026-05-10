#!/usr/bin/env bash
set -euo pipefail

python main.py "I am a beginner and want to understand Vision Transformer" \
  --demo \
  --llm off \
  --max-papers 40 \
  --output-dir outputs/demo_vit
