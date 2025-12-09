#!/usr/bin/env bash
set -e

# Ativa o ambiente virtual, se for o seu padrão
if [ -d ".venv" ]; then
  # Linux
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  # Windows (Git Bash)
  elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
  fi
fi

# Executa o script Python de baselines
python scripts/run_dt_baseline.py "$@"
