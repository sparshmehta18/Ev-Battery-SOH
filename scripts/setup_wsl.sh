#!/usr/bin/env bash
set -euo pipefail

# WSL2 setup helper for ev-battery-rul
# Run from your WSL Ubuntu shell. Assumes the repo is at /mnt/c/Users/HP/Desktop/ev-battery-rul

REPO_PATH="${1:-/mnt/c/Users/HP/Desktop/ev-battery-rul}"
if [ ! -d "$REPO_PATH" ]; then
  echo "Repo not found at $REPO_PATH. Usage: $0 /path/to/repo (or place the repo at /mnt/c/Users/HP/Desktop/ev-battery-rul)"
  exit 1
fi

sudo apt update
sudo apt install -y build-essential git ca-certificates libffi-dev python3-venv python3-pip

cd "$REPO_PATH"

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Run explainability script
python src/models/explainability.py

echo "Explainability run finished. Outputs:"
echo " - $REPO_PATH/results/shap_explanation_example.json"
echo " - $REPO_PATH/results/figures/09_shap_summary.png"
