#!/usr/bin/env bash
# Linux / macOS launcher:  bash run.sh [--mode quick]
cd "$(dirname "$0")" || exit 1
python3 run.py "$@"
