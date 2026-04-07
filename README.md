# ollama-profiler

Profile and compare [Ollama](https://ollama.com) model performance side-by-side.

[![PyPI](https://img.shields.io/pypi/v/ollama-profiler)](https://pypi.org/project/ollama-profiler/)
[![Tests](https://github.com/piercecohen1/ollama-profiler/actions/workflows/test.yml/badge.svg)](https://github.com/piercecohen1/ollama-profiler/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Measure tokens/sec, time-to-first-token, and 7 other metrics across your local models. Interactive TUI or rich CLI. Export to JSON, HTML, or PNG.

## Install

```sh
# Run directly — no install needed
uvx ollama-profiler --help

# Or install globally
pip install ollama-profiler
```

<details>
<summary>Other install methods</summary>

```sh
# Go
go install github.com/piercecohen1/ollama-profiler@latest

# Direct binary (macOS/Linux)
curl -fsSL https://raw.githubusercontent.com/piercecohen1/ollama-profiler/main/install.sh | sh
```

</details>

## Quick Start

```sh
# Compare two models
ollama-profiler gemma4:e4b llama3.2:3b

# Interactive TUI with mouse support
ollama-profiler --tui

# Rigorous comparison with balanced scheduling
ollama-profiler gemma4:e4b gemma4:26b gemma4:31b \
  -n 4 --rounds 4 --balanced --warmup --cooldown 30
```

## Features

- **Side-by-side comparison** of multiple models on the same prompt
- **TUI + CLI** — interactive full-screen UI or styled terminal output
- **Fair scheduling** — sequential, round-robin, randomized rounds, or Latin-square balanced
- **Statistical summary** — mean, stddev, min/max, relative %, color-coded winners
- **Export** — JSON with metadata, self-contained HTML report, retina PNG charts
- **Reproducible** — deterministic seed, configurable token limits, warmup & cooldown

Requires [Ollama](https://ollama.com) running locally, or pass `--url` for a remote server. Run `ollama-profiler --help` for full usage.

## License

MIT
