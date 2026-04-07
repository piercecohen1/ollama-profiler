# ollama-profiler

[![PyPI](https://img.shields.io/pypi/v/ollama-profiler.svg)](https://pypi.org/project/ollama-profiler/)
[![Tests](https://github.com/piercecohen1/ollama-profiler/actions/workflows/test.yml/badge.svg)](https://github.com/piercecohen1/ollama-profiler/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Profile and compare [Ollama](https://ollama.com) model performance side-by-side.

Measure tokens/sec, time-to-first-token, and 7 other metrics across your local models. Interactive TUI or rich CLI. Export to JSON, HTML, or PNG.

![demo](.github/demo.gif)

## Installation

This Go tool can be installed directly [from PyPI](https://pypi.org/project/ollama-profiler/) using `pip` or `uv`.

You can run it without installing it first using `uvx`:

```bash
uvx ollama-profiler --help
```

Or install it, then run `ollama-profiler`:

```bash
uv tool install ollama-profiler
# or
pip install ollama-profiler
```

<details>
<summary>Other install methods</summary>

You can also install the Go binary directly:

```bash
go install github.com/piercecohen1/ollama-profiler@latest
```

Compiled binaries are available [on the releases page](https://github.com/piercecohen1/ollama-profiler/releases).

</details>

## Quick Start

```bash
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
