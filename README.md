# ollama-bench

Benchmark and compare [Ollama](https://ollama.com) model performance side-by-side.

Single binary. No dependencies. Works on macOS, Linux, and Windows.

## Features

- **Side-by-side comparison** of multiple models on identical prompts
- **Interactive TUI** with live progress, color-coded results, and Unicode bar charts
- **Rich CLI mode** with styled terminal output (no TUI required)
- **Multiple scheduling modes** — sequential, round-robin, randomized rounds, or Latin-square balanced
- **Statistical summary** — mean, stddev, min/max, relative performance, winner highlighting
- **Export** — JSON, self-contained HTML report, and retina-quality PNG charts
- **Warmup & cooldown** — control for cold-start and thermal throttling effects
- **Dry-run mode** — test the UI without a running Ollama instance

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/piercecohen1/ollama-bench/main/install.sh | sh
```

Or build from source:

```sh
go install github.com/piercecohen1/ollama-bench@latest
```

## Quick Start

```sh
# Compare two models (3 runs each, CLI output)
ollama-bench gemma4:e4b llama3.2:3b

# Interactive TUI
ollama-bench --tui

# Rigorous comparison with balanced scheduling
ollama-bench gemma4:e4b gemma4:26b gemma4:31b \
  -n 4 --rounds 4 --balanced --warmup --cooldown 30
```

## Usage

```
ollama-bench [models...] [flags]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--tui` | Interactive TUI mode | — |
| `-n, --runs` | Runs per model per round | `3` |
| `--rounds N` | Number of rounds (randomized order per round) | `1` |
| `--balanced` | Latin-square positional balancing (with `--rounds`) | — |
| `--round-robin` | Interleave models each run | — |
| `--warmup` | Uncounted warmup run per model | — |
| `--cooldown SEC` | Sleep between rounds | `0` |
| `-p, --prompt` | Prompt text | `"Write a 200 word explanation..."` |
| `--prompt-file` | Read prompt from file | — |
| `--num-predict N` | Max tokens to generate per run (0 = unlimited) | `256` |
| `--seed N` | Random seed for deterministic output (0 = random) | `42` |
| `--think` | Allow model thinking tokens | — |
| `--json FILE` | Export results to JSON (includes metadata) | — |
| `--no-per-run` | Skip per-run detail tables | — |
| `--dry-run` | Fake data, no Ollama needed | — |
| `--url` | Ollama API base URL | `$OLLAMA_HOST` or `http://localhost:11434` |

## Scheduling Modes

| Mode | Flag | Behavior |
|------|------|----------|
| Sequential | *(default)* | All runs of A, then B, then C. Model stays loaded in VRAM. |
| Round Robin | `--round-robin` | Cycle A, B, C, A, B, C... Forces model reloading. |
| Random Rounds | `--rounds N` | N rounds, model order randomized each round. |
| Balanced | `--rounds N --balanced` | Latin-square rotation for positional fairness. Requires `N >= len(models)`. |

## Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| TTFT | ms | Time to first token |
| Generation TPS | tok/s | Token generation throughput |
| Prompt Eval TPS | tok/s | Prompt processing throughput |
| Total Duration | ms | End-to-end inference time |
| Load Duration | ms | Model load time |
| Eval Time | ms | Generation duration |
| Prompt Tokens | tok | Input token count |
| Generated Tokens | tok | Output token count |

## Export

Press `e` in the TUI results screen to export:

```
ollama-bench-2024-01-15-1430/
├── results.json    # Raw metrics data
├── report.html     # Self-contained dark-themed HTML report
└── charts.png      # Retina-quality bar charts (generated natively)
```

## Requirements

- [Ollama](https://ollama.com) running locally (or specify `--url` for remote)
- At least one model pulled (`ollama pull gemma4:e4b`)

## License

MIT
