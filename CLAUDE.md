# ollama-bench

CLI + TUI tool for benchmarking and comparing Ollama model performance side-by-side.

## Architecture

Single Go binary (`cmd/ollama-bench/main.go`) with two modes:

- **CLI mode** (default): Non-interactive, rich terminal output via lipgloss. Entry point: `internal/cli/cli.go`
- **TUI mode** (`--tui`): Interactive full-screen UI via tview with mouse support. Entry point: `internal/tui/tui.go`

### Package layout

```
cmd/ollama-bench/main.go    # Entry point, cobra CLI flag parsing
internal/
  bench/                    # Core engine (no UI dependencies)
    bench.go                # Metrics, RunResult, scheduling, Ollama API, model listing
    bench_test.go           # Table-driven tests for Stats, FmtVal, BuildSchedule
  cli/
    cli.go                  # CLI-mode output: progress bar, per-run tables, summary, relative, JSON export
  tui/
    tui.go                  # TUI-mode: config screen, benchmark screen, results screen (4 tabs),
                            #   export (JSON + HTML + PNG), dry-run fake data
```

### Core concepts

- **Metrics**: Defined in `bench.AllMetrics` — TTFT, Generation TPS, Prompt Eval TPS, Total Duration, Load Duration, Prompt Eval Time, Eval Time, Prompt Tokens, Generated Tokens. Each has a `LowerIsBetter` flag (nil = neutral).
- **RunResult**: Struct holding all metrics from one benchmark pass. Has `.Get(key)` for dynamic metric access.
- **Scheduling**: `BuildSchedule()` generates run order as `[][]string` (list of rounds, each round is a list of model names). Modes:
  - **Sequential** (default): All N runs of model A, then B, then C. Model stays loaded in VRAM.
  - **Round-robin** (`--round-robin`): Interleave A,B,C,A,B,C...
  - **Rounds** (`--rounds R`): R rounds with randomized model order per round (controls for thermal throttling).
  - **Balanced** (`--rounds R --balanced`): Latin-square rotation for positional fairness.
- **Stats**: `Stats()` computes mean/stddev/min/max, filtering NaN values.

### TUI screens

1. **Config**: tview List (model multi-select with click/space/enter toggle) + Form (runs, schedule dropdown, rounds, cooldown, warmup, manual models, prompt). Models auto-detected from Ollama `/api/tags`.
2. **Benchmark**: Progress bar, live results Table, scrolling log TextView. Benchmarks run in a goroutine, UI updates via `app.QueueUpdateDraw()`.
3. **Results**: 4 tabs (Summary, Per-Run, Relative, Charts) switchable via tab/arrows. Summary and Relative color-code winners (green) and losers (yellow <10%, red >10%). Charts use Unicode block characters for horizontal bar charts.

### Export (`e` key in results)

Creates a timestamped directory `ollama-bench-YYYY-MM-DD-HHMM/` containing:
- `results.json` — raw metrics data
- `report.html` — self-contained dark-themed HTML with summary table + bar charts
- `charts.png` — generated natively via Go `image/png` + `golang.org/x/image/font/gofont/gomono` (no browser needed). 2x resolution for retina quality.

In charts (HTML/PNG), winner is indicated by green model name text. In TUI charts, winner gets a ★ marker. Bar colors are consistent per model across all charts (never change based on winner status).

## Building

```bash
go build -o ollama-bench ./cmd/ollama-bench/   # local build
make build                                      # same, with ldflags
make test                                       # run tests
make dist                                       # cross-compile all platforms → dist/
```

Cross-compilation targets: darwin/amd64, darwin/arm64, linux/amd64, linux/arm64, windows/amd64.

## Dependencies

- `github.com/spf13/cobra` — CLI flag parsing
- `github.com/charmbracelet/lipgloss` — CLI styled output
- `github.com/rivo/tview` + `github.com/gdamore/tcell/v2` — TUI
- `golang.org/x/image` — PNG export (Go Mono font, opentype rendering)

## Key flags

| Flag | Description |
|------|-------------|
| `--tui` | Launch interactive TUI |
| `--dry-run` | Fake data, no Ollama needed (50ms per run) |
| `-n, --runs` | Runs per model per round (default 3) |
| `--rounds R` | Number of rounds with randomized order |
| `--balanced` | Latin-square positional balancing with `--rounds` |
| `--round-robin` | Interleave models each run |
| `--cooldown SEC` | Sleep between rounds (thermal recovery) |
| `--warmup` | One uncounted warmup run per model |
| `--json FILE` | Export raw results (CLI mode only) |
| `--no-per-run` | Skip per-run detail tables (CLI mode only) |
| `-p, --prompt` | Prompt text (default: transformers explanation) |
| `--prompt-file` | Read prompt from file |

## Theme

Colors match the "bench" theme (Claude Code-inspired dark palette):
- Background: `#0d1117`, Surface: `#161b22`, Panel: `#21262d`, Border: `#30363d`
- Primary/purple: `#c9a0ff`, Green: `#3fb950`, Red: `#f85149`, Yellow: `#d29922`
- Foreground: `#e6edf3`, Dim: `#484f58`

## Testing

```bash
go test ./...                                    # unit tests
./ollama-bench --tui --dry-run                   # manual TUI testing without Ollama
./ollama-bench gemma4:e4b llama3.2:3b -n 2       # live CLI test
```

## Install script

`install.sh` detects OS/arch, downloads the matching binary from GitHub releases, and installs to `/usr/local/bin`. Supports curl and wget.
