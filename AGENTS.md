# ollama-profiler

CLI + TUI tool for benchmarking and comparing Ollama model performance side-by-side.

**Note:** The git repo directory is `ollama-bench` (original name). The binary, module path, and all code references use `ollama-profiler`. TUI title bars display "Ollama Bench".

## Architecture

Single Go binary (`main.go` at the repo root) with two modes:

- **CLI mode** (default): Non-interactive, rich terminal output via lipgloss. Entry point: `internal/cli/cli.go`
- **TUI mode** (`--tui`): Interactive full-screen UI via tview with mouse support. Entry point: `internal/tui/tui.go`

### Package layout

```
main.go                     # Entry point, cobra CLI flag parsing
internal/
  bench/                    # Core engine (no UI dependencies)
    bench.go                # Metrics, RunResult, scheduling, Ollama API, model listing
    bench_test.go           # Table-driven tests for Stats, FmtVal, BuildSchedule
  cli/
    cli.go                  # CLI-mode output: progress bar, per-run tables, summary, relative
  export/
    export.go               # Shared export: JSON, HTML, PNG, Bundle (used by both CLI and TUI)
    export_test.go          # Integration tests for all export formats
  tui/
    tui.go                  # TUI-mode: config screen, benchmark screen, results screen (4 tabs),
                            #   export via shared export package, dry-run fake data
```

**Module:** `github.com/piercecohen1/ollama-profiler` — Go 1.26.1

### Core concepts

- **Metrics**: Defined in `bench.AllMetrics` — TTFT, Generation TPS, Prompt Eval TPS, Total Duration, Load Duration, Prompt Eval Time, Eval Time, Prompt Tokens, Generated Tokens. Each has a `LowerIsBetter` flag (nil = neutral). Subsets: `PerRunMetrics` (7), `RelativeMetrics` (4), `ChartMetrics` (6) control which metrics appear in each view.
- **RunResult**: Struct holding all metrics from one benchmark pass. Has `.Get(key)` for dynamic metric access.
- **Scheduling**: `BuildSchedule()` generates run order as `[][]string` (list of rounds, each round is a list of model names). Modes:
  - **Sequential** (default): All N runs of model A, then B, then C. Model stays loaded in VRAM.
  - **Round-robin** (`--round-robin`): Interleave A,B,C,A,B,C...
  - **Rounds** (`--rounds R`): R rounds with randomized model order per round (controls for thermal throttling).
  - **Balanced** (`--rounds R --balanced`): Latin-square rotation for positional fairness. Requires `rounds >= len(models)`; validated in both TUI and CLI. Uses simple modular rotation (`r % len(models)`) for clean positional cycling.
- **Stats**: `Stats()` computes mean/stddev/min/max, filtering NaN values. Zero-duration rates from Ollama (e.g. cached prompts) are stored as NaN to avoid poisoning averages.
- **BenchmarkOpts**: Controls Ollama request parameters (`num_predict`, `seed`, `think`) passed to `/api/generate` for reproducible benchmarking. Defaults: 256 max tokens, seed 42, thinking disabled.
- **ResolveBaseURL**: Resolves the Ollama server URL in priority order: explicit `--url` flag → `OLLAMA_HOST` env var → `http://localhost:11434`. Normalizes scheme (adds `http://` if missing), handles bare port (`:11434`), and trims trailing slashes. Both CLI and TUI use this. The TUI receives the resolved URL via `tui.Run(dryRun, baseURL)`.

### TUI screens

1. **Config**: tview List (model multi-select with click/space/enter toggle) + Form (runs, schedule dropdown, rounds, cooldown, warmup, max tokens, seed, think, manual models, prompt). Models auto-detected from Ollama `/api/tags`.
2. **Benchmark**: Progress bar, live results Table, scrolling log TextView. Benchmarks run in a goroutine with `context.Context` for cancellation; all shared state (`results` map, `done` counter) is mutated inside `QueueUpdateDraw()` callbacks to avoid data races. A `resultsShown` guard prevents duplicate results page creation.
3. **Results**: 4 tabs (Summary, Per-Run, Relative, Charts) switchable via tab/arrows. Summary and Relative color-code winners (green) and losers (yellow <10%, red >10%). Charts use Unicode block characters for horizontal bar charts.

### Export

Export logic lives in `internal/export/` and is shared by both CLI and TUI.

**TUI**: Press `e` on the results screen to create a full bundle.
**CLI**: Use `--export DIR` for the full bundle, or `--json FILE`, `--html FILE`, `--png FILE` individually.

The full bundle creates a directory containing:
- `results.json` — raw metrics data wrapped in `{"meta": {...}, "results": {...}}` with full benchmark config for reproducibility
- `report.html` — self-contained dark-themed HTML with summary table + bar charts
- `charts.png` — generated natively via Go `image/png` + `golang.org/x/image/font/gofont/gomono` (no browser needed). 2x resolution for retina quality.

In charts (HTML/PNG), winner is indicated by green model name text. In TUI charts, winner gets a ★ marker. Bar colors are consistent per model across all charts (never change based on winner status).

## Building

```bash
go build -o ollama-profiler .                     # local build
make build                                      # same, with ldflags
make test                                       # run tests
make dist                                       # cross-compile all platforms → dist/
```

Cross-compilation targets: darwin/amd64, darwin/arm64, linux/amd64, linux/arm64, windows/amd64, windows/arm64.

## CI/CD

- **test.yml**: Runs `go test` + build + smoke test on push to `main` and PRs. Matrix: ubuntu, macos, windows.
- **publish.yml**: On GitHub release, cross-compiles binaries (6 targets), uploads to release, builds Python wheels via `go-to-wheel`, and publishes to PyPI. Installable via `pip install ollama-profiler` or `uvx ollama-profiler`.

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
| `--cooldown SEC` | Sleep between rounds; requires `--rounds > 1` or `--round-robin` |
| `--warmup` | One uncounted warmup run before each model's first counted run per round |
| `--url` | Ollama API base URL (default: `$OLLAMA_HOST` or `http://localhost:11434`) |
| `--num-predict N` | Max tokens to generate per run (default 256, 0 = unlimited) |
| `--seed N` | Random seed for deterministic output (default 42, 0 = random) |
| `--think` | Allow model thinking tokens (disabled by default). Accepts `true`, `low`, `medium`, `high`; bare `--think` = `true` |
| `--json FILE` | Export raw results with metadata to JSON file (CLI mode only) |
| `--html FILE` | Export HTML report to file (CLI mode only) |
| `--png FILE` | Export charts PNG to file (CLI mode only) |
| `--export DIR` | Export full bundle (JSON + HTML + PNG) to directory (CLI mode only) |
| `--no-per-run` | Skip per-run detail tables (CLI mode only) |
| `-p, --prompt` | Prompt text (default: transformers explanation) |
| `--prompt-file` | Read prompt from file |

### Flag validation

- At least one model required unless `--tui`
- `--runs` >= 1, `--rounds` >= 1, `--cooldown` >= 0
- `--round-robin` and `--rounds` are mutually exclusive
- `--balanced` requires `--rounds > 1`
- `--cooldown` requires `--rounds > 1` or `--round-robin`
- `--export` cannot be combined with `--json`, `--html`, or `--png`

## Theme

Colors match the "bench" theme (Claude Code-inspired dark palette):
- Background: `#0d1117`, Surface: `#161b22`, Panel: `#21262d`, Border: `#30363d`
- Primary/purple: `#c9a0ff`, Accent/orange: `#f78166`, Green: `#3fb950`, Red: `#f85149`, Yellow: `#d29922`
- Foreground: `#e6edf3`, Dim: `#484f58`
- Chart bar colors (per-model, 6 total): `#c9a0ff`, `#f78166`, `#79c0ff`, `#d29922`, `#f85149`, `#a5d6ff`

## Testing

```bash
go test ./...                                    # unit tests
./ollama-profiler --tui --dry-run                   # manual TUI testing without Ollama
./ollama-profiler gemma4:e4b llama3.2:3b -n 2       # live CLI test
```

## Concurrency model

The TUI benchmark goroutine follows these rules:
- `BenchmarkOnce()` accepts `context.Context` — cancelled when the user navigates away ('r' on results page)
- All writes to shared state (`results` map, `done` counter) happen inside `QueueUpdateDraw()`, serializing with UI reads on the main thread
- `resultsShown` flag prevents the completion handler from clobbering an already-displayed results page
- Export functions return errors; the UI shows red error text on failure instead of a false success message

## Install script

`install.sh` detects OS/arch, downloads the matching binary from GitHub releases, and installs to `/usr/local/bin`. Supports curl and wget.
