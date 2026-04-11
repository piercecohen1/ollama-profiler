package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/piercecohen1/ollama-profiler/internal/bench"
	"github.com/piercecohen1/ollama-profiler/internal/cli"
	"github.com/piercecohen1/ollama-profiler/internal/tui"
	"github.com/spf13/cobra"
)

var version = "dev"

const defaultPrompt = "Write a 200 word explanation of transformers in ML."

var (
	runs       int
	prompt     string
	promptFile string
	baseURL    string
	warmup     bool
	roundRobin bool
	rounds     int
	balanced   bool
	cooldown   int
	jsonFile   string
	htmlFile   string
	pngFile    string
	exportDir  string
	noPerRun   bool
	useTUI     bool
	dryRun     bool
	numPredict int
	seed       int
	think      string
)

var (
	runCLI = cli.Run
	runTUI = tui.Run
)

func main() {
	if err := newRootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}

func newRootCmd() *cobra.Command {
	resetFlagDefaults()

	rootCmd := &cobra.Command{
		Use:     "ollama-profiler [models...]",
		Version: version,
		Short:   "Benchmark and compare Ollama models side-by-side",
		Long: `ollama-profiler — Compare Ollama model performance side-by-side.

Runs in non-interactive CLI mode by default. Pass --tui to launch the
interactive full-screen interface instead.

Scheduling modes:
  (default)       Sequential — all runs of model A, then B, then C.
  --round-robin   Interleave — cycle through models each run.
  --rounds R      Multiple rounds with randomized model order per round.
  --balanced      With --rounds, use Latin-square positional balancing.`,
		Example: `  ollama-profiler gemma4:e4b gemma4:26b gemma4:31b -n 4
  ollama-profiler gemma4:e4b gemma4:26b -n 3 --rounds 4 --cooldown 30
  ollama-profiler gemma4:e4b gemma4:26b -n 3 --rounds 3 --balanced --warmup
  ollama-profiler --tui                                         # interactive TUI
  ollama-profiler --tui --dry-run                               # TUI with fake data`,
		Args: cobra.ArbitraryArgs,
		RunE: runRoot,
	}

	f := rootCmd.Flags()
	f.IntVarP(&runs, "runs", "n", 3, "Runs per model per round")
	f.StringVarP(&prompt, "prompt", "p", defaultPrompt, "Prompt to send")
	f.StringVar(&promptFile, "prompt-file", "", "Read prompt from file")
	f.StringVar(&baseURL, "url", "", "Ollama API base URL (default: $OLLAMA_HOST or http://localhost:11434)")
	f.BoolVar(&warmup, "warmup", false, "Run each model once before benchmarking (not counted)")
	f.BoolVar(&roundRobin, "round-robin", false, "Interleave models each run")
	f.IntVar(&rounds, "rounds", 1, "Number of rounds (model order randomized per round)")
	f.BoolVar(&balanced, "balanced", false, "With --rounds, use Latin-square positional balancing")
	f.IntVar(&cooldown, "cooldown", 0, "Seconds to sleep between rounds (thermal recovery)")
	f.StringVar(&jsonFile, "json", "", "Export raw results to JSON file")
	f.StringVar(&htmlFile, "html", "", "Export HTML report to file")
	f.StringVar(&pngFile, "png", "", "Export charts PNG to file")
	f.StringVar(&exportDir, "export", "", "Export full bundle (JSON + HTML + PNG) to directory")
	f.BoolVar(&noPerRun, "no-per-run", false, "Skip per-run detail tables")
	f.IntVar(&numPredict, "num-predict", 256, "Max tokens to generate per run (0 = unlimited)")
	f.IntVar(&seed, "seed", 42, "Random seed for deterministic output (0 = random)")
	f.StringVar(&think, "think", "", `Thinking mode: "true", "low", "medium", "high" (default: disabled)`)
	f.Lookup("think").NoOptDefVal = "true"
	f.BoolVar(&useTUI, "tui", false, "Launch interactive TUI")
	f.BoolVar(&dryRun, "dry-run", false, "Use fake data (TUI only; no Ollama required)")

	return rootCmd
}

func resetFlagDefaults() {
	runs = 3
	prompt = defaultPrompt
	promptFile = ""
	baseURL = ""
	warmup = false
	roundRobin = false
	rounds = 1
	balanced = false
	cooldown = 0
	jsonFile = ""
	htmlFile = ""
	pngFile = ""
	exportDir = ""
	noPerRun = false
	useTUI = false
	dryRun = false
	numPredict = 256
	seed = 42
	think = ""
}

func runRoot(cmd *cobra.Command, args []string) error {
	resolvedURL := bench.ResolveBaseURL(baseURL)

	if useTUI {
		if jsonFile != "" || htmlFile != "" || pngFile != "" || exportDir != "" {
			return fmt.Errorf("--json, --html, --png, and --export are only valid in CLI mode")
		}
		return runTUI(dryRun, resolvedURL)
	}

	if len(args) == 0 {
		return fmt.Errorf("at least one model is required (or use --tui for interactive mode)")
	}

	if dryRun {
		return fmt.Errorf("--dry-run is only valid in TUI mode (use --tui)")
	}
	if runs < 1 {
		return fmt.Errorf("--runs must be at least 1")
	}
	if rounds < 1 {
		return fmt.Errorf("--rounds must be at least 1")
	}
	if cooldown < 0 {
		return fmt.Errorf("--cooldown must be non-negative")
	}
	if numPredict < 0 {
		return fmt.Errorf("--num-predict must be non-negative")
	}
	if seed < 0 {
		return fmt.Errorf("--seed must be non-negative")
	}
	if err := validateThinkFlag(think); err != nil {
		return err
	}
	think = strings.ToLower(strings.TrimSpace(think))

	if roundRobin && rounds > 1 {
		return fmt.Errorf("--round-robin and --rounds are mutually exclusive")
	}
	if balanced && rounds <= 1 {
		return fmt.Errorf("--balanced requires --rounds > 1")
	}
	if cooldown > 0 && rounds <= 1 && !roundRobin {
		return fmt.Errorf("--cooldown requires --rounds > 1")
	}

	if exportDir != "" && (jsonFile != "" || htmlFile != "" || pngFile != "") {
		return fmt.Errorf("--export cannot be combined with --json, --html, or --png")
	}

	// Reject duplicate output paths among standalone export flags.
	if jsonFile != "" || htmlFile != "" || pngFile != "" {
		paths := map[string]string{}
		for flag, raw := range map[string]string{"--json": jsonFile, "--html": htmlFile, "--png": pngFile} {
			if raw == "" {
				continue
			}
			abs, err := filepath.Abs(filepath.Clean(raw))
			if err != nil {
				abs = raw
			}
			if prev, ok := paths[abs]; ok {
				return fmt.Errorf("%s and %s resolve to the same output path %q", prev, flag, abs)
			}
			paths[abs] = flag
		}
	}

	if promptFile != "" {
		data, err := os.ReadFile(promptFile)
		if err != nil {
			return fmt.Errorf("reading prompt file: %w", err)
		}
		prompt = string(data)
	}

	return runCLI(cli.Config{
		Models:     args,
		Runs:       runs,
		Prompt:     prompt,
		BaseURL:    resolvedURL,
		Warmup:     warmup,
		RoundRobin: roundRobin,
		Rounds:     rounds,
		Balanced:   balanced,
		Cooldown:   cooldown,
		JSONFile:   jsonFile,
		HTMLFile:   htmlFile,
		PNGFile:    pngFile,
		ExportDir:  exportDir,
		NoPerRun:   noPerRun,
		NumPredict: numPredict,
		Seed:       seed,
		Think:      think,
	})
}

func validateThinkFlag(value string) error {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "true", "low", "medium", "high":
		return nil
	default:
		return fmt.Errorf(`--think must be one of: "true", "low", "medium", "high"`)
	}
}
