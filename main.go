package main

import (
	"fmt"
	"os"

	"github.com/piercecohen1/ollama-bench/internal/bench"
	"github.com/piercecohen1/ollama-bench/internal/cli"
	"github.com/piercecohen1/ollama-bench/internal/tui"
	"github.com/spf13/cobra"
)

var version = "dev"

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
	noPerRun   bool
	useTUI     bool
	dryRun     bool
	numPredict int
	seed       int
	think      string
)

func main() {
	rootCmd := &cobra.Command{
		Use:     "ollama-bench [models...]",
		Version: version,
		Short:   "Benchmark and compare Ollama models side-by-side",
		Long: `ollama-bench — Compare Ollama model performance side-by-side.

Scheduling modes:
  (default)       Sequential — all runs of model A, then B, then C.
  --round-robin   Interleave — cycle through models each run.
  --rounds R      Multiple rounds with randomized model order per round.
  --balanced      With --rounds, use Latin-square positional balancing.`,
		Example: `  ollama-bench gemma4:e4b gemma4:26b gemma4:31b -n 4
  ollama-bench gemma4:e4b gemma4:26b -n 3 --rounds 4 --cooldown 30
  ollama-bench gemma4:e4b gemma4:26b -n 3 --rounds 3 --balanced --warmup
  ollama-bench gemma4:e4b gemma4:26b --tui`,
		Args: cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			resolvedURL := bench.ResolveBaseURL(baseURL)

			if useTUI {
				return tui.Run(dryRun, resolvedURL)
			}

			if len(args) == 0 {
				return fmt.Errorf("at least one model is required (or use --tui for interactive mode)")
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

			if roundRobin && rounds > 1 {
				return fmt.Errorf("--round-robin and --rounds are mutually exclusive")
			}
			if balanced && rounds <= 1 {
				return fmt.Errorf("--balanced requires --rounds > 1")
			}
			if cooldown > 0 && rounds <= 1 && !roundRobin {
				return fmt.Errorf("--cooldown requires --rounds > 1")
			}

			if promptFile != "" {
				data, err := os.ReadFile(promptFile)
				if err != nil {
					return fmt.Errorf("reading prompt file: %w", err)
				}
				prompt = string(data)
			}

			return cli.Run(cli.Config{
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
				NoPerRun:   noPerRun,
				NumPredict: numPredict,
				Seed:       seed,
				Think:      think,
			})
		},
	}

	f := rootCmd.Flags()
	f.IntVarP(&runs, "runs", "n", 3, "Runs per model per round")
	f.StringVarP(&prompt, "prompt", "p", "Write a 200 word explanation of transformers in ML.", "Prompt to send")
	f.StringVar(&promptFile, "prompt-file", "", "Read prompt from file")
	f.StringVar(&baseURL, "url", "", "Ollama API base URL (default: $OLLAMA_HOST or http://localhost:11434)")
	f.BoolVar(&warmup, "warmup", false, "Run each model once before benchmarking (not counted)")
	f.BoolVar(&roundRobin, "round-robin", false, "Interleave models each run")
	f.IntVar(&rounds, "rounds", 1, "Number of rounds (model order randomized per round)")
	f.BoolVar(&balanced, "balanced", false, "With --rounds, use Latin-square positional balancing")
	f.IntVar(&cooldown, "cooldown", 0, "Seconds to sleep between rounds (thermal recovery)")
	f.StringVar(&jsonFile, "json", "", "Export raw results to JSON file")
	f.BoolVar(&noPerRun, "no-per-run", false, "Skip per-run detail tables")
	f.IntVar(&numPredict, "num-predict", 256, "Max tokens to generate per run (0 = unlimited)")
	f.IntVar(&seed, "seed", 42, "Random seed for deterministic output (0 = random)")
	f.StringVar(&think, "think", "", `Thinking mode: "true", "low", "medium", "high" (default: disabled)`)
	f.Lookup("think").NoOptDefVal = "true"
	f.BoolVar(&useTUI, "tui", false, "Launch interactive TUI")
	f.BoolVar(&dryRun, "dry-run", false, "Use fake data to test TUI without Ollama")

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

