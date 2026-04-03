// Package cli implements the non-interactive CLI output for ollama-bench.
package cli

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/piercecohen1/ollama-bench/internal/bench"
)

// Config holds the CLI benchmark configuration.
type Config struct {
	Models     []string
	Runs       int
	Prompt     string
	BaseURL    string
	Warmup     bool
	RoundRobin bool
	Rounds     int
	Balanced   bool
	Cooldown   int
	JSONFile   string
	NoPerRun   bool
}

// styles
var (
	dimStyle    = lipgloss.NewStyle().Faint(true)
	boldStyle   = lipgloss.NewStyle().Bold(true)
	greenBold   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#3fb950"))
	yellowStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#d29922"))
	redStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("#f85149"))
	headerStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#c9a0ff"))
	ruleStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("#30363d"))
)

func rule(width int) string {
	return ruleStyle.Render(strings.Repeat("─", width))
}

// Run executes the CLI benchmark flow.
func Run(cfg Config) error {
	schedule := bench.BuildSchedule(cfg.Models, cfg.Runs, bench.ScheduleConfig{
		RoundRobin: cfg.RoundRobin,
		Rounds:     cfg.Rounds,
		Balanced:   cfg.Balanced,
	})
	nRounds := len(schedule)
	totalPerModel := cfg.Runs * nRounds

	// Header
	fmt.Println()
	fmt.Println(rule(80))
	fmt.Println(headerStyle.Render("  Ollama Model Benchmark"))
	fmt.Println(rule(80))

	fmt.Printf("  %s  %s\n", dimStyle.Render("Models:"), strings.Join(cfg.Models, ", "))

	modeStr := "sequential"
	if cfg.RoundRobin {
		modeStr = "round-robin"
	} else if nRounds > 1 {
		kind := "random order"
		if cfg.Balanced {
			kind = "balanced"
		}
		modeStr = fmt.Sprintf("%d rounds (%s)", nRounds, kind)
	}
	fmt.Printf("  %s  %s\n", dimStyle.Render("Mode:"), modeStr)
	fmt.Printf("  %s  %d/model/round x %d round(s) = %d total/model\n",
		dimStyle.Render("Runs:"), cfg.Runs, nRounds, totalPerModel)

	if cfg.Cooldown > 0 {
		fmt.Printf("  %s  %ds between rounds\n", dimStyle.Render("Cooldown:"), cfg.Cooldown)
	}
	if cfg.Warmup {
		fmt.Printf("  %s  enabled\n", dimStyle.Render("Warmup:"))
	}

	prompt := cfg.Prompt
	if len(prompt) > 80 {
		prompt = prompt[:80] + "…"
	}
	fmt.Printf("  %s  %s\n", dimStyle.Render("Prompt:"), prompt)

	// Show schedule for multi-round
	if nRounds > 1 {
		fmt.Println()
		for ri, round := range schedule {
			seen := []string{}
			for _, m := range round {
				found := false
				for _, s := range seen {
					if s == m {
						found = true
						break
					}
				}
				if !found {
					seen = append(seen, m)
				}
			}
			fmt.Printf("  %s  %s\n", dimStyle.Render(fmt.Sprintf("Round %d:", ri+1)), strings.Join(seen, " -> "))
		}
	}
	fmt.Println()

	// Run benchmarks
	results := runBenchmarks(cfg, schedule)
	fmt.Println()

	// Display
	if !cfg.NoPerRun {
		showPerRun(results, cfg.Models, nRounds)
	}
	showSummary(results, cfg.Models)
	fmt.Println()
	showRelative(results, cfg.Models)

	// JSON export
	if cfg.JSONFile != "" {
		if err := exportJSON(results, cfg.Models, cfg.JSONFile); err != nil {
			return fmt.Errorf("export failed: %w", err)
		}
		fmt.Printf("\n%s\n", dimStyle.Render("Results exported to "+cfg.JSONFile))
	}

	fmt.Println()
	return nil
}

// ── Benchmark runner ────────────────────────────────────────────────────────

func runBenchmarks(cfg Config, schedule [][]string) map[string][]*bench.RunResult {
	results := make(map[string][]*bench.RunResult)
	nRounds := len(schedule)
	total := 0
	for _, s := range schedule {
		total += len(s)
	}
	if cfg.Warmup {
		total += len(cfg.Models)
	}
	done := 0

	clearLine := func() {
		fmt.Print("\r\033[K")
	}

	progress := func(desc string) {
		pct := 0
		if total > 0 {
			pct = done * 100 / total
		}
		barWidth := 30
		filled := pct * barWidth / 100
		bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
		clearLine()
		fmt.Printf("  %s %s %3d%%  %s", headerStyle.Render(bar), dimStyle.Render(fmt.Sprintf("%d/%d", done, total)), pct, desc)
	}

	// Warmup
	if cfg.Warmup {
		for _, model := range cfg.Models {
			progress(fmt.Sprintf("Warmup: %s", model))
			_, err := bench.BenchmarkOnce(model, cfg.Prompt, cfg.BaseURL)
			if err != nil {
				clearLine()
				fmt.Printf("  %s\n", yellowStyle.Render(fmt.Sprintf("⚠ warmup failed for %s: %v", model, err)))
			}
			done++
		}
	}

	// Benchmark
	for ri, sched := range schedule {
		if ri > 0 && cfg.Cooldown > 0 {
			for s := cfg.Cooldown; s > 0; s-- {
				progress(fmt.Sprintf("Cooldown %ds...", s))
				time.Sleep(time.Second)
			}
		}

		counts := make(map[string]int)
		perModel := make(map[string]int)
		for _, m := range sched {
			perModel[m]++
		}

		for _, model := range sched {
			counts[model]++
			run := counts[model]
			rnd := ri + 1

			desc := fmt.Sprintf("%s  run %d/%d", model, run, perModel[model])
			if nRounds > 1 {
				desc = fmt.Sprintf("R%d/%d  %s", rnd, nRounds, desc)
			}
			progress(desc)

			res, err := bench.BenchmarkOnce(model, cfg.Prompt, cfg.BaseURL)
			if err != nil {
				clearLine()
				fmt.Printf("  %s\n", redStyle.Render(fmt.Sprintf("✗ %s: %v", model, err)))
				results[model] = append(results[model], &bench.RunResult{Error: err})
			} else {
				res.Round = rnd
				results[model] = append(results[model], res)
			}
			done++
		}
	}

	clearLine()
	return results
}

// ── Display functions ───────────────────────────────────────────────────────

func showPerRun(results map[string][]*bench.RunResult, models []string, nRounds int) {
	metrics := bench.PerRunMetrics

	for _, model := range models {
		runs := results[model]
		if len(runs) == 0 {
			continue
		}

		fmt.Println(headerStyle.Render("  " + model))
		fmt.Println()

		// Header row
		header := "  "
		if nRounds > 1 {
			header += fmt.Sprintf("%-4s ", "Rnd")
		}
		header += fmt.Sprintf("%-4s", "Run")
		for _, m := range metrics {
			header += fmt.Sprintf("  %10s", m.Label)
		}
		fmt.Println(boldStyle.Render(header))
		fmt.Println("  " + strings.Repeat("─", len(header)-2))

		curRound := 0
		runInRound := 0
		for _, r := range runs {
			if r.Error != nil {
				row := "  "
				if nRounds > 1 {
					row += fmt.Sprintf("%-4s ", "?")
				}
				row += fmt.Sprintf("%-4s", "?")
				for range metrics {
					row += fmt.Sprintf("  %10s", redStyle.Render("err"))
				}
				fmt.Println(row)
				continue
			}

			if r.Round != curRound {
				if curRound != 0 && nRounds > 1 {
					fmt.Println()
				}
				curRound = r.Round
				runInRound = 0
			}
			runInRound++

			row := "  "
			if nRounds > 1 {
				row += fmt.Sprintf("%-4d ", r.Round)
			}
			row += fmt.Sprintf("%-4d", runInRound)
			for _, m := range metrics {
				row += fmt.Sprintf("  %10s", bench.FmtVal(r.Get(m.Key), m.Unit))
			}
			fmt.Println(row)
		}

		// Averages
		valid := validRuns(runs)
		if len(valid) > 1 {
			fmt.Println("  " + strings.Repeat("─", len(header)-2))
			row := "  "
			if nRounds > 1 {
				row += fmt.Sprintf("%-4s ", "")
			}
			row += fmt.Sprintf("%-4s", "avg")
			for _, m := range metrics {
				vals := collectMetric(valid, m.Key)
				s := bench.Stats(vals)
				row += fmt.Sprintf("  %10s", boldStyle.Render(bench.FmtVal(s.Mean, m.Unit)))
			}
			fmt.Println(row)
		}
		fmt.Println()
	}
}

func showSummary(results map[string][]*bench.RunResult, models []string) {
	fmt.Println(headerStyle.Render("  Comparison Summary"))
	fmt.Println()

	// Header
	header := fmt.Sprintf("  %-18s", "Metric")
	for _, m := range models {
		header += fmt.Sprintf("  %18s", m)
	}
	fmt.Println(boldStyle.Render(header))
	fmt.Println("  " + strings.Repeat("─", len(header)-2))

	for _, metric := range bench.AllMetrics {
		modelAvgs := make(map[string]float64)
		cells := make(map[string]string)

		for _, model := range models {
			valid := validRuns(results[model])
			if len(valid) == 0 {
				cells[model] = dimStyle.Render("n/a")
				continue
			}
			vals := collectMetric(valid, metric.Key)
			s := bench.Stats(vals)
			if math.IsNaN(s.Mean) {
				cells[model] = dimStyle.Render("n/a")
				continue
			}
			modelAvgs[model] = s.Mean
			if s.N > 1 && s.StdDev > 0 {
				cells[model] = fmt.Sprintf("%s %s",
					bench.FmtVal(s.Mean, metric.Unit),
					dimStyle.Render("±"+bench.FmtVal(s.StdDev, metric.Unit)))
			} else {
				cells[model] = bench.FmtVal(s.Mean, metric.Unit)
			}
		}

		// Find best
		best := findBest(modelAvgs, metric.LowerIsBetter)

		row := fmt.Sprintf("  %-18s", metric.Label)
		for _, model := range models {
			cell := cells[model]
			if model == best {
				cell = greenBold.Render(cell)
			}
			row += fmt.Sprintf("  %18s", cell)
		}
		fmt.Println(row)
	}
}

func showRelative(results map[string][]*bench.RunResult, models []string) {
	if len(models) < 2 {
		return
	}

	fmt.Println(headerStyle.Render("  Relative Performance (vs best)"))
	fmt.Println()

	header := fmt.Sprintf("  %-18s", "Metric")
	for _, m := range models {
		header += fmt.Sprintf("  %18s", m)
	}
	fmt.Println(boldStyle.Render(header))
	fmt.Println("  " + strings.Repeat("─", len(header)-2))

	for _, metric := range bench.RelativeMetrics {
		modelAvgs := make(map[string]float64)
		for _, model := range models {
			valid := validRuns(results[model])
			vals := collectMetric(valid, metric.Key)
			s := bench.Stats(vals)
			if !math.IsNaN(s.Mean) {
				modelAvgs[model] = s.Mean
			}
		}
		if len(modelAvgs) == 0 {
			continue
		}

		bestModel := findBest(modelAvgs, metric.LowerIsBetter)
		bestVal := modelAvgs[bestModel]

		row := fmt.Sprintf("  %-18s", metric.Label)
		for _, model := range models {
			avg, ok := modelAvgs[model]
			if !ok || bestVal == 0 {
				row += fmt.Sprintf("  %18s", dimStyle.Render("n/a"))
				continue
			}

			var pct float64
			if *metric.LowerIsBetter {
				pct = ((avg - bestVal) / bestVal) * 100
			} else {
				pct = ((bestVal - avg) / bestVal) * 100
			}

			if math.Abs(pct) < 0.5 {
				row += fmt.Sprintf("  %18s", greenBold.Render("★ best"))
			} else {
				label := fmt.Sprintf("+%.1f%%", pct)
				style := yellowStyle
				if pct > 30 {
					style = redStyle
				} else if pct <= 10 {
					style = lipgloss.NewStyle()
				}
				row += fmt.Sprintf("  %18s", style.Render(label))
			}
		}
		fmt.Println(row)
	}
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func validRuns(runs []*bench.RunResult) []*bench.RunResult {
	var out []*bench.RunResult
	for _, r := range runs {
		if r != nil && r.Error == nil {
			out = append(out, r)
		}
	}
	return out
}

func collectMetric(runs []*bench.RunResult, key string) []float64 {
	vals := make([]float64, len(runs))
	for i, r := range runs {
		vals[i] = r.Get(key)
	}
	return vals
}

func findBest(avgs map[string]float64, lowerIsBetter *bool) string {
	if lowerIsBetter == nil || len(avgs) == 0 {
		return ""
	}
	var best string
	var bestVal float64
	first := true
	for m, v := range avgs {
		if first {
			best, bestVal = m, v
			first = false
		} else if *lowerIsBetter && v < bestVal {
			best, bestVal = m, v
		} else if !*lowerIsBetter && v > bestVal {
			best, bestVal = m, v
		}
	}
	return best
}

func exportJSON(results map[string][]*bench.RunResult, models []string, path string) error {
	export := make(map[string][]map[string]interface{})
	for _, model := range models {
		var entries []map[string]interface{}
		for _, r := range results[model] {
			if r == nil || r.Error != nil {
				entries = append(entries, nil)
				continue
			}
			entry := map[string]interface{}{}
			for _, m := range bench.AllMetrics {
				v := r.Get(m.Key)
				if math.IsNaN(v) || math.IsInf(v, 0) {
					entry[m.Key] = nil
				} else {
					entry[m.Key] = v
				}
			}
			entries = append(entries, entry)
		}
		export[model] = entries
	}
	data, err := json.MarshalIndent(export, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
