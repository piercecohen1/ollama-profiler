// Package tui implements the interactive TUI for ollama-profiler using tview.
package tui

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/piercecohen1/ollama-profiler/internal/bench"
	"github.com/piercecohen1/ollama-profiler/internal/export"
	"github.com/rivo/tview"
)

// Colors — matched to Python "bench" theme
var (
	colorPrimary = tcell.NewRGBColor(201, 160, 255) // #c9a0ff
	colorGreen   = tcell.NewRGBColor(63, 185, 80)   // #3fb950
	colorRed     = tcell.NewRGBColor(248, 81, 73)    // #f85149
	colorYellow  = tcell.NewRGBColor(210, 153, 34)   // #d29922
	colorAccent  = tcell.NewRGBColor(247, 129, 102)  // #f78166
	colorDim     = tcell.NewRGBColor(72, 79, 88)     // #484f58
	colorFg      = tcell.NewRGBColor(230, 237, 243)  // #e6edf3
	colorBg      = tcell.NewRGBColor(13, 17, 23)     // #0d1117
	colorSurface = tcell.NewRGBColor(22, 27, 34)     // #161b22
	colorPanel   = tcell.NewRGBColor(33, 38, 45)     // #21262d
	colorBorder  = tcell.NewRGBColor(48, 54, 61)     // #30363d

	chartColors = []tcell.Color{
		colorPrimary, colorAccent,
		tcell.NewRGBColor(121, 192, 255), // #79c0ff
		colorYellow, colorRed,
		tcell.NewRGBColor(165, 214, 255), // #a5d6ff
	}
)

// benchConfig holds configuration gathered from the config screen.
type benchConfig struct {
	Models     []string
	Runs       int
	Prompt     string
	BaseURL    string
	Warmup     bool
	RoundRobin bool
	Rounds     int
	Balanced   bool
	Cooldown   int
	NumPredict int
	Seed       int
	Think      string
}

var (
	dryRunMode bool
	tuiBaseURL string
)

// Run starts the TUI application.
func Run(dryRun bool, baseURL string) error {
	dryRunMode = dryRun
	tuiBaseURL = baseURL
	app := tview.NewApplication().EnableMouse(true)

	// Apply bench theme — matches the Python textual "bench" theme
	tview.Styles.PrimitiveBackgroundColor = colorBg
	tview.Styles.ContrastBackgroundColor = colorSurface
	tview.Styles.MoreContrastBackgroundColor = colorPanel
	tview.Styles.BorderColor = colorBorder
	tview.Styles.TitleColor = colorPrimary
	tview.Styles.GraphicsColor = colorBorder
	tview.Styles.PrimaryTextColor = colorFg
	tview.Styles.SecondaryTextColor = colorPrimary
	tview.Styles.TertiaryTextColor = colorGreen
	tview.Styles.InverseTextColor = colorBg
	tview.Styles.ContrastSecondaryTextColor = colorDim

	pages := tview.NewPages()

	// ── Config page ─────────────────────────────────────────────────────────
	configPage := buildConfigPage(app, pages)
	pages.AddPage("config", configPage, true, true)

	app.SetRoot(pages, true)
	return app.Run()
}

// ── Config Page ─────────────────────────────────────────────────────────────

func buildConfigPage(app *tview.Application, pages *tview.Pages) tview.Primitive {
	// Model selection list
	modelList := tview.NewList().
		ShowSecondaryText(false).
		SetHighlightFullLine(true).
		SetMainTextColor(colorFg).
		SetSelectedBackgroundColor(colorPanel).
		SetSelectedTextColor(colorFg)
	modelList.SetBorder(true).
		SetTitle(" Models (space/enter/click to toggle) ").
		SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).
		SetBackgroundColor(colorBg)

	selected := map[string]bool{}
	var models []bench.OllamaModel
	var err error
	if dryRunMode {
		models = fakeFetchModels()
	} else {
		models, err = bench.FetchModels(tuiBaseURL)
	}

	modelList.SetUseStyleTags(true, false)

	toggleItem := func(idx int) {
		if idx < 0 || idx >= len(models) {
			return
		}
		name := models[idx].Name
		selected[name] = !selected[name]
		label := bench.FormatModelLabel(models[idx])
		if selected[name] {
			modelList.SetItemText(idx, "[green::b]  ✓ "+label+"[-:-:-]", "")
		} else {
			modelList.SetItemText(idx, "    "+label, "")
		}
	}

	if err == nil {
		for _, m := range models {
			modelList.AddItem("    "+bench.FormatModelLabel(m), "", 0, nil)
		}
	}

	// Enter key or mouse click on item toggles it
	modelList.SetSelectedFunc(func(idx int, mainText string, secondaryText string, shortcut rune) {
		toggleItem(idx)
	})

	// Space key also toggles
	modelList.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyRune && event.Rune() == ' ' {
			toggleItem(modelList.GetCurrentItem())
			return nil
		}
		return event
	})

	// Settings form
	modeOptions := []string{"Sequential", "Round Robin", "Rounds — Random", "Rounds — Balanced"}
	currentMode := 0

	form := tview.NewForm().
		AddInputField("Runs / round", "3", 6, tview.InputFieldInteger, nil).
		AddDropDown("Schedule", modeOptions, 0, nil).
		AddInputField("Rounds", "3", 6, tview.InputFieldInteger, nil).
		AddInputField("Cooldown (sec)", "0", 6, tview.InputFieldInteger, nil).
		AddCheckbox("Warmup", false, nil).
		AddInputField("Max tokens", "256", 6, tview.InputFieldInteger, nil).
		AddInputField("Seed", "42", 6, tview.InputFieldInteger, nil).
		AddDropDown("Think", []string{"Disabled", "Enabled", "Low", "Medium", "High"}, 0, nil).
		AddInputField("Manual models", "", 40, nil, nil).
		AddTextArea("Prompt", "Write a 200 word explanation of transformers in ML.", 60, 3, 0, nil)

	// Get references to conditionally disabled fields
	roundsField := form.GetFormItemByLabel("Rounds").(*tview.InputField)
	cooldownField := form.GetFormItemByLabel("Cooldown (sec)").(*tview.InputField)

	savedRounds := "3"
	savedCooldown := "0"

	updateRoundsEnabled := func(mode int) {
		if mode >= 2 { // Rounds — Random or Rounds — Balanced
			roundsField.SetDisabled(false)
			roundsField.SetText(savedRounds)
			roundsField.SetLabel("Rounds")
			cooldownField.SetDisabled(false)
			cooldownField.SetText(savedCooldown)
			cooldownField.SetLabel("Cooldown (sec)")
		} else {
			// Save current values before clearing
			if r := roundsField.GetText(); r != "" {
				savedRounds = r
			}
			if c := cooldownField.GetText(); c != "" {
				savedCooldown = c
			}
			roundsField.SetDisabled(true)
			roundsField.SetText("n/a")
			roundsField.SetLabel("Rounds")
			cooldownField.SetDisabled(true)
			cooldownField.SetText("n/a")
			cooldownField.SetLabel("Cooldown (sec)")
		}
	}
	updateRoundsEnabled(0) // initial state: disabled

	// Wire up dropdown handler
	if dd, ok := form.GetFormItemByLabel("Schedule").(*tview.DropDown); ok {
		dd.SetSelectedFunc(func(text string, index int) {
			currentMode = index
			updateRoundsEnabled(index)
		})
	}

	form.SetBorder(true).
		SetTitle(" Settings ").
		SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).
		SetBackgroundColor(colorBg)
	form.SetFieldBackgroundColor(colorSurface)
	form.SetFieldTextColor(colorFg)
	form.SetButtonBackgroundColor(colorPrimary)
	form.SetButtonTextColor(colorBg)
	form.SetLabelColor(colorFg)

	// Style the dropdown lists for better contrast
	for _, label := range []string{"Schedule", "Think"} {
		if dd, ok := form.GetFormItemByLabel(label).(*tview.DropDown); ok {
			dd.SetListStyles(
				tcell.StyleDefault.Foreground(colorFg).Background(colorSurface),       // unselected
				tcell.StyleDefault.Foreground(colorBg).Background(colorPrimary),        // selected/highlighted
			)
		}
	}

	// Error text
	errorText := tview.NewTextView().
		SetDynamicColors(true).
		SetTextAlign(tview.AlignCenter)

	// Buttons
	form.AddButton("Start Benchmark", func() {
		// Gather selected models
		var selectedModels []string
		for _, m := range models {
			if selected[m.Name] {
				selectedModels = append(selectedModels, m.Name)
			}
		}

		// Add manual models
		manualItem := form.GetFormItemByLabel("Manual models")
		if input, ok := manualItem.(*tview.InputField); ok {
			for _, name := range strings.Split(input.GetText(), ",") {
				name = strings.TrimSpace(name)
				if name != "" {
					found := false
					for _, s := range selectedModels {
						if s == name {
							found = true
							break
						}
					}
					if !found {
						selectedModels = append(selectedModels, name)
					}
				}
			}
		}

		if len(selectedModels) == 0 {
			errorText.SetText("[red]Select at least one model")
			return
		}

		runsStr := form.GetFormItemByLabel("Runs / round").(*tview.InputField).GetText()
		runs, _ := strconv.Atoi(runsStr)
		if runs < 1 {
			runs = 3
		}

		roundsStr := form.GetFormItemByLabel("Rounds").(*tview.InputField).GetText()
		nRounds, _ := strconv.Atoi(roundsStr)
		if nRounds < 1 {
			nRounds = 1
		}

		if currentMode == 3 && nRounds < len(selectedModels) {
			errorText.SetText(fmt.Sprintf("[red]Balanced mode requires at least %d rounds (%d models)", len(selectedModels), len(selectedModels)))
			return
		}

		cooldownStr := form.GetFormItemByLabel("Cooldown (sec)").(*tview.InputField).GetText()
		cooldown, _ := strconv.Atoi(cooldownStr)

		warmup := form.GetFormItemByLabel("Warmup").(*tview.Checkbox).IsChecked()

		numPredictStr := form.GetFormItemByLabel("Max tokens").(*tview.InputField).GetText()
		numPredict, _ := strconv.Atoi(numPredictStr)
		seedStr := form.GetFormItemByLabel("Seed").(*tview.InputField).GetText()
		seed, _ := strconv.Atoi(seedStr)
		thinkIdx, _ := form.GetFormItemByLabel("Think").(*tview.DropDown).GetCurrentOption()
		thinkVal := ""
		switch thinkIdx {
		case 1:
			thinkVal = "true"
		case 2:
			thinkVal = "low"
		case 3:
			thinkVal = "medium"
		case 4:
			thinkVal = "high"
		}

		promptItem := form.GetFormItemByLabel("Prompt").(*tview.TextArea)
		prompt := strings.TrimSpace(promptItem.GetText())
		if prompt == "" {
			errorText.SetText("[red]Enter a prompt")
			return
		}

		cfg := &benchConfig{
			Models:     selectedModels,
			Runs:       runs,
			Prompt:     prompt,
			BaseURL:    tuiBaseURL,
			Warmup:     warmup,
			RoundRobin: currentMode == 1,
			Rounds:     nRounds,
			Balanced:   currentMode == 3,
			Cooldown:   cooldown,
			NumPredict: numPredict,
			Seed:       seed,
			Think:      thinkVal,
		}
		if currentMode >= 2 && nRounds < 2 {
			errorText.SetText("[red]Rounds mode requires at least 2 rounds")
			return
		}

		// Switch to benchmark screen
		benchPage := buildBenchPage(app, pages, cfg)
		pages.AddPage("bench", benchPage, true, true)
		pages.SwitchToPage("bench")
	})

	form.AddButton("Quit", func() {
		app.Stop()
	})

	// Layout
	layout := tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(tview.NewTextView().
			SetText("  Ollama Bench — Configuration").
			SetTextColor(colorPrimary).
			SetTextAlign(tview.AlignLeft), 2, 0, false).
		AddItem(tview.NewFlex().
			AddItem(modelList, 0, 1, true).
			AddItem(form, 0, 1, false), 0, 1, true).
		AddItem(errorText, 1, 0, false)

	return layout
}

// ── Benchmark Page ──────────────────────────────────────────────────────────

func buildBenchPage(app *tview.Application, pages *tview.Pages, cfg *benchConfig) tview.Primitive {
	opts := bench.BenchmarkOpts{
		NumPredict: cfg.NumPredict,
		Seed:       cfg.Seed,
		Think:      cfg.Think,
	}
	results := make(map[string][]*bench.RunResult)
	ctx, cancel := context.WithCancel(context.Background())

	schedule := bench.BuildSchedule(cfg.Models, cfg.Runs, bench.ScheduleConfig{
		RoundRobin: cfg.RoundRobin,
		Rounds:     cfg.Rounds,
		Balanced:   cfg.Balanced,
	})
	total := 0
	for _, s := range schedule {
		total += len(s)
	}
	if cfg.Warmup {
		for _, s := range schedule {
			seen := make(map[string]bool)
			for _, m := range s {
				if !seen[m] {
					seen[m] = true
					total++
				}
			}
		}
	}
	done := 0

	// Status text
	statusView := tview.NewTextView().
		SetDynamicColors(true).
		SetTextAlign(tview.AlignLeft)
	statusView.SetText("[purple]  Starting...")

	// Progress bar (custom draw)
	progressView := tview.NewTextView().SetDynamicColors(true)
	updateProgress := func() {
		pct := 0
		if total > 0 {
			pct = done * 100 / total
			if pct > 100 {
				pct = 100
			}
		}
		barWidth := 50
		filled := pct * barWidth / 100
		if filled > barWidth {
			filled = barWidth
		}
		bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)
		progressView.SetText(fmt.Sprintf("  [purple]%s[white] %3d%%  %d/%d", bar, pct, done, total))
	}
	updateProgress()

	// Live results table
	liveTable := tview.NewTable().
		SetBorders(false).
		SetFixed(1, 0).
		SetSelectable(false, false)
	liveTable.SetBorder(true).SetTitle(" Live Results ").SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).SetBackgroundColor(colorBg)

	for i, h := range []string{"Model", "Rnd", "Run", "TTFT", "Gen TPS", "Prompt TPS", "Total"} {
		liveTable.SetCell(0, i, tview.NewTableCell(h).
			SetTextColor(colorPrimary).
			SetSelectable(false).
			SetAlign(tview.AlignCenter))
	}

	// Log view
	logView := tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true).
		SetMaxLines(500).
		SetChangedFunc(func() { app.Draw() })
	logView.SetBorder(true).SetTitle(" Log ").SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).SetBackgroundColor(colorBg)

	addLogLine := func(line string) {
		fmt.Fprintln(logView, line)
	}

	addResultRow := func(model string, rnd, run int, res *bench.RunResult) {
		row := liveTable.GetRowCount()
		liveTable.SetCell(row, 0, tview.NewTableCell(model).SetTextColor(colorFg))
		liveTable.SetCell(row, 1, tview.NewTableCell(strconv.Itoa(rnd)).SetAlign(tview.AlignCenter))
		liveTable.SetCell(row, 2, tview.NewTableCell(strconv.Itoa(run)).SetAlign(tview.AlignCenter))
		liveTable.SetCell(row, 3, tview.NewTableCell(bench.FmtVal(res.TTFT, "ms")).SetAlign(tview.AlignRight))
		liveTable.SetCell(row, 4, tview.NewTableCell(bench.FmtVal(res.EvalRate, "tok/s")).SetAlign(tview.AlignRight).SetTextColor(colorGreen))
		liveTable.SetCell(row, 5, tview.NewTableCell(bench.FmtVal(res.PromptEvalRate, "tok/s")).SetAlign(tview.AlignRight))
		liveTable.SetCell(row, 6, tview.NewTableCell(bench.FmtVal(res.TotalDuration, "ms")).SetAlign(tview.AlignRight))
		liveTable.ScrollToEnd()
	}

	showResults := func() {
		resultsPage := buildResultsPage(app, pages, cfg, results, cancel)
		pages.AddPage("results", resultsPage, true, true)
		pages.SwitchToPage("results")
	}

	// Run benchmarks in background
	go func() {
		nRounds := len(schedule)

		// Benchmark rounds
		for ri, sched := range schedule {
			// Cooldown
			if ri > 0 && cfg.Cooldown > 0 {
				for s := cfg.Cooldown; s > 0; s-- {
					select {
					case <-ctx.Done():
						return
					default:
					}
					remaining := s
					app.QueueUpdateDraw(func() {
						statusView.SetText(fmt.Sprintf("[yellow]  Cooldown %ds...", remaining))
					})
					time.Sleep(time.Second)
				}
				app.QueueUpdateDraw(func() {
					addLogLine("[white]  Cooldown complete")
				})
			}

			counts := make(map[string]int)
			perModel := make(map[string]int)
			for _, mm := range sched {
				perModel[mm]++
			}

			warmedUp := make(map[string]bool)
			for _, model := range sched {
				select {
				case <-ctx.Done():
					return
				default:
				}

				// Warmup: one uncounted run before first counted run per model per round
				if cfg.Warmup && !warmedUp[model] {
					warmedUp[model] = true
					modelCopy := model
					app.QueueUpdateDraw(func() {
						statusView.SetText(fmt.Sprintf("[purple]  Warmup: %s", modelCopy))
					})
					var warmupErr error
					if dryRunMode {
						time.Sleep(50 * time.Millisecond)
					} else {
						_, warmupErr = bench.BenchmarkOnce(ctx, model, cfg.Prompt, cfg.BaseURL, opts)
					}
					if ctx.Err() != nil {
						return
					}
					errCopy := warmupErr
					app.QueueUpdateDraw(func() {
						done++
						if errCopy != nil {
							addLogLine(fmt.Sprintf("[yellow]  ⚠ Warmup %s failed: %v", modelCopy, errCopy))
						} else {
							addLogLine(fmt.Sprintf("[white]  Warmup %s done", modelCopy))
						}
						updateProgress()
					})
				}

				counts[model]++
				run := counts[model]
				rnd := ri + 1

				status := fmt.Sprintf("%s  run %d/%d", model, run, perModel[model])
				if nRounds > 1 {
					status = fmt.Sprintf("R%d/%d  %s", rnd, nRounds, status)
				}
				statusCopy := status
				app.QueueUpdateDraw(func() {
					statusView.SetText("[purple]  " + statusCopy)
				})

				var res *bench.RunResult
				var err error
				if dryRunMode {
					res = fakeBenchmarkOnce(model)
				} else {
					res, err = bench.BenchmarkOnce(ctx, model, cfg.Prompt, cfg.BaseURL, opts)
				}
				if ctx.Err() != nil {
					return
				}
				modelCopy, rndCopy, runCopy := model, rnd, run

				if err != nil {
					errResult := &bench.RunResult{Error: err}
					app.QueueUpdateDraw(func() {
						done++
						results[modelCopy] = append(results[modelCopy], errResult)
						addLogLine(fmt.Sprintf("[red]  ✗ %s: %v", modelCopy, err))
						updateProgress()
					})
				} else {
					res.Round = rndCopy
					resCopy := res
					app.QueueUpdateDraw(func() {
						done++
						results[modelCopy] = append(results[modelCopy], resCopy)
						addResultRow(modelCopy, rndCopy, runCopy, resCopy)
						addLogLine(fmt.Sprintf("[green]  ✓ %s R%d/run%d — %s tok/s",
							modelCopy, rndCopy, runCopy, bench.FmtVal(resCopy.EvalRate, "tok/s")))
						updateProgress()
					})
				}
			}
		}

		app.QueueUpdateDraw(func() {
			statusView.SetText("[green]  Complete!")
			showResults()
		})
	}()

	layout := tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(tview.NewTextView().
			SetText("  Ollama Bench — Running").
			SetTextColor(colorPrimary), 2, 0, false).
		AddItem(statusView, 1, 0, false).
		AddItem(progressView, 1, 0, false).
		AddItem(liveTable, 0, 2, false).
		AddItem(logView, 0, 1, false)

	return layout
}

// ── Results Page ────────────────────────────────────────────────────────────

func buildResultsPage(app *tview.Application, pages *tview.Pages, cfg *benchConfig, results map[string][]*bench.RunResult, cancel context.CancelFunc) tview.Primitive {
	tabPages := tview.NewPages()

	// Build all tab content
	tabPages.AddPage("Summary", buildSummaryTable(cfg, results), true, true)
	tabPages.AddPage("Per-Run", buildPerRunTable(cfg, results), true, false)
	tabPages.AddPage("Relative", buildRelativeTable(cfg, results), true, false)
	tabPages.AddPage("Charts", buildChartsView(cfg, results), true, false)

	tabNames := []string{"Summary", "Per-Run", "Relative", "Charts"}
	activeTab := 0

	// Tab bar
	tabBar := tview.NewTextView().
		SetDynamicColors(true).
		SetTextAlign(tview.AlignLeft)

	renderTabs := func() {
		var parts []string
		for i, name := range tabNames {
			if i == activeTab {
				parts = append(parts, fmt.Sprintf("[#c9a0ff::b][ %s ][-:-:-]", name))
			} else {
				parts = append(parts, fmt.Sprintf("[#484f58]  %s  [-]", name))
			}
		}
		tabBar.SetText(" " + strings.Join(parts, ""))
	}
	renderTabs()

	// Footer
	footer := tview.NewTextView().
		SetDynamicColors(true).
		SetText("[gray]  tab/←/→: switch view   r: restart   e: export   q: quit")

	layout := tview.NewFlex().SetDirection(tview.FlexRow).
		AddItem(tview.NewTextView().
			SetText("  Ollama Bench — Results").
			SetTextColor(colorPrimary), 2, 0, false).
		AddItem(tabBar, 1, 0, false).
		AddItem(tabPages, 0, 1, true).
		AddItem(footer, 1, 0, false)

	layout.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		switch {
		case event.Key() == tcell.KeyTab || (event.Key() == tcell.KeyRight && event.Modifiers() == 0):
			activeTab = (activeTab + 1) % len(tabNames)
			tabPages.SwitchToPage(tabNames[activeTab])
			renderTabs()
			return nil
		case event.Key() == tcell.KeyBacktab || (event.Key() == tcell.KeyLeft && event.Modifiers() == 0):
			activeTab = (activeTab + len(tabNames) - 1) % len(tabNames)
			tabPages.SwitchToPage(tabNames[activeTab])
			renderTabs()
			return nil
		case event.Key() == tcell.KeyRune:
			switch event.Rune() {
			case 'q':
				app.Stop()
				return nil
			case 'r':
				cancel()
				pages.SwitchToPage("config")
				return nil
			case 'e':
				dir, err := exportResults(cfg, results)
				if err != nil {
					footer.SetText("[red]  ✗ Export failed: " + err.Error())
				} else {
					footer.SetText("[green]  ✓ Exported to " + dir + "/[-]")
				}
				go func() {
					time.Sleep(4 * time.Second)
					app.QueueUpdateDraw(func() {
						footer.SetText("[gray]  tab/←/→: switch view   r: restart   e: export   q: quit")
					})
				}()
				return nil
			}
		}
		return event
	})

	return layout
}

// ── Summary Table ───────────────────────────────────────────────────────────

func buildSummaryTable(cfg *benchConfig, results map[string][]*bench.RunResult) *tview.Table {
	t := tview.NewTable().SetBorders(false).SetFixed(1, 1).SetSelectable(true, false)
	t.SetBorder(true).SetTitle(" Summary ").SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).SetBackgroundColor(colorBg)

	// Header
	t.SetCell(0, 0, headerCell("Metric"))
	for i, model := range cfg.Models {
		t.SetCell(0, i+1, headerCell(model))
	}

	// Warn about dropped failed runs
	warningRow := 0
	for i, model := range cfg.Models {
		total := len(results[model])
		valid := len(validResults(results[model]))
		if failed := total - valid; failed > 0 {
			if warningRow == 0 {
				warningRow = 1
				t.SetCell(1, 0, tview.NewTableCell("").SetSelectable(false))
			}
			t.SetCell(1, i+1, tview.NewTableCell(
				fmt.Sprintf("%d/%d runs failed", failed, total)).
				SetTextColor(colorYellow).SetAlign(tview.AlignCenter).SetSelectable(false))
		}
	}

	for row, metric := range bench.AllMetrics {
		row += warningRow
		t.SetCell(row+1, 0, tview.NewTableCell(metric.Label).
			SetTextColor(colorFg).SetSelectable(false))

		avgs := make(map[string]float64)
		cells := make(map[string]string)

		for _, model := range cfg.Models {
			valid := validResults(results[model])
			if len(valid) == 0 {
				cells[model] = "n/a"
				continue
			}
			vals := collectVals(valid, metric.Key)
			s := bench.Stats(vals)
			if math.IsNaN(s.Mean) {
				cells[model] = "n/a"
				continue
			}
			avgs[model] = s.Mean
			if s.N > 1 && s.StdDev > 0 {
				cells[model] = fmt.Sprintf("%s ±%s",
					bench.FmtVal(s.Mean, metric.Unit), bench.FmtVal(s.StdDev, metric.Unit))
			} else {
				cells[model] = bench.FmtVal(s.Mean, metric.Unit)
			}
		}

		best := findBestModel(avgs, metric.LowerIsBetter)
		for col, model := range cfg.Models {
			cell := tview.NewTableCell(cells[model]).SetAlign(tview.AlignRight)
			if model == best {
				cell.SetText("★ " + cells[model])
				cell.SetTextColor(colorGreen).SetAttributes(tcell.AttrBold)
			} else if best != "" && metric.LowerIsBetter != nil {
				// Not the best — show in dim/red depending on how much worse
				if bestVal, ok := avgs[best]; ok {
					if val, ok2 := avgs[model]; ok2 && bestVal > 0 {
						var pct float64
						if *metric.LowerIsBetter {
							pct = ((val - bestVal) / bestVal) * 100
						} else {
							pct = ((bestVal - val) / bestVal) * 100
						}
						if pct > 10 {
							cell.SetTextColor(colorRed)
						} else {
							cell.SetTextColor(colorYellow)
						}
					}
				}
			}
			t.SetCell(row+1, col+1, cell)
		}
	}

	return t
}

// ── Per-Run Table ───────────────────────────────────────────────────────────

func buildPerRunTable(cfg *benchConfig, results map[string][]*bench.RunResult) *tview.Table {
	t := tview.NewTable().SetBorders(false).SetFixed(1, 0).SetSelectable(true, false)
	t.SetBorder(true).SetTitle(" Per-Run Details ").SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).SetBackgroundColor(colorBg)

	metrics := bench.PerRunMetrics
	col := 0
	t.SetCell(0, col, headerCell("Model")); col++
	nRounds := cfg.Rounds
	if cfg.RoundRobin {
		nRounds = 1
	}
	if nRounds > 1 {
		t.SetCell(0, col, headerCell("Rnd")); col++
	}
	t.SetCell(0, col, headerCell("Run")); col++
	for _, m := range metrics {
		t.SetCell(0, col, headerCell(m.Label)); col++
	}

	row := 1
	for _, model := range cfg.Models {
		runs := results[model]
		curRound := 0
		runInRound := 0
		for _, r := range runs {
			if r == nil || r.Error != nil {
				continue
			}
			if r.Round != curRound {
				curRound = r.Round
				runInRound = 0
			}
			runInRound++

			col = 0
			t.SetCell(row, col, tview.NewTableCell(model).SetTextColor(colorPrimary)); col++
			if nRounds > 1 {
				t.SetCell(row, col, tview.NewTableCell(strconv.Itoa(r.Round)).SetAlign(tview.AlignCenter)); col++
			}
			t.SetCell(row, col, tview.NewTableCell(strconv.Itoa(runInRound)).SetAlign(tview.AlignCenter)); col++
			for _, m := range metrics {
				t.SetCell(row, col, tview.NewTableCell(bench.FmtVal(r.Get(m.Key), m.Unit)).
					SetAlign(tview.AlignRight)); col++
			}
			row++
		}
	}

	return t
}

// ── Relative Table ──────────────────────────────────────────────────────────

func buildRelativeTable(cfg *benchConfig, results map[string][]*bench.RunResult) *tview.Table {
	t := tview.NewTable().SetBorders(false).SetFixed(1, 1).SetSelectable(true, false)
	t.SetBorder(true).SetTitle(" Relative Performance ").SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).SetBackgroundColor(colorBg)

	if len(cfg.Models) < 2 {
		t.SetCell(0, 0, tview.NewTableCell("Need 2+ models for comparison").SetTextColor(colorDim))
		return t
	}

	t.SetCell(0, 0, headerCell("Metric"))
	for i, model := range cfg.Models {
		t.SetCell(0, i+1, headerCell(model))
	}

	for row, metric := range bench.RelativeMetrics {
		t.SetCell(row+1, 0, tview.NewTableCell(metric.Label).SetTextColor(colorFg).SetSelectable(false))

		avgs := make(map[string]float64)
		for _, model := range cfg.Models {
			valid := validResults(results[model])
			vals := collectVals(valid, metric.Key)
			s := bench.Stats(vals)
			if !math.IsNaN(s.Mean) {
				avgs[model] = s.Mean
			}
		}

		bestModel := findBestModel(avgs, metric.LowerIsBetter)
		bestVal := avgs[bestModel]

		for col, model := range cfg.Models {
			avg, ok := avgs[model]
			if !ok || bestVal == 0 {
				t.SetCell(row+1, col+1, tview.NewTableCell("n/a").SetTextColor(colorDim).SetAlign(tview.AlignRight))
				continue
			}

			var pct float64
			if metric.LowerIsBetter != nil && *metric.LowerIsBetter {
				pct = ((avg - bestVal) / bestVal) * 100
			} else {
				pct = ((bestVal - avg) / bestVal) * 100
			}

			cell := tview.NewTableCell("").SetAlign(tview.AlignRight)
			if math.Abs(pct) < 0.5 {
				cell.SetText("★ winner").SetTextColor(colorGreen).SetAttributes(tcell.AttrBold)
			} else {
				cell.SetText(fmt.Sprintf("+%.1f%%", pct))
				if pct > 30 {
					cell.SetTextColor(colorRed)
				} else if pct > 10 {
					cell.SetTextColor(colorYellow)
				}
			}
			t.SetCell(row+1, col+1, cell)
		}
	}

	return t
}

// ── Charts View ─────────────────────────────────────────────────────────────

func buildChartsView(cfg *benchConfig, results map[string][]*bench.RunResult) *tview.TextView {
	tv := tview.NewTextView().
		SetDynamicColors(true).
		SetScrollable(true)
	tv.SetBorder(true).SetTitle(" Charts ").SetTitleColor(colorPrimary).
		SetBorderColor(colorBorder).SetBackgroundColor(colorBg)

	var b strings.Builder
	barWidth := 45
	blocks := " ▏▎▍▌▋▊▉█"

	colorTags := []string{"purple", "orange", "skyblue", "yellow", "red", "teal"}

	for _, metric := range bench.ChartMetrics {
		avgs := make(map[string]float64)
		for _, model := range cfg.Models {
			valid := validResults(results[model])
			if len(valid) > 0 {
				vals := collectVals(valid, metric.Key)
				s := bench.Stats(vals)
				if !math.IsNaN(s.Mean) {
					avgs[model] = s.Mean
				}
			}
		}
		if len(avgs) == 0 {
			continue
		}

		maxVal := 0.0
		for _, v := range avgs {
			if v > maxVal {
				maxVal = v
			}
		}
		if maxVal == 0 {
			continue
		}

		best := findBestModel(avgs, metric.LowerIsBetter)

		direction := ""
		if metric.LowerIsBetter != nil {
			if *metric.LowerIsBetter {
				direction = "lower is better"
			} else {
				direction = "higher is better"
			}
		}

		b.WriteString(fmt.Sprintf("  [white::b]%s[-:-:-]  [gray]%s[-]\n\n", metric.Label, direction))

		maxNameLen := 0
		for _, model := range cfg.Models {
			if _, ok := avgs[model]; ok && len(model) > maxNameLen {
				maxNameLen = len(model)
			}
		}

		for i, model := range cfg.Models {
			val, ok := avgs[model]
			if !ok {
				continue
			}

			fraction := val / maxVal
			fullBlocks := int(fraction * float64(barWidth))
			remainder := (fraction * float64(barWidth)) - float64(fullBlocks)
			partialIdx := int(remainder * 8)

			bar := strings.Repeat("█", fullBlocks)
			if partialIdx > 0 && fullBlocks < barWidth {
				bar += string([]rune(blocks)[partialIdx])
			}
			for len([]rune(bar)) < barWidth {
				bar += " "
			}

			barColor := colorTags[i%len(colorTags)]

			name := model
			for len(name) < maxNameLen {
				name += " "
			}

			valStr := bench.FmtVal(val, metric.Unit)
			if model == best {
				b.WriteString(fmt.Sprintf("  %s  [%s]%s[-]  %s  ★\n", name, barColor, bar, valStr))
			} else {
				b.WriteString(fmt.Sprintf("  %s  [%s]%s[-]  %s\n", name, barColor, bar, valStr))
			}
		}
		b.WriteString("\n")
	}

	tv.SetText(b.String())
	return tv
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// ── Dry-run helpers ─────────────────────────────────────────────────────────

func fakeFetchModels() []bench.OllamaModel {
	return []bench.OllamaModel{
		{Name: "gemma4:e4b"},
		{Name: "gemma4:26b"},
		{Name: "llama3.2:3b"},
		{Name: "deepseek-r1:8b"},
		{Name: "phi3:latest"},
	}
}

func fakeBenchmarkOnce(model string) *bench.RunResult {
	time.Sleep(50 * time.Millisecond)

	base := map[string]float64{
		"gemma4:e4b":     100,
		"gemma4:26b":     55,
		"llama3.2:3b":    180,
		"deepseek-r1:8b": 90,
		"phi3:latest":    120,
	}
	tps := base[model]
	if tps == 0 {
		tps = 80
	}
	jitter := func(v, pct float64) float64 { return v * (1 + (rand.Float64()-0.5)*2*pct/100) }

	return &bench.RunResult{
		TTFT:            jitter(float64(200+rand.Intn(3000)), 20),
		EvalRate:        jitter(tps, 5),
		PromptEvalRate:  jitter(tps*1.8, 10),
		TotalDuration:   jitter(float64(3000+rand.Intn(5000)), 15),
		LoadDuration:    jitter(float64(100+rand.Intn(200)), 30),
		PromptEvalTime:  jitter(float64(50+rand.Intn(100)), 20),
		EvalTime:        jitter(float64(2000+rand.Intn(4000)), 15),
		PromptEvalCount: float64(14 + rand.Intn(20)),
		EvalCount:       float64(200 + rand.Intn(500)),
	}
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func headerCell(text string) *tview.TableCell {
	return tview.NewTableCell(text).
		SetTextColor(colorPrimary).
		SetAttributes(tcell.AttrBold).
		SetSelectable(false).
		SetAlign(tview.AlignCenter)
}

func validResults(runs []*bench.RunResult) []*bench.RunResult {
	var out []*bench.RunResult
	for _, r := range runs {
		if r != nil && r.Error == nil {
			out = append(out, r)
		}
	}
	return out
}

func collectVals(runs []*bench.RunResult, key string) []float64 {
	vals := make([]float64, len(runs))
	for i, r := range runs {
		vals[i] = r.Get(key)
	}
	return vals
}

func findBestModel(avgs map[string]float64, lowerIsBetter *bool) string {
	if lowerIsBetter == nil || len(avgs) == 0 {
		return ""
	}
	// Sort keys for deterministic tie-breaking (alphabetical)
	keys := make([]string, 0, len(avgs))
	for m := range avgs {
		keys = append(keys, m)
	}
	sort.Strings(keys)

	best := keys[0]
	bestVal := avgs[best]
	for _, m := range keys[1:] {
		v := avgs[m]
		if *lowerIsBetter && v < bestVal {
			best, bestVal = m, v
		} else if !*lowerIsBetter && v > bestVal {
			best, bestVal = m, v
		}
	}
	return best
}

// ── Export ───────────────────────────────────────────────────────────────────

func exportResults(cfg *benchConfig, results map[string][]*bench.RunResult) (string, error) {
	ts := time.Now().Format("2006-01-02-1504")
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("getting working directory: %w", err)
	}
	dir := fmt.Sprintf("%s/ollama-profiler-%s", cwd, ts)
	expCfg := export.Config{
		Models:     cfg.Models,
		Runs:       cfg.Runs,
		Rounds:     cfg.Rounds,
		RoundRobin: cfg.RoundRobin,
		Balanced:   cfg.Balanced,
		Warmup:     cfg.Warmup,
		Cooldown:   cfg.Cooldown,
		Prompt:     cfg.Prompt,
		NumPredict: cfg.NumPredict,
		Seed:       cfg.Seed,
		Think:      cfg.Think,
	}
	if err := export.Bundle(expCfg, results, dir); err != nil {
		return "", err
	}
	return dir, nil
}

