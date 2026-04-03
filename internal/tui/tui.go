// Package tui implements the interactive TUI for ollama-bench using tview.
package tui

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gdamore/tcell/v2"
	"github.com/piercecohen1/ollama-bench/internal/bench"
	"github.com/rivo/tview"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gomono"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
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
}

var dryRunMode bool

// Run starts the TUI application.
func Run(dryRun bool) error {
	dryRunMode = dryRun
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
		models, err = bench.FetchModels("http://localhost:11434")
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

	// Style the dropdown list for better contrast
	if dd, ok := form.GetFormItemByLabel("Schedule").(*tview.DropDown); ok {
		dd.SetListStyles(
			tcell.StyleDefault.Foreground(colorFg).Background(colorSurface),       // unselected
			tcell.StyleDefault.Foreground(colorBg).Background(colorPrimary),        // selected/highlighted
		)
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
			BaseURL:    "http://localhost:11434",
			Warmup:     warmup,
			RoundRobin: currentMode == 1,
			Rounds:     nRounds,
			Balanced:   currentMode == 3,
			Cooldown:   cooldown,
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
	results := make(map[string][]*bench.RunResult)
	ctx, cancel := context.WithCancel(context.Background())
	resultsShown := false

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
		total += len(cfg.Models)
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
		if resultsShown {
			return
		}
		resultsShown = true
		resultsPage := buildResultsPage(app, pages, cfg, results, cancel)
		pages.AddPage("results", resultsPage, true, true)
		pages.SwitchToPage("results")
	}

	// Run benchmarks in background
	go func() {
		nRounds := len(schedule)

		// Warmup
		if cfg.Warmup {
			for _, model := range cfg.Models {
				select {
				case <-ctx.Done():
					return
				default:
				}
				modelCopy := model
				app.QueueUpdateDraw(func() {
					statusView.SetText(fmt.Sprintf("[purple]  Warmup: %s", modelCopy))
				})
				var err error
				if dryRunMode {
					time.Sleep(50 * time.Millisecond)
				} else {
					_, err = bench.BenchmarkOnce(ctx, model, cfg.Prompt, cfg.BaseURL)
				}
				if ctx.Err() != nil {
					return
				}
				errCopy := err
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
		}

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

			for _, model := range sched {
				select {
				case <-ctx.Done():
					return
				default:
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
					res, err = bench.BenchmarkOnce(ctx, model, cfg.Prompt, cfg.BaseURL)
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
						addLogLine(fmt.Sprintf("[green]  ✓ %s R%d/run%d — %.1f tok/s",
							modelCopy, rndCopy, runCopy, resCopy.EvalRate))
						updateProgress()
					})
				}
			}
		}

		app.QueueUpdateDraw(func() {
			statusView.SetText("[green]  Complete!")
			resultsShown = false // allow rebuild with final data
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

	layout.SetInputCapture(func(event *tcell.EventKey) *tcell.EventKey {
		if event.Key() == tcell.KeyRune && event.Rune() == 'c' {
			showResults()
			return nil
		}
		return event
	})

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

	for row, metric := range bench.AllMetrics {
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

// ── Export ───────────────────────────────────────────────────────────────────

func exportResults(cfg *benchConfig, results map[string][]*bench.RunResult) (string, error) {
	ts := time.Now().Format("2006-01-02-1504")
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("getting working directory: %w", err)
	}
	dir := fmt.Sprintf("%s/ollama-bench-%s", cwd, ts)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", fmt.Errorf("creating export directory: %w", err)
	}
	if err := exportResultsJSON(cfg, results, dir); err != nil {
		return "", fmt.Errorf("exporting JSON: %w", err)
	}
	if err := exportResultsHTML(cfg, results, dir, ts); err != nil {
		return "", fmt.Errorf("exporting HTML: %w", err)
	}
	if err := exportResultsPNG(cfg, results, dir, ts); err != nil {
		return "", fmt.Errorf("exporting PNG: %w", err)
	}
	return dir, nil
}

func exportResultsJSON(cfg *benchConfig, results map[string][]*bench.RunResult, dir string) error {
	export := make(map[string][]map[string]interface{})
	for _, model := range cfg.Models {
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
	return os.WriteFile(dir+"/results.json", data, 0644)
}

func exportResultsHTML(cfg *benchConfig, results map[string][]*bench.RunResult, dir, ts string) error {
	path := dir + "/report.html"

	models := cfg.Models
	palette := []string{"#c9a0ff", "#f78166", "#79c0ff", "#d29922", "#f85149", "#a5d6ff"}

	// Compute averages for chart metrics
	type chartData struct {
		Label     string
		Unit      string
		Direction string
		Values    map[string]float64
		Best      string
	}
	var charts []chartData

	for _, metric := range bench.ChartMetrics {
		avgs := make(map[string]float64)
		for _, model := range models {
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
		dir := ""
		if metric.LowerIsBetter != nil {
			if *metric.LowerIsBetter {
				dir = "lower is better"
			} else {
				dir = "higher is better"
			}
		}
		charts = append(charts, chartData{
			Label:     metric.Label,
			Unit:      metric.Unit,
			Direction: dir,
			Values:    avgs,
			Best:      findBestModel(avgs, metric.LowerIsBetter),
		})
	}

	// Build summary table rows
	var summaryRows strings.Builder
	for _, metric := range bench.AllMetrics {
		avgs := make(map[string]float64)
		cells := make(map[string]string)
		for _, model := range models {
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
				cells[model] = fmt.Sprintf("%s ±%s", bench.FmtVal(s.Mean, metric.Unit), bench.FmtVal(s.StdDev, metric.Unit))
			} else {
				cells[model] = bench.FmtVal(s.Mean, metric.Unit)
			}
		}
		best := findBestModel(avgs, metric.LowerIsBetter)
		summaryRows.WriteString("<tr>")
		summaryRows.WriteString(fmt.Sprintf("<td class='metric'>%s</td>", metric.Label))
		for _, model := range models {
			cls := ""
			if model == best {
				cls = " class='best'"
			}
			summaryRows.WriteString(fmt.Sprintf("<td%s>%s</td>", cls, html.EscapeString(cells[model])))
		}
		summaryRows.WriteString("</tr>\n")
	}

	// Model header cells
	var modelHeaders strings.Builder
	for _, model := range models {
		modelHeaders.WriteString(fmt.Sprintf("<th>%s</th>", html.EscapeString(model)))
	}

	// Build chart HTML sections
	var chartSections strings.Builder
	for ci, ch := range charts {
		maxVal := 0.0
		for _, v := range ch.Values {
			if v > maxVal {
				maxVal = v
			}
		}
		chartSections.WriteString(fmt.Sprintf("<div class='chart'><h3>%s <span class='dir'>%s</span></h3>\n", ch.Label, ch.Direction))
		for mi, model := range models {
			val, ok := ch.Values[model]
			if !ok {
				continue
			}
			pct := 0.0
			if maxVal > 0 {
				pct = val / maxVal * 100
			}
			color := palette[mi%len(palette)]
			_ = ci
			labelStyle := ""
			if model == ch.Best {
				labelStyle = " style='color:#3fb950;font-weight:bold'"
			}
			chartSections.WriteString(fmt.Sprintf(
				"<div class='bar-row'><span class='bar-label'%s>%s</span>"+
					"<div class='bar-track'><div class='bar-fill' style='width:%.1f%%;background:%s'></div></div>"+
					"<span class='bar-value'>%s</span></div>\n",
				labelStyle, html.EscapeString(model), pct, color, bench.FmtVal(val, ch.Unit)))
		}
		chartSections.WriteString("</div>\n")
	}

	htmlContent := fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ollama Bench Results — %s</title>
<style>
  :root { --bg: #0d1117; --surface: #161b22; --panel: #21262d; --border: #30363d;
          --fg: #e6edf3; --dim: #484f58; --primary: #c9a0ff; --green: #3fb950;
          --red: #f85149; --yellow: #d29922; }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--fg); font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace; padding: 2rem; max-width: 1000px; margin: 0 auto; }
  h1 { color: var(--primary); margin-bottom: 0.5rem; font-size: 1.4rem; }
  h2 { color: var(--primary); margin: 2rem 0 1rem; font-size: 1.1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
  h3 { color: var(--fg); font-size: 0.95rem; margin-bottom: 0.8rem; }
  .dir { color: var(--dim); font-weight: normal; font-size: 0.8rem; }
  .meta { color: var(--dim); font-size: 0.85rem; margin-bottom: 2rem; }
  table { border-collapse: collapse; width: 100%%; margin-bottom: 1rem; }
  th { text-align: center; color: var(--primary); padding: 0.5rem 1rem; border-bottom: 2px solid var(--border); }
  td { text-align: right; padding: 0.4rem 1rem; border-bottom: 1px solid var(--border); }
  td.metric { text-align: left; color: var(--fg); font-weight: 600; }
  td.best { color: var(--green); font-weight: bold; }
  .chart { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.2rem; margin-bottom: 1.2rem; }
  .bar-row { display: flex; align-items: center; margin: 0.4rem 0; }
  .bar-label { width: 160px; font-size: 0.85rem; flex-shrink: 0; }
  .bar-track { flex: 1; height: 22px; background: var(--panel); border-radius: 4px; overflow: hidden; margin: 0 0.8rem; }
  .bar-fill { height: 100%%; border-radius: 4px; transition: width 0.3s; }
  .bar-value { width: 120px; text-align: right; font-size: 0.85rem; flex-shrink: 0; }
  .footer { margin-top: 3rem; color: var(--dim); font-size: 0.75rem; text-align: center; }
</style>
</head>
<body>
<h1>Ollama Bench Results</h1>
<p class="meta">Generated %s · %d model(s) · %d run(s)/model</p>

<h2>Summary</h2>
<table>
<tr><th style="text-align:left">Metric</th>%s</tr>
%s</table>

<h2>Performance Charts</h2>
%s

<p class="footer">Generated by ollama-bench</p>
</body>
</html>`,
		ts,
		time.Now().Format("2006-01-02 15:04"),
		len(models),
		cfg.Runs*max(cfg.Rounds, 1),
		modelHeaders.String(),
		summaryRows.String(),
		chartSections.String(),
	)

	return os.WriteFile(path, []byte(htmlContent), 0644)
}

// ── PNG Export ───────────────────────────────────────────────────────────────

var (
	pngBg      = color.RGBA{13, 17, 23, 255}    // #0d1117
	pngSurface = color.RGBA{22, 27, 34, 255}     // #161b22
	pngBorder  = color.RGBA{48, 54, 61, 255}     // #30363d
	pngFg      = color.RGBA{230, 237, 243, 255}  // #e6edf3
	pngPrimary = color.RGBA{201, 160, 255, 255}  // #c9a0ff
	pngGreen   = color.RGBA{63, 185, 80, 255}    // #3fb950
	pngDim     = color.RGBA{72, 79, 88, 255}     // #484f58
	pngBarColors = []color.RGBA{
		{201, 160, 255, 255}, // primary
		{247, 129, 102, 255}, // orange
		{121, 192, 255, 255}, // blue
		{210, 153, 34, 255},  // yellow
		{248, 81, 73, 255},   // red
		{165, 214, 255, 255}, // light blue
	}
)

func exportResultsPNG(cfg *benchConfig, results map[string][]*bench.RunResult, dir, ts string) error {
	models := cfg.Models

	// Load Go Mono font (embedded, works on all platforms)
	tt, err := opentype.Parse(gomono.TTF)
	if err != nil {
		return fmt.Errorf("parsing font: %w", err)
	}
	face, _ := opentype.NewFace(tt, &opentype.FaceOptions{Size: 13, DPI: 144})

	// Scale factor: 2x the original layout for retina quality
	S := 2
	imgWidth := 700 * S
	padX := 30 * S
	barAreaX := 180 * S
	barHeight := 18 * S
	barGap := 6 * S
	sectionGap := 30 * S
	valueWidth := 100 * S
	charW := 8 * S // approximate character width at this font size

	// Compute chart data
	type chartEntry struct {
		label, unit, direction string
		avgs                   map[string]float64
		best                   string
	}
	var charts []chartEntry
	for _, metric := range bench.ChartMetrics {
		avgs := make(map[string]float64)
		for _, model := range models {
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
		metricDir := ""
		if metric.LowerIsBetter != nil {
			if *metric.LowerIsBetter {
				metricDir = "lower is better"
			} else {
				metricDir = "higher is better"
			}
		}
		charts = append(charts, chartEntry{
			label:     metric.Label,
			unit:      metric.Unit,
			direction: metricDir,
			avgs:      avgs,
			best:      findBestModel(avgs, metric.LowerIsBetter),
		})
	}

	// Calculate image height
	titleHeight := 55 * S
	chartHeight := 0
	for _, ch := range charts {
		n := 0
		for _, m := range models {
			if _, ok := ch.avgs[m]; ok {
				n++
			}
		}
		chartHeight += 20*S + n*(barHeight+barGap) + sectionGap
	}
	imgHeight := titleHeight + chartHeight + 20*S

	// Create image
	img := image.NewRGBA(image.Rect(0, 0, imgWidth, imgHeight))
	draw.Draw(img, img.Bounds(), &image.Uniform{pngBg}, image.Point{}, draw.Src)

	drawText := func(x, y int, text string, col color.RGBA) {
		d := &font.Drawer{
			Dst:  img,
			Src:  &image.Uniform{col},
			Face: face,
			Dot:  fixed.P(x, y),
		}
		d.DrawString(text)
	}

	fillRect := func(x, y, w, h int, col color.RGBA) {
		draw.Draw(img,
			image.Rect(x, y, x+w, y+h),
			&image.Uniform{col},
			image.Point{}, draw.Src)
	}

	// Title
	drawText(padX, 20*S, "Ollama Bench Results", pngPrimary)
	drawText(padX, 38*S, fmt.Sprintf("Generated %s  |  %d model(s)  |  %d run(s)/model",
		time.Now().Format("2006-01-02 15:04"), len(models), cfg.Runs*max(cfg.Rounds, 1)), pngDim)

	// Charts
	y := titleHeight
	barMaxWidth := imgWidth - barAreaX - padX - valueWidth

	for _, ch := range charts {
		drawText(padX, y+13*S, ch.label, pngFg)
		drawText(padX+len(ch.label)*charW+10*S, y+13*S, ch.direction, pngDim)
		y += 22 * S

		maxVal := 0.0
		for _, v := range ch.avgs {
			if v > maxVal {
				maxVal = v
			}
		}

		for mi, model := range models {
			val, ok := ch.avgs[model]
			if !ok {
				continue
			}

			nameColor := pngFg
			if model == ch.best {
				nameColor = pngGreen
			}
			drawText(padX, y+13*S, model, nameColor)
			fillRect(barAreaX, y+2*S, barMaxWidth, barHeight, pngSurface)

			barW := 0
			if maxVal > 0 {
				barW = int(val / maxVal * float64(barMaxWidth))
			}
			if barW < 2 {
				barW = 2
			}
			barColor := pngBarColors[mi%len(pngBarColors)]
			fillRect(barAreaX, y+2*S, barW, barHeight, barColor)

			valStr := bench.FmtVal(val, ch.unit)
			drawText(barAreaX+barMaxWidth+8*S, y+13*S, valStr, pngFg)

			y += barHeight + barGap
		}
		y += sectionGap - barGap
	}

	f, err := os.Create(dir + "/charts.png")
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
