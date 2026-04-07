// Package export provides JSON, HTML, and PNG export for benchmark results.
package export

import (
	"encoding/json"
	"fmt"
	"html"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/piercecohen1/ollama-profiler/internal/bench"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gomono"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

// Config holds the metadata needed for exports.
type Config struct {
	Models     []string
	Runs       int
	Rounds     int
	RoundRobin bool
	Balanced   bool
	Warmup     bool
	Cooldown   int
	Prompt     string
	NumPredict int
	Seed       int
	Think      string
}

// Bundle creates a timestamped directory with JSON, HTML, and PNG exports.
func Bundle(cfg Config, results map[string][]*bench.RunResult, dir string) error {
	if dir == "" {
		ts := time.Now().Format("2006-01-02-1504")
		cwd, err := os.Getwd()
		if err != nil {
			return fmt.Errorf("getting working directory: %w", err)
		}
		dir = fmt.Sprintf("%s/ollama-profiler-%s", cwd, ts)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("creating export directory: %w", err)
	}
	if err := JSON(cfg, results, dir+"/results.json"); err != nil {
		return fmt.Errorf("exporting JSON: %w", err)
	}
	if err := HTML(cfg, results, dir+"/report.html"); err != nil {
		return fmt.Errorf("exporting HTML: %w", err)
	}
	if err := PNG(cfg, results, dir+"/charts.png"); err != nil {
		return fmt.Errorf("exporting PNG: %w", err)
	}
	return nil
}

// JSON exports raw results with metadata to the given path.
func JSON(cfg Config, results map[string][]*bench.RunResult, path string) error {
	schedule := scheduleStr(cfg)

	resultData := make(map[string][]map[string]interface{})
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
		resultData[model] = entries
	}

	export := map[string]interface{}{
		"meta": map[string]interface{}{
			"timestamp":    time.Now().Format(time.RFC3339),
			"models":       cfg.Models,
			"runs":         cfg.Runs,
			"rounds":       max(cfg.Rounds, 1),
			"schedule":     schedule,
			"warmup":       cfg.Warmup,
			"cooldown_sec": cfg.Cooldown,
			"prompt":       cfg.Prompt,
			"num_predict":  cfg.NumPredict,
			"seed":         cfg.Seed,
			"think":        cfg.Think,
		},
		"results": resultData,
	}
	data, err := json.MarshalIndent(export, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// HTML exports a self-contained dark-themed HTML report.
func HTML(cfg Config, results map[string][]*bench.RunResult, path string) error {
	models := cfg.Models
	ts := time.Now().Format("2006-01-02-1504")
	palette := []string{"#c9a0ff", "#f78166", "#79c0ff", "#d29922", "#f85149", "#a5d6ff"}

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
			Best:      findBest(avgs, metric.LowerIsBetter),
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
		best := findBest(avgs, metric.LowerIsBetter)
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
	for _, ch := range charts {
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
			clr := palette[mi%len(palette)]
			labelStyle := ""
			if model == ch.Best {
				labelStyle = " style='color:#3fb950;font-weight:bold'"
			}
			chartSections.WriteString(fmt.Sprintf(
				"<div class='bar-row'><span class='bar-label'%s>%s</span>"+
					"<div class='bar-track'><div class='bar-fill' style='width:%.1f%%;background:%s'></div></div>"+
					"<span class='bar-value'>%s</span></div>\n",
				labelStyle, html.EscapeString(model), pct, clr, bench.FmtVal(val, ch.Unit)))
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

<p class="footer">Generated by ollama-profiler</p>
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
	pngBg      = color.RGBA{13, 17, 23, 255}
	pngSurface = color.RGBA{22, 27, 34, 255}
	pngFg      = color.RGBA{230, 237, 243, 255}
	pngPrimary = color.RGBA{201, 160, 255, 255}
	pngGreen   = color.RGBA{63, 185, 80, 255}
	pngDim     = color.RGBA{72, 79, 88, 255}
	pngBarColors = []color.RGBA{
		{201, 160, 255, 255},
		{247, 129, 102, 255},
		{121, 192, 255, 255},
		{210, 153, 34, 255},
		{248, 81, 73, 255},
		{165, 214, 255, 255},
	}
)

// PNG exports benchmark charts as a PNG image.
func PNG(cfg Config, results map[string][]*bench.RunResult, path string) error {
	models := cfg.Models

	tt, err := opentype.Parse(gomono.TTF)
	if err != nil {
		return fmt.Errorf("parsing font: %w", err)
	}
	face, _ := opentype.NewFace(tt, &opentype.FaceOptions{Size: 13, DPI: 144})

	S := 2
	imgWidth := 700 * S
	padX := 30 * S
	barAreaX := 180 * S
	barHeight := 18 * S
	barGap := 6 * S
	sectionGap := 30 * S
	valueWidth := 100 * S
	charW := 8 * S

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
			best:      findBest(avgs, metric.LowerIsBetter),
		})
	}

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

	drawText(padX, 20*S, "Ollama Bench Results", pngPrimary)
	drawText(padX, 38*S, fmt.Sprintf("Generated %s  |  %d model(s)  |  %d run(s)/model",
		time.Now().Format("2006-01-02 15:04"), len(models), cfg.Runs*max(cfg.Rounds, 1)), pngDim)

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

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func scheduleStr(cfg Config) string {
	if cfg.RoundRobin {
		return "round-robin"
	}
	if cfg.Rounds > 1 {
		if cfg.Balanced {
			return "rounds-balanced"
		}
		return "rounds"
	}
	return "sequential"
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

func findBest(avgs map[string]float64, lowerIsBetter *bool) string {
	if lowerIsBetter == nil || len(avgs) == 0 {
		return ""
	}
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
