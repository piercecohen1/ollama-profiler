// Package bench provides the core benchmarking engine for ollama-profiler.
package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// ── Metrics ─────────────────────────────────────────────────────────────────

// Metric describes a measurable benchmark value.
type Metric struct {
	Key           string
	Label         string
	Unit          string
	LowerIsBetter *bool // nil = neutral
}

var (
	trueVal  = true
	falseVal = false
)

// AllMetrics is the full list of tracked metrics.
var AllMetrics = []Metric{
	{"ttft_ms", "TTFT", "ms", &trueVal},
	{"eval_rate", "Generation TPS", "tok/s", &falseVal},
	{"prompt_eval_rate", "Prompt Eval TPS", "tok/s", &falseVal},
	{"total_duration_ms", "Total Duration", "ms", &trueVal},
	{"load_duration_ms", "Load Duration", "ms", &trueVal},
	{"prompt_eval_duration_ms", "Prompt Eval Time", "ms", &trueVal},
	{"eval_duration_ms", "Eval Time", "ms", &trueVal},
	{"prompt_eval_count", "Prompt Tokens", "tok", nil},
	{"eval_count", "Generated Tokens", "tok", nil},
}

// PerRunMetrics is a subset shown in per-run detail tables.
var PerRunMetrics = []Metric{
	{"ttft_ms", "TTFT", "ms", &trueVal},
	{"eval_rate", "Gen TPS", "tok/s", &falseVal},
	{"prompt_eval_rate", "Prompt TPS", "tok/s", &falseVal},
	{"total_duration_ms", "Total", "ms", &trueVal},
	{"load_duration_ms", "Load", "ms", &trueVal},
	{"eval_duration_ms", "Eval Time", "ms", &trueVal},
	{"eval_count", "Tokens", "tok", nil},
}

// RelativeMetrics is the subset shown in relative comparison.
var RelativeMetrics = []Metric{
	{"ttft_ms", "TTFT", "ms", &trueVal},
	{"eval_rate", "Generation TPS", "tok/s", &falseVal},
	{"prompt_eval_rate", "Prompt Eval TPS", "tok/s", &falseVal},
	{"total_duration_ms", "Total Duration", "ms", &trueVal},
}

// ChartMetrics is the subset shown in the charts tab.
var ChartMetrics = []Metric{
	{"ttft_ms", "TTFT", "ms", &trueVal},
	{"eval_rate", "Generation TPS", "tok/s", &falseVal},
	{"prompt_eval_rate", "Prompt Eval TPS", "tok/s", &falseVal},
	{"total_duration_ms", "Total Duration", "ms", &trueVal},
	{"eval_duration_ms", "Eval Time", "ms", &trueVal},
	{"eval_count", "Generated Tokens", "tok", nil},
}

// ── RunResult ───────────────────────────────────────────────────────────────

// RunResult holds the metrics from a single benchmark run.
type RunResult struct {
	TTFT            float64 // wall-clock time to first token (ms)
	EvalRate        float64 // tokens/sec generation
	PromptEvalRate  float64 // tokens/sec prompt processing
	TotalDuration   float64 // total duration (ms)
	LoadDuration    float64 // model load duration (ms)
	PromptEvalTime  float64 // prompt eval duration (ms)
	EvalTime        float64 // generation duration (ms)
	PromptEvalCount float64 // prompt token count
	EvalCount       float64 // generated token count
	Round           int     // which round this run belongs to
	Error           error   // non-nil if the run failed
}

// Get returns a metric value by key.
func (r *RunResult) Get(key string) float64 {
	switch key {
	case "ttft_ms":
		return r.TTFT
	case "eval_rate":
		return r.EvalRate
	case "prompt_eval_rate":
		return r.PromptEvalRate
	case "total_duration_ms":
		return r.TotalDuration
	case "load_duration_ms":
		return r.LoadDuration
	case "prompt_eval_duration_ms":
		return r.PromptEvalTime
	case "eval_duration_ms":
		return r.EvalTime
	case "prompt_eval_count":
		return r.PromptEvalCount
	case "eval_count":
		return r.EvalCount
	default:
		return math.NaN()
	}
}

// ── Formatting ──────────────────────────────────────────────────────────────

// FmtVal formats a metric value for display.
func FmtVal(val float64, unit string) string {
	if math.IsNaN(val) || math.IsInf(val, 0) {
		return "n/a"
	}
	switch unit {
	case "tok":
		return fmt.Sprintf("%d", int(val))
	case "tok/s":
		return fmt.Sprintf("%.1f", val)
	case "ms":
		if val >= 10000 {
			return fmt.Sprintf("%.2fs", val/1000)
		}
		return fmt.Sprintf("%.1fms", val)
	default:
		return fmt.Sprintf("%.2f", val)
	}
}

// ── Statistics ──────────────────────────────────────────────────────────────

// StatResult holds computed statistics.
type StatResult struct {
	Mean   float64
	StdDev float64
	Min    float64
	Max    float64
	N      int
}

// Stats computes mean, stddev, min, max over values, filtering NaN.
func Stats(values []float64) StatResult {
	var clean []float64
	for _, v := range values {
		if !math.IsNaN(v) {
			clean = append(clean, v)
		}
	}
	if len(clean) == 0 {
		return StatResult{
			Mean:   math.NaN(),
			StdDev: math.NaN(),
			Min:    math.NaN(),
			Max:    math.NaN(),
		}
	}

	sum := 0.0
	mn, mx := clean[0], clean[0]
	for _, v := range clean {
		sum += v
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	mean := sum / float64(len(clean))

	sd := 0.0
	if len(clean) > 1 {
		variance := 0.0
		for _, v := range clean {
			d := v - mean
			variance += d * d
		}
		sd = math.Sqrt(variance / float64(len(clean)-1))
	}

	return StatResult{Mean: mean, StdDev: sd, Min: mn, Max: mx, N: len(clean)}
}

// ── Scheduling ──────────────────────────────────────────────────────────────

// ScheduleConfig controls how benchmark runs are ordered.
type ScheduleConfig struct {
	RoundRobin bool
	Rounds     int  // number of rounds (0 or 1 = single round)
	Balanced   bool // use Latin-square rotation for positional fairness
}

// BuildSchedule generates the execution order as a list of rounds.
// Each round is a slice of model names in the order they should be run.
func BuildSchedule(models []string, runsPerModel int, cfg ScheduleConfig) [][]string {
	if cfg.RoundRobin {
		round := make([]string, 0, len(models)*runsPerModel)
		for i := 0; i < runsPerModel; i++ {
			round = append(round, models...)
		}
		return [][]string{round}
	}

	nRounds := cfg.Rounds
	if nRounds <= 1 {
		nRounds = 1
	}
	if len(models) == 0 {
		rounds := make([][]string, nRounds)
		for i := range rounds {
			rounds[i] = []string{}
		}
		return rounds
	}

	if nRounds > 1 {
		rounds := make([][]string, nRounds)
		if cfg.Balanced {
			// Shuffle base order, then rotate for Latin-square.
			// Simple modular rotation (r % len) gives clean cycling
			// even when nRounds > len(models).
			base := make([]string, len(models))
			copy(base, models)
			rand.Shuffle(len(base), func(i, j int) { base[i], base[j] = base[j], base[i] })

			for r := 0; r < nRounds; r++ {
				offset := r % len(base)
				order := make([]string, len(base))
				for i := range base {
					order[i] = base[(i+offset)%len(base)]
				}
				round := make([]string, 0, len(order)*runsPerModel)
				for _, m := range order {
					for j := 0; j < runsPerModel; j++ {
						round = append(round, m)
					}
				}
				rounds[r] = round
			}
		} else {
			for r := 0; r < nRounds; r++ {
				order := make([]string, len(models))
				copy(order, models)
				rand.Shuffle(len(order), func(i, j int) { order[i], order[j] = order[j], order[i] })
				round := make([]string, 0, len(order)*runsPerModel)
				for _, m := range order {
					for j := 0; j < runsPerModel; j++ {
						round = append(round, m)
					}
				}
				rounds[r] = round
			}
		}
		return rounds
	}

	// Sequential (default): all runs of A, then B, then C
	round := make([]string, 0, len(models)*runsPerModel)
	for _, m := range models {
		for j := 0; j < runsPerModel; j++ {
			round = append(round, m)
		}
	}
	return [][]string{round}
}

// ── Base URL resolution ─────────────────────────────────────────────────────

// ResolveBaseURL returns the Ollama API base URL, preferring:
//  1. explicit (if non-empty) — typically from a --url flag
//  2. OLLAMA_HOST environment variable
//  3. http://localhost:11434 (default)
//
// The returned URL always has a scheme and no trailing slash.
func ResolveBaseURL(explicit string) string {
	if explicit != "" {
		return normalizeHost(explicit)
	}
	if envHost := os.Getenv("OLLAMA_HOST"); envHost != "" {
		return normalizeHost(envHost)
	}
	return "http://localhost:11434"
}

// normalizeHost mirrors Ollama's own Host()/ConnectableHost() semantics:
//   - no scheme → scheme "http", default port 11434
//   - "http://" → default port 80; "https://" → default port 443
//   - scheme matching is case-insensitive
//   - bare ":port" resolves to http://localhost:port
//   - unspecified bind addresses (0.0.0.0, ::) are rewritten to localhost for client use
//   - IPv6 addresses and URL paths are preserved
//
// See: github.com/ollama/ollama/envconfig/config.go — Host(), ConnectableHost()
func normalizeHost(s string) string {
	const ollamaDefaultPort = "11434"
	s = strings.TrimSpace(s)
	if s == "" {
		return "http://localhost:" + ollamaDefaultPort
	}

	// Extract scheme (case-insensitive). Default port depends on scheme.
	scheme := "http"
	defaultPort := ollamaDefaultPort
	rest := s
	if i := strings.Index(s, "://"); i >= 0 {
		scheme = strings.ToLower(s[:i])
		rest = s[i+3:]
		switch scheme {
		case "http":
			defaultPort = "80"
		case "https":
			defaultPort = "443"
		}
	}

	// Separate host:port from path.
	var path string
	if i := strings.Index(rest, "/"); i >= 0 {
		path = strings.TrimRight(rest[i:], "/")
		rest = rest[:i]
	}

	// Parse host and port.
	host, port, err := net.SplitHostPort(rest)
	if err != nil {
		switch {
		case strings.HasPrefix(rest, ":"):
			host, port = "localhost", strings.TrimPrefix(rest, ":")
		case rest == "":
			host, port = "localhost", defaultPort
		default:
			host, port = strings.Trim(rest, "[]"), defaultPort
		}
	}

	// Rewrite unspecified bind addresses to loopback for client connections.
	switch host {
	case "0.0.0.0", "::", "":
		host = "localhost"
	}

	return scheme + "://" + net.JoinHostPort(host, port) + path
}

// ── Benchmark execution ─────────────────────────────────────────────────────

// ollamaChunk is a single JSON line from the Ollama streaming response.
type ollamaChunk struct {
	Response           string `json:"response"`
	Thinking           string `json:"thinking"`
	Done               bool   `json:"done"`
	TotalDuration      int64  `json:"total_duration"`
	LoadDuration       int64  `json:"load_duration"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalCount          int    `json:"eval_count"`
	EvalDuration       int64  `json:"eval_duration"`
}

// BenchmarkOpts controls per-request Ollama parameters for reproducible benchmarking.
type BenchmarkOpts struct {
	NumPredict int    // >0: limit generated tokens, 0: unlimited (-1 to Ollama)
	Seed       int    // >0: fixed seed for determinism, 0: random
	Think      string // "": disabled, "true": enabled, "low"/"medium"/"high": thinking level
}

// OllamaConnectionError wraps connection failures with a user-friendly message.
// For local servers, it distinguishes between Ollama not being installed vs not running.
func OllamaConnectionError(baseURL string, err error) error {
	var opErr *net.OpError
	if !errors.As(err, &opErr) {
		return err
	}

	// Only check local install status when targeting localhost.
	if isLocalURL(baseURL) {
		if _, lookErr := exec.LookPath("ollama"); lookErr != nil {
			var installCmd string
			switch runtime.GOOS {
			case "windows":
				installCmd = "irm https://ollama.com/install.ps1 | iex"
			default: // macOS, Linux
				installCmd = "curl -fsSL https://ollama.com/install.sh | sh"
			}
			return fmt.Errorf("Ollama is not installed.\n\n  Install it with:\n\n    %s\n\n  Then start the server with: ollama serve", installCmd)
		}
	}

	return fmt.Errorf("could not connect to Ollama at %s\n\n  Is Ollama running? Start it with: ollama serve\n\n  To use a different server: --url <host:port> or set OLLAMA_HOST", baseURL)
}

// isLocalURL returns true if the URL targets localhost or a loopback address.
func isLocalURL(baseURL string) bool {
	host := baseURL
	// Strip scheme.
	if i := strings.Index(host, "://"); i >= 0 {
		host = host[i+3:]
	}
	// Strip path.
	if i := strings.Index(host, "/"); i >= 0 {
		host = host[:i]
	}
	// Strip port.
	if i := strings.LastIndex(host, ":"); i >= 0 {
		host = host[:i]
	}
	// Strip IPv6 brackets.
	host = strings.TrimPrefix(host, "[")
	host = strings.TrimSuffix(host, "]")

	switch host {
	case "localhost", "127.0.0.1", "::1":
		return true
	}
	return false
}

// BenchmarkOnce runs a single inference pass against a model and returns metrics.
func BenchmarkOnce(ctx context.Context, model, prompt, baseURL string, opts BenchmarkOpts) (*RunResult, error) {
	body := map[string]interface{}{
		"model":  model,
		"prompt": prompt,
	}
	switch opts.Think {
	case "true":
		body["think"] = true
	case "low", "medium", "high":
		body["think"] = opts.Think
	default:
		body["think"] = false
	}
	options := map[string]interface{}{}
	if opts.NumPredict > 0 {
		options["num_predict"] = opts.NumPredict
	} else if opts.NumPredict == 0 {
		options["num_predict"] = -1 // Ollama uses -1 for unlimited
	}
	if opts.Seed > 0 {
		options["seed"] = opts.Seed
	}
	if len(options) > 0 {
		body["options"] = options
	}
	bodyBytes, _ := json.Marshal(body)

	req, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/api/generate", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Minute}
	start := time.Now()

	resp, err := client.Do(req)
	if err != nil {
		return nil, OllamaConnectionError(baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		var errResp struct {
			Error string `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error != "" {
			return nil, fmt.Errorf("Ollama error (status %d): %s", resp.StatusCode, errResp.Error)
		}
		return nil, fmt.Errorf("Ollama returned status %d", resp.StatusCode)
	}

	decoder := json.NewDecoder(resp.Body)
	var firstTokenTime time.Time
	var final ollamaChunk
	gotFinal := false

	for {
		var chunk ollamaChunk
		err := decoder.Decode(&chunk)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("decoding response from %s: %w", model, err)
		}
		if firstTokenTime.IsZero() && (chunk.Response != "" || chunk.Thinking != "") {
			firstTokenTime = time.Now()
		}
		if chunk.Done {
			final = chunk
			gotFinal = true
			break
		}
	}

	if !gotFinal {
		return nil, fmt.Errorf("no final chunk received from %s", model)
	}

	ttft := math.NaN()
	if !firstTokenTime.IsZero() {
		ttft = float64(firstTokenTime.Sub(start).Milliseconds())
	}

	pRate := math.NaN()
	if final.PromptEvalDuration > 0 {
		pRate = float64(final.PromptEvalCount) * 1e9 / float64(final.PromptEvalDuration)
	}
	eRate := math.NaN()
	if final.EvalDuration > 0 {
		eRate = float64(final.EvalCount) * 1e9 / float64(final.EvalDuration)
	}

	return &RunResult{
		TTFT:            ttft,
		EvalRate:        eRate,
		PromptEvalRate:  pRate,
		TotalDuration:   float64(final.TotalDuration) / 1e6,
		LoadDuration:    float64(final.LoadDuration) / 1e6,
		PromptEvalTime:  float64(final.PromptEvalDuration) / 1e6,
		EvalTime:        float64(final.EvalDuration) / 1e6,
		PromptEvalCount: float64(final.PromptEvalCount),
		EvalCount:       float64(final.EvalCount),
	}, nil
}

// ── Ollama model listing ────────────────────────────────────────────────────

// OllamaModel represents a model available in Ollama.
type OllamaModel struct {
	Name    string `json:"name"`
	Size    int64  `json:"size"`
	Details struct {
		ParameterSize     string `json:"parameter_size"`
		QuantizationLevel string `json:"quantization_level"`
	} `json:"details"`
}

// FetchModels retrieves available models from the Ollama API.
func FetchModels(baseURL string) ([]OllamaModel, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(baseURL + "/api/tags")
	if err != nil {
		return nil, OllamaConnectionError(baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		var errResp struct {
			Error string `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&errResp); err == nil && errResp.Error != "" {
			return nil, fmt.Errorf("Ollama error (status %d): %s", resp.StatusCode, errResp.Error)
		}
		return nil, fmt.Errorf("Ollama returned status %d for /api/tags", resp.StatusCode)
	}

	var result struct {
		Models []OllamaModel `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return result.Models, nil
}

// FormatModelLabel creates a display label for a model.
func FormatModelLabel(m OllamaModel) string {
	label := m.Name
	if m.Details.ParameterSize != "" {
		label += " — " + m.Details.ParameterSize
	}
	if m.Details.QuantizationLevel != "" {
		label += " " + m.Details.QuantizationLevel
	}
	if m.Size > 0 {
		label += fmt.Sprintf(" (%.1f GB)", float64(m.Size)/1e9)
	}
	return label
}
