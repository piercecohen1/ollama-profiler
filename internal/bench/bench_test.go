package bench

import (
	"math"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// ── ResolveBaseURL tests ────────────────────────────────────────────────────

func TestResolveBaseURL(t *testing.T) {
	tests := []struct {
		name     string
		explicit string
		envHost  string
		want     string
	}{
		{"default", "", "", "http://localhost:11434"},
		{"explicit full URL", "http://example.com:8080", "", "http://example.com:8080"},
		{"https default port", "https://api.example.com", "", "https://api.example.com:443"},
		{"http default port", "http://remotebox", "", "http://remotebox:80"},
		{"plain host no scheme gets 11434", "remotebox", "", "http://remotebox:11434"},
		{"explicit host:port", "gpu-box:11434", "", "http://gpu-box:11434"},
		{"explicit bare port", ":9999", "", "http://localhost:9999"},
		{"trailing slash trimmed", "http://host:11434/", "", "http://host:11434"},
		{"url with path preserved", "http://host:11434/api/v1", "", "http://host:11434/api/v1"},
		{"uppercase scheme lowercased", "HTTP://host:8080", "", "http://host:8080"},
		{"unspecified ipv4 rewritten to localhost", "0.0.0.0:11434", "", "http://localhost:11434"},
		{"unspecified ipv6 rewritten to localhost", "[::]:11434", "", "http://localhost:11434"},
		{"ipv6 loopback preserved", "[::1]:11434", "", "http://[::1]:11434"},
		{"env var fallback", "", "remote:11434", "http://remote:11434"},
		{"env var full URL", "", "http://remote:11434", "http://remote:11434"},
		{"env var bare port", "", ":12345", "http://localhost:12345"},
		{"env var 0.0.0.0 rewritten", "", "0.0.0.0:11434", "http://localhost:11434"},
		{"env var plain host", "", "gpu-server", "http://gpu-server:11434"},
		{"explicit overrides env", "http://explicit:1111", "http://envvar:2222", "http://explicit:1111"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.envHost)
			got := ResolveBaseURL(tt.explicit)
			if got != tt.want {
				t.Errorf("ResolveBaseURL(%q) with OLLAMA_HOST=%q = %q, want %q",
					tt.explicit, tt.envHost, got, tt.want)
			}
		})
	}
}

// ── FmtVal tests ────────────────────────────────────────────────────────────

func TestFmtVal(t *testing.T) {
	tests := []struct {
		name string
		val  float64
		unit string
		want string
	}{
		{"ms small", 42.3, "ms", "42.3ms"},
		{"ms large", 12345.6, "ms", "12.35s"},
		{"tokens per sec", 99.05, "tok/s", "99.0"},
		{"tokens", 552, "tok", "552"},
		{"nan", math.NaN(), "ms", "n/a"},
		{"inf", math.Inf(1), "ms", "n/a"},
		{"zero ms", 0, "ms", "0.0ms"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FmtVal(tt.val, tt.unit)
			if got != tt.want {
				t.Errorf("FmtVal(%v, %q) = %q, want %q", tt.val, tt.unit, got, tt.want)
			}
		})
	}
}

// ── Stats tests ─────────────────────────────────────────────────────────────

func TestStats(t *testing.T) {
	t.Run("basic", func(t *testing.T) {
		s := Stats([]float64{10, 20, 30})
		if s.Mean != 20 {
			t.Errorf("mean = %v, want 20", s.Mean)
		}
		if s.Min != 10 {
			t.Errorf("min = %v, want 10", s.Min)
		}
		if s.Max != 30 {
			t.Errorf("max = %v, want 30", s.Max)
		}
		if s.StdDev == 0 {
			t.Error("stddev should be > 0")
		}
	})

	t.Run("single value", func(t *testing.T) {
		s := Stats([]float64{42})
		if s.Mean != 42 {
			t.Errorf("mean = %v, want 42", s.Mean)
		}
		if s.StdDev != 0 {
			t.Errorf("stddev = %v, want 0", s.StdDev)
		}
	})

	t.Run("with NaN", func(t *testing.T) {
		s := Stats([]float64{10, math.NaN(), 30})
		if s.Mean != 20 {
			t.Errorf("mean = %v, want 20", s.Mean)
		}
	})

	t.Run("all NaN", func(t *testing.T) {
		s := Stats([]float64{math.NaN(), math.NaN()})
		if !math.IsNaN(s.Mean) {
			t.Errorf("mean = %v, want NaN", s.Mean)
		}
	})

	t.Run("empty", func(t *testing.T) {
		s := Stats([]float64{})
		if !math.IsNaN(s.Mean) {
			t.Errorf("mean = %v, want NaN", s.Mean)
		}
	})
}

// ── Schedule tests ──────────────────────────────────────────────────────────

func TestBuildSchedule_Sequential(t *testing.T) {
	models := []string{"A", "B", "C"}
	sched := BuildSchedule(models, 3, ScheduleConfig{})

	if len(sched) != 1 {
		t.Fatalf("expected 1 round, got %d", len(sched))
	}
	// Should be A,A,A,B,B,B,C,C,C
	want := []string{"A", "A", "A", "B", "B", "B", "C", "C", "C"}
	if len(sched[0]) != len(want) {
		t.Fatalf("round 0 length = %d, want %d", len(sched[0]), len(want))
	}
	for i, m := range sched[0] {
		if m != want[i] {
			t.Errorf("sched[0][%d] = %q, want %q", i, m, want[i])
		}
	}
}

func TestBuildSchedule_RoundRobin(t *testing.T) {
	models := []string{"A", "B"}
	sched := BuildSchedule(models, 3, ScheduleConfig{RoundRobin: true})

	if len(sched) != 1 {
		t.Fatalf("expected 1 round, got %d", len(sched))
	}
	// Should be A,B,A,B,A,B
	want := []string{"A", "B", "A", "B", "A", "B"}
	if len(sched[0]) != len(want) {
		t.Fatalf("round 0 length = %d, want %d", len(sched[0]), len(want))
	}
	for i, m := range sched[0] {
		if m != want[i] {
			t.Errorf("sched[0][%d] = %q, want %q", i, m, want[i])
		}
	}
}

func TestBuildSchedule_Rounds(t *testing.T) {
	models := []string{"A", "B", "C"}
	sched := BuildSchedule(models, 2, ScheduleConfig{Rounds: 3})

	if len(sched) != 3 {
		t.Fatalf("expected 3 rounds, got %d", len(sched))
	}
	// Each round should have 6 entries (2 per model)
	for i, round := range sched {
		if len(round) != 6 {
			t.Errorf("round %d length = %d, want 6", i, len(round))
		}
		// Each model should appear exactly 2 times
		counts := map[string]int{}
		for _, m := range round {
			counts[m]++
		}
		for _, m := range models {
			if counts[m] != 2 {
				t.Errorf("round %d: model %q appears %d times, want 2", i, m, counts[m])
			}
		}
	}
}

func TestBuildSchedule_Balanced(t *testing.T) {
	models := []string{"A", "B", "C"}
	sched := BuildSchedule(models, 1, ScheduleConfig{Rounds: 3, Balanced: true})

	if len(sched) != 3 {
		t.Fatalf("expected 3 rounds, got %d", len(sched))
	}

	// Track which model is first in each round
	firstPositions := map[string]int{}
	for _, round := range sched {
		firstPositions[round[0]]++
	}

	// With 3 models and 3 rounds, each should be first exactly once
	for _, m := range models {
		if firstPositions[m] != 1 {
			t.Errorf("model %q was first %d times, want 1", m, firstPositions[m])
		}
	}
}

func TestBuildSchedule_BalancedFullLatinSquare(t *testing.T) {
	// With rounds == len(models) and runsPerModel=1, each model should
	// appear in every position (column) exactly once across all rounds.
	models := []string{"A", "B", "C", "D"}
	sched := BuildSchedule(models, 1, ScheduleConfig{Rounds: 4, Balanced: true})

	if len(sched) != 4 {
		t.Fatalf("expected 4 rounds, got %d", len(sched))
	}

	// colCounts[pos][model] = number of times model appears at position pos
	colCounts := make([]map[string]int, len(models))
	for i := range colCounts {
		colCounts[i] = map[string]int{}
	}
	for _, round := range sched {
		if len(round) != len(models) {
			t.Fatalf("round length = %d, want %d", len(round), len(models))
		}
		for pos, m := range round {
			colCounts[pos][m]++
		}
	}

	for pos, counts := range colCounts {
		for _, m := range models {
			if counts[m] != 1 {
				t.Errorf("column %d: model %q appears %d times, want 1", pos, m, counts[m])
			}
		}
	}
}

func TestBuildSchedule_BalancedRoundsGreaterThanModels(t *testing.T) {
	models := []string{"A", "B", "C"}
	sched := BuildSchedule(models, 1, ScheduleConfig{Rounds: 6, Balanced: true})

	if len(sched) != 6 {
		t.Fatalf("expected 6 rounds, got %d", len(sched))
	}

	// Each model should appear first exactly twice (6 rounds / 3 models)
	firstPositions := map[string]int{}
	for _, round := range sched {
		firstPositions[round[0]]++
	}
	for _, m := range models {
		if firstPositions[m] != 2 {
			t.Errorf("model %q was first %d times, want 2", m, firstPositions[m])
		}
	}

	// Consecutive rounds should not have the same first model
	for i := 1; i < len(sched); i++ {
		if sched[i][0] == sched[i-1][0] {
			t.Errorf("rounds %d and %d have the same first model %q", i-1, i, sched[i][0])
		}
	}
}

func TestBuildSchedule_SequentialGrouping(t *testing.T) {
	// In rounds mode, within a round, each model's runs should be grouped
	models := []string{"A", "B"}
	sched := BuildSchedule(models, 3, ScheduleConfig{Rounds: 2, Balanced: true})

	for ri, round := range sched {
		// Check that runs are grouped: all of model X, then all of model Y
		// Not interleaved
		seen := map[string]bool{}
		var prev string
		for _, m := range round {
			if m != prev && seen[m] {
				t.Errorf("round %d: model %q appeared non-contiguously", ri, m)
			}
			seen[m] = true
			prev = m
		}
	}
}

// ── Metrics extraction test ─────────────────────────────────────────────────

func TestMetricKeys(t *testing.T) {
	// Verify all expected metrics exist
	expected := []string{
		"ttft_ms", "eval_rate", "prompt_eval_rate", "total_duration_ms",
		"load_duration_ms", "prompt_eval_duration_ms", "eval_duration_ms",
		"prompt_eval_count", "eval_count",
	}
	for _, key := range expected {
		found := false
		for _, m := range AllMetrics {
			if m.Key == key {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("metric %q not found in AllMetrics", key)
		}
	}
}

// ── OllamaConnectionError tests ────────────────────────────────────────────

func newOpError() *net.OpError {
	return &net.OpError{Op: "dial", Net: "tcp", Err: &os.SyscallError{Syscall: "connect", Err: os.ErrNotExist}}
}

func TestOllamaConnectionError_NonOpError(t *testing.T) {
	orig := os.ErrClosed
	got := OllamaConnectionError("http://localhost:11434", orig)
	if got != orig {
		t.Errorf("expected original error returned unchanged, got %v", got)
	}
}

func TestOllamaConnectionError_NotInstalled(t *testing.T) {
	// Point PATH to an empty dir so LookPath("ollama") fails.
	tmp := t.TempDir()
	t.Setenv("PATH", tmp)

	got := OllamaConnectionError("http://localhost:11434", newOpError())
	msg := got.Error()

	if !strings.Contains(msg, "not installed") {
		t.Errorf("expected 'not installed' message, got: %s", msg)
	}
	if runtime.GOOS == "windows" {
		if !strings.Contains(msg, "install.ps1") {
			t.Errorf("expected PowerShell install command on Windows, got: %s", msg)
		}
	} else {
		if !strings.Contains(msg, "install.sh") {
			t.Errorf("expected curl install command, got: %s", msg)
		}
	}
}

func TestOllamaConnectionError_RemoteURL(t *testing.T) {
	// For a remote URL, even if ollama is not on PATH locally,
	// we should NOT say "not installed" — it's a remote server.
	tmp := t.TempDir()
	t.Setenv("PATH", tmp) // no ollama binary

	got := OllamaConnectionError("http://gpu-box:11434", newOpError())
	msg := got.Error()

	if strings.Contains(msg, "not installed") {
		t.Errorf("remote URL should not say 'not installed', got: %s", msg)
	}
	if !strings.Contains(msg, "could not connect") {
		t.Errorf("expected 'could not connect' message, got: %s", msg)
	}
	if !strings.Contains(msg, "gpu-box:11434") {
		t.Errorf("expected base URL in message, got: %s", msg)
	}
}

func TestOllamaConnectionError_NotRunning(t *testing.T) {
	// Create a fake "ollama" binary on PATH so LookPath succeeds.
	tmp := t.TempDir()
	fakeBin := filepath.Join(tmp, "ollama")
	if runtime.GOOS == "windows" {
		fakeBin += ".exe"
	}
	if err := os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", tmp)

	got := OllamaConnectionError("http://gpu-box:11434", newOpError())
	msg := got.Error()

	if !strings.Contains(msg, "could not connect") {
		t.Errorf("expected 'could not connect' message, got: %s", msg)
	}
	if !strings.Contains(msg, "gpu-box:11434") {
		t.Errorf("expected base URL in message, got: %s", msg)
	}
	if strings.Contains(msg, "not installed") {
		t.Errorf("should not say 'not installed' when binary exists, got: %s", msg)
	}
}

func TestIsLocalURL(t *testing.T) {
	tests := []struct {
		url  string
		want bool
	}{
		{"http://localhost:11434", true},
		{"http://127.0.0.1:11434", true},
		{"http://[::1]:11434", true},
		{"http://localhost", true},
		{"https://localhost:443", true},
		{"http://gpu-box:11434", false},
		{"http://192.168.1.100:11434", false},
		{"http://example.com:11434", false},
		{"http://10.0.0.1:11434", false},
	}
	for _, tt := range tests {
		t.Run(tt.url, func(t *testing.T) {
			got := isLocalURL(tt.url)
			if got != tt.want {
				t.Errorf("isLocalURL(%q) = %v, want %v", tt.url, got, tt.want)
			}
		})
	}
}

func TestRunResult_Get(t *testing.T) {
	r := RunResult{
		TTFT:            3215,
		EvalRate:        99.0,
		PromptEvalRate:  188.5,
		TotalDuration:   5922,
		LoadDuration:    137,
		PromptEvalTime:  86,
		EvalTime:        5577,
		PromptEvalCount: 28,
		EvalCount:       552,
		Round:           1,
	}

	if v := r.Get("ttft_ms"); v != 3215 {
		t.Errorf("Get(ttft_ms) = %v, want 3215", v)
	}
	if v := r.Get("eval_rate"); v != 99.0 {
		t.Errorf("Get(eval_rate) = %v, want 99.0", v)
	}
	if v := r.Get("eval_count"); v != 552 {
		t.Errorf("Get(eval_count) = %v, want 552", v)
	}
}
