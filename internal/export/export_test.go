package export

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/piercecohen1/ollama-profiler/internal/bench"
)

// testConfig returns a Config with sensible defaults for testing.
func testConfig() Config {
	return Config{
		Models:     []string{"model-a", "model-b"},
		Runs:       2,
		Rounds:     1,
		RoundRobin: false,
		Balanced:   false,
		Warmup:     false,
		Cooldown:   0,
		Prompt:     "test prompt",
		NumPredict: 256,
		Seed:       42,
		Think:      "",
	}
}

// testResults returns fake benchmark results for two models.
func testResults() map[string][]*bench.RunResult {
	return map[string][]*bench.RunResult{
		"model-a": {
			{TTFT: 100, EvalRate: 150, PromptEvalRate: 200, TotalDuration: 1000, LoadDuration: 50, PromptEvalTime: 10, EvalTime: 900, PromptEvalCount: 20, EvalCount: 180},
			{TTFT: 110, EvalRate: 145, PromptEvalRate: 210, TotalDuration: 1050, LoadDuration: 55, PromptEvalTime: 12, EvalTime: 920, PromptEvalCount: 20, EvalCount: 175},
		},
		"model-b": {
			{TTFT: 200, EvalRate: 100, PromptEvalRate: 150, TotalDuration: 2000, LoadDuration: 100, PromptEvalTime: 20, EvalTime: 1800, PromptEvalCount: 22, EvalCount: 256},
			{TTFT: 210, EvalRate: 105, PromptEvalRate: 155, TotalDuration: 2100, LoadDuration: 110, PromptEvalTime: 22, EvalTime: 1850, PromptEvalCount: 22, EvalCount: 256},
		},
	}
}

// ── JSON Export Tests ───────────────────────────────────────────────────────

func TestJSON_CreatesValidFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "results.json")

	err := JSON(testConfig(), testResults(), path)
	if err != nil {
		t.Fatalf("JSON() returned error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read exported file: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("exported JSON is not valid: %v", err)
	}

	// Verify meta section exists
	meta, ok := parsed["meta"].(map[string]interface{})
	if !ok {
		t.Fatal("JSON missing 'meta' section")
	}

	// Verify meta fields
	if models, ok := meta["models"].([]interface{}); !ok || len(models) != 2 {
		t.Errorf("meta.models: expected 2 models, got %v", meta["models"])
	}
	if runs, ok := meta["runs"].(float64); !ok || runs != 2 {
		t.Errorf("meta.runs: expected 2, got %v", meta["runs"])
	}
	if schedule, ok := meta["schedule"].(string); !ok || schedule != "sequential" {
		t.Errorf("meta.schedule: expected 'sequential', got %v", meta["schedule"])
	}
	if seed, ok := meta["seed"].(float64); !ok || seed != 42 {
		t.Errorf("meta.seed: expected 42, got %v", meta["seed"])
	}
	if prompt, ok := meta["prompt"].(string); !ok || prompt != "test prompt" {
		t.Errorf("meta.prompt: expected 'test prompt', got %v", meta["prompt"])
	}

	// Verify results section exists with both models
	results, ok := parsed["results"].(map[string]interface{})
	if !ok {
		t.Fatal("JSON missing 'results' section")
	}
	if _, ok := results["model-a"]; !ok {
		t.Error("results missing model-a")
	}
	if _, ok := results["model-b"]; !ok {
		t.Error("results missing model-b")
	}

	// Verify result entries have correct count
	modelA := results["model-a"].([]interface{})
	if len(modelA) != 2 {
		t.Errorf("model-a: expected 2 entries, got %d", len(modelA))
	}
}

func TestJSON_ScheduleStrings(t *testing.T) {
	tests := []struct {
		name     string
		cfg      Config
		wantSch  string
	}{
		{"sequential", Config{Models: []string{"m"}, Runs: 1, Rounds: 1}, "sequential"},
		{"round-robin", Config{Models: []string{"m"}, Runs: 1, Rounds: 1, RoundRobin: true}, "round-robin"},
		{"rounds", Config{Models: []string{"m"}, Runs: 1, Rounds: 3}, "rounds"},
		{"rounds-balanced", Config{Models: []string{"m"}, Runs: 1, Rounds: 3, Balanced: true}, "rounds-balanced"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "results.json")
			results := map[string][]*bench.RunResult{
				"m": {{TTFT: 100, EvalRate: 100}},
			}
			if err := JSON(tt.cfg, results, path); err != nil {
				t.Fatalf("JSON() error: %v", err)
			}
			data, _ := os.ReadFile(path)
			var parsed map[string]interface{}
			json.Unmarshal(data, &parsed)
			meta := parsed["meta"].(map[string]interface{})
			if got := meta["schedule"].(string); got != tt.wantSch {
				t.Errorf("schedule = %q, want %q", got, tt.wantSch)
			}
		})
	}
}

func TestJSON_ErrorRunsAreNull(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "results.json")
	results := map[string][]*bench.RunResult{
		"m": {
			{TTFT: 100, EvalRate: 100},
			{Error: os.ErrNotExist},
		},
	}
	cfg := Config{Models: []string{"m"}, Runs: 2, Rounds: 1}
	if err := JSON(cfg, results, path); err != nil {
		t.Fatalf("JSON() error: %v", err)
	}
	data, _ := os.ReadFile(path)
	var parsed map[string]interface{}
	json.Unmarshal(data, &parsed)
	entries := parsed["results"].(map[string]interface{})["m"].([]interface{})
	if entries[1] != nil {
		t.Errorf("error run should be null, got %v", entries[1])
	}
}

// ── HTML Export Tests ───────────────────────────────────────────────────────

func TestHTML_CreatesValidFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.html")

	err := HTML(testConfig(), testResults(), path)
	if err != nil {
		t.Fatalf("HTML() returned error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read exported file: %v", err)
	}

	content := string(data)

	// Verify it's valid HTML structure
	if !strings.Contains(content, "<!DOCTYPE html>") {
		t.Error("HTML missing DOCTYPE")
	}
	if !strings.Contains(content, "<html") {
		t.Error("HTML missing <html> tag")
	}
	if !strings.Contains(content, "</html>") {
		t.Error("HTML missing closing </html> tag")
	}
}

func TestHTML_ContainsModelNames(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.html")

	HTML(testConfig(), testResults(), path)
	data, _ := os.ReadFile(path)
	content := string(data)

	if !strings.Contains(content, "model-a") {
		t.Error("HTML missing model-a")
	}
	if !strings.Contains(content, "model-b") {
		t.Error("HTML missing model-b")
	}
}

func TestHTML_ContainsSummaryTable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.html")

	HTML(testConfig(), testResults(), path)
	data, _ := os.ReadFile(path)
	content := string(data)

	if !strings.Contains(content, "Summary") {
		t.Error("HTML missing Summary section")
	}
	if !strings.Contains(content, "<table>") {
		t.Error("HTML missing table element")
	}
	if !strings.Contains(content, "TTFT") {
		t.Error("HTML missing TTFT metric")
	}
	if !strings.Contains(content, "Generation TPS") {
		t.Error("HTML missing Generation TPS metric")
	}
}

func TestHTML_ContainsCharts(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.html")

	HTML(testConfig(), testResults(), path)
	data, _ := os.ReadFile(path)
	content := string(data)

	if !strings.Contains(content, "Performance Charts") {
		t.Error("HTML missing Performance Charts section")
	}
	if !strings.Contains(content, "bar-fill") {
		t.Error("HTML missing bar chart elements")
	}
}

func TestHTML_HighlightsBestModel(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.html")

	HTML(testConfig(), testResults(), path)
	data, _ := os.ReadFile(path)
	content := string(data)

	// model-a has better metrics (lower TTFT, higher TPS)
	// Should have "best" class and green color for winner
	if !strings.Contains(content, "class='best'") {
		t.Error("HTML missing best class highlighting")
	}
	if !strings.Contains(content, "color:#3fb950") {
		t.Error("HTML missing green winner color in charts")
	}
}

func TestHTML_ThemeColors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.html")

	HTML(testConfig(), testResults(), path)
	data, _ := os.ReadFile(path)
	content := string(data)

	// Verify dark theme CSS vars
	themeColors := []string{"#0d1117", "#161b22", "#21262d", "#30363d", "#e6edf3", "#c9a0ff", "#3fb950"}
	for _, c := range themeColors {
		if !strings.Contains(content, c) {
			t.Errorf("HTML missing theme color %s", c)
		}
	}
}

// ── PNG Export Tests ────────────────────────────────────────────────────────

func TestPNG_CreatesValidFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "charts.png")

	err := PNG(testConfig(), testResults(), path)
	if err != nil {
		t.Fatalf("PNG() returned error: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("failed to stat exported file: %v", err)
	}
	if info.Size() == 0 {
		t.Error("PNG file is empty")
	}
}

func TestPNG_HasPNGHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "charts.png")

	PNG(testConfig(), testResults(), path)

	data, _ := os.ReadFile(path)
	// PNG magic bytes: 137 80 78 71 13 10 26 10
	pngMagic := []byte{0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A}
	if len(data) < 8 {
		t.Fatal("PNG file too small")
	}
	for i, b := range pngMagic {
		if data[i] != b {
			t.Errorf("PNG magic byte %d: got 0x%02X, want 0x%02X", i, data[i], b)
		}
	}
}

func TestPNG_RetinaResolution(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "charts.png")

	PNG(testConfig(), testResults(), path)

	f, _ := os.Open(path)
	defer f.Close()

	// Decode to check dimensions (should be 2x: 1400x...)
	// Read IHDR chunk for width
	data, _ := os.ReadFile(path)
	// IHDR is at bytes 16-19 (width) in PNG
	if len(data) < 24 {
		t.Fatal("PNG too small for IHDR")
	}
	width := int(data[16])<<24 | int(data[17])<<16 | int(data[18])<<8 | int(data[19])
	if width != 1400 { // 700 * 2 (retina)
		t.Errorf("PNG width = %d, want 1400 (2x retina)", width)
	}
}

// ── Bundle Export Tests ─────────────────────────────────────────────────────

func TestBundle_CreatesAllFiles(t *testing.T) {
	dir := t.TempDir()
	bundleDir := filepath.Join(dir, "test-bundle")

	_, err := Bundle(testConfig(), testResults(), bundleDir)
	if err != nil {
		t.Fatalf("Bundle() returned error: %v", err)
	}

	expectedFiles := []string{"results.json", "report.html", "charts.png"}
	for _, f := range expectedFiles {
		path := filepath.Join(bundleDir, f)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("Bundle missing file: %s", f)
		}
	}
}

func TestBundle_CreatesDirectoryIfNotExists(t *testing.T) {
	dir := t.TempDir()
	bundleDir := filepath.Join(dir, "nested", "deep", "bundle")

	_, err := Bundle(testConfig(), testResults(), bundleDir)
	if err != nil {
		t.Fatalf("Bundle() returned error: %v", err)
	}

	info, err := os.Stat(bundleDir)
	if err != nil {
		t.Fatalf("bundle directory not created: %v", err)
	}
	if !info.IsDir() {
		t.Error("bundle path is not a directory")
	}
}

func TestBundle_AutoTimestampDir(t *testing.T) {
	// When dir is empty, Bundle creates a timestamped directory in cwd and returns its path
	dir, err := Bundle(testConfig(), testResults(), "")
	if err != nil {
		t.Fatalf("Bundle() returned error: %v", err)
	}
	if dir == "" {
		t.Fatal("Bundle() returned empty dir")
	}
	if !strings.Contains(dir, "ollama-profiler-") {
		t.Errorf("Bundle() dir %q does not contain expected prefix", dir)
	}

	// Clean up the auto-created directory
	os.RemoveAll(dir)
}

func TestBundle_JSONContentsMatchStandalone(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig()
	results := testResults()

	// Export via bundle
	bundleDir := filepath.Join(dir, "bundle")
	Bundle(cfg, results, bundleDir) //nolint:errcheck
	bundleJSON, _ := os.ReadFile(filepath.Join(bundleDir, "results.json"))

	// Export standalone
	standaloneJSON := filepath.Join(dir, "standalone.json")
	JSON(cfg, results, standaloneJSON)
	standaloneData, _ := os.ReadFile(standaloneJSON)

	// Both should parse to same structure (timestamps may differ)
	var bundleParsed, standaloneParsed map[string]interface{}
	json.Unmarshal(bundleJSON, &bundleParsed)
	json.Unmarshal(standaloneData, &standaloneParsed)

	// Compare results sections (skip meta which has timestamps)
	bundleResults := bundleParsed["results"].(map[string]interface{})
	standaloneResults := standaloneParsed["results"].(map[string]interface{})

	for _, model := range []string{"model-a", "model-b"} {
		bEntries := bundleResults[model].([]interface{})
		sEntries := standaloneResults[model].([]interface{})
		if len(bEntries) != len(sEntries) {
			t.Errorf("model %s: bundle has %d entries, standalone has %d", model, len(bEntries), len(sEntries))
		}
	}
}

// ── Edge Cases ──────────────────────────────────────────────────────────────

func TestExport_SingleModel(t *testing.T) {
	cfg := Config{
		Models:     []string{"solo-model"},
		Runs:       1,
		Rounds:     1,
		NumPredict: 256,
		Seed:       42,
	}
	results := map[string][]*bench.RunResult{
		"solo-model": {{TTFT: 100, EvalRate: 150, PromptEvalRate: 200, TotalDuration: 1000, LoadDuration: 50, PromptEvalTime: 10, EvalTime: 900, PromptEvalCount: 20, EvalCount: 180}},
	}

	dir := t.TempDir()

	// All three export formats should work with a single model
	if err := JSON(cfg, results, filepath.Join(dir, "r.json")); err != nil {
		t.Errorf("JSON single model: %v", err)
	}
	if err := HTML(cfg, results, filepath.Join(dir, "r.html")); err != nil {
		t.Errorf("HTML single model: %v", err)
	}
	if err := PNG(cfg, results, filepath.Join(dir, "r.png")); err != nil {
		t.Errorf("PNG single model: %v", err)
	}
}

func TestExport_ManyModels(t *testing.T) {
	models := []string{"m1", "m2", "m3", "m4", "m5", "m6", "m7"}
	cfg := Config{
		Models:     models,
		Runs:       1,
		Rounds:     1,
		NumPredict: 256,
		Seed:       42,
	}
	results := make(map[string][]*bench.RunResult)
	for i, m := range models {
		results[m] = []*bench.RunResult{{
			TTFT: float64(100 + i*10), EvalRate: float64(150 - i*5),
			PromptEvalRate: 200, TotalDuration: 1000,
			LoadDuration: 50, PromptEvalTime: 10, EvalTime: 900,
			PromptEvalCount: 20, EvalCount: 180,
		}}
	}

	dir := t.TempDir()

	// Should handle more models than palette colors (6 colors, 7 models)
	if err := HTML(cfg, results, filepath.Join(dir, "r.html")); err != nil {
		t.Errorf("HTML many models: %v", err)
	}
	if err := PNG(cfg, results, filepath.Join(dir, "r.png")); err != nil {
		t.Errorf("PNG many models: %v", err)
	}
}

func TestExport_AllErrorRuns(t *testing.T) {
	cfg := Config{
		Models:     []string{"bad-model"},
		Runs:       2,
		Rounds:     1,
		NumPredict: 256,
		Seed:       42,
	}
	results := map[string][]*bench.RunResult{
		"bad-model": {
			{Error: os.ErrNotExist},
			{Error: os.ErrPermission},
		},
	}

	dir := t.TempDir()

	// Should not crash with all-error results
	if err := JSON(cfg, results, filepath.Join(dir, "r.json")); err != nil {
		t.Errorf("JSON all errors: %v", err)
	}
	if err := HTML(cfg, results, filepath.Join(dir, "r.html")); err != nil {
		t.Errorf("HTML all errors: %v", err)
	}
	if err := PNG(cfg, results, filepath.Join(dir, "r.png")); err != nil {
		t.Errorf("PNG all errors: %v", err)
	}
}

func TestExport_InvalidPath(t *testing.T) {
	cfg := testConfig()
	results := testResults()

	// Non-existent directory should fail
	badPath := "/nonexistent/dir/file.json"
	if err := JSON(cfg, results, badPath); err == nil {
		t.Error("JSON should fail with invalid path")
	}
	if err := HTML(cfg, results, badPath); err == nil {
		t.Error("HTML should fail with invalid path")
	}
	if err := PNG(cfg, results, badPath+".png"); err == nil {
		t.Error("PNG should fail with invalid path")
	}
}

// ── Helper Tests ────────────────────────────────────────────────────────────

func TestValidResults(t *testing.T) {
	runs := []*bench.RunResult{
		{TTFT: 100},
		nil,
		{Error: os.ErrNotExist},
		{TTFT: 200},
	}
	valid := validResults(runs)
	if len(valid) != 2 {
		t.Errorf("validResults: expected 2, got %d", len(valid))
	}
}

func TestFindBest(t *testing.T) {
	lower := true
	higher := false

	tests := []struct {
		name    string
		avgs    map[string]float64
		lower   *bool
		want    string
	}{
		{"lower is better", map[string]float64{"a": 100, "b": 200}, &lower, "a"},
		{"higher is better", map[string]float64{"a": 100, "b": 200}, &higher, "b"},
		{"nil direction", map[string]float64{"a": 100, "b": 200}, nil, ""},
		{"empty map", map[string]float64{}, &lower, ""},
		{"single entry", map[string]float64{"only": 42}, &lower, "only"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := findBest(tt.avgs, tt.lower)
			if got != tt.want {
				t.Errorf("findBest() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestScheduleStr(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want string
	}{
		{"default", Config{}, "sequential"},
		{"round-robin", Config{RoundRobin: true}, "round-robin"},
		{"rounds", Config{Rounds: 3}, "rounds"},
		{"balanced", Config{Rounds: 3, Balanced: true}, "rounds-balanced"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := scheduleStr(tt.cfg); got != tt.want {
				t.Errorf("scheduleStr() = %q, want %q", got, tt.want)
			}
		})
	}
}
