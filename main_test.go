package main

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/piercecohen1/ollama-profiler/internal/cli"
)

func executeRoot(t *testing.T, args ...string) error {
	t.Helper()

	cmd := newRootCmd()
	cmd.SilenceUsage = true
	cmd.SilenceErrors = true
	cmd.SetArgs(args)
	cmd.SetOut(&bytes.Buffer{})
	cmd.SetErr(&bytes.Buffer{})
	return cmd.Execute()
}

func TestRootCmd_InvalidThinkRejected(t *testing.T) {
	err := executeRoot(t, "model-a", "--think=medum")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "--think must be one of") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRootCmd_RejectsNegativeNumPredict(t *testing.T) {
	err := executeRoot(t, "model-a", "--num-predict", "-5")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "--num-predict must be non-negative") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRootCmd_RejectsNegativeSeed(t *testing.T) {
	err := executeRoot(t, "model-a", "--seed", "-1")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "--seed must be non-negative") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRootCmd_DryRunWithModelsRejected(t *testing.T) {
	err := executeRoot(t, "model-a", "--dry-run")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "--dry-run is only valid in TUI mode") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRootCmd_DuplicateExportPathsRejected(t *testing.T) {
	err := executeRoot(t, "model-a", "--json", "out.json", "--html", "./out.json")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "resolve to the same output path") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRootCmd_PromptFileOverridesPrompt(t *testing.T) {
	tmpDir := t.TempDir()
	promptPath := filepath.Join(tmpDir, "prompt.txt")
	wantPrompt := "prompt from file\nwith a newline"
	if err := os.WriteFile(promptPath, []byte(wantPrompt), 0o644); err != nil {
		t.Fatalf("write prompt file: %v", err)
	}

	origRunCLI := runCLI
	origRunTUI := runTUI
	t.Cleanup(func() {
		runCLI = origRunCLI
		runTUI = origRunTUI
	})

	var gotCfg cli.Config
	runCLI = func(cfg cli.Config) error {
		gotCfg = cfg
		return nil
	}
	runTUI = func(bool, string) error {
		return errors.New("unexpected TUI invocation")
	}

	if err := executeRoot(t, "model-a", "--prompt", "flag prompt", "--prompt-file", promptPath, "--think=HIGH", "--url", "gpu-box:11434"); err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if gotCfg.Prompt != wantPrompt {
		t.Fatalf("Prompt = %q, want %q", gotCfg.Prompt, wantPrompt)
	}
	if gotCfg.Think != "high" {
		t.Fatalf("Think = %q, want %q", gotCfg.Think, "high")
	}
	if gotCfg.BaseURL != "http://gpu-box:11434" {
		t.Fatalf("BaseURL = %q, want %q", gotCfg.BaseURL, "http://gpu-box:11434")
	}
	if len(gotCfg.Models) != 1 || gotCfg.Models[0] != "model-a" {
		t.Fatalf("Models = %v, want [model-a]", gotCfg.Models)
	}
}

func TestRootCmd_BareThinkFlagUsesTrue(t *testing.T) {
	origRunCLI := runCLI
	t.Cleanup(func() {
		runCLI = origRunCLI
	})

	var gotCfg cli.Config
	runCLI = func(cfg cli.Config) error {
		gotCfg = cfg
		return nil
	}

	if err := executeRoot(t, "model-a", "--think"); err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if gotCfg.Think != "true" {
		t.Fatalf("Think = %q, want %q", gotCfg.Think, "true")
	}
}

func TestRootCmd_NoArgsErrors(t *testing.T) {
	origRunCLI := runCLI
	origRunTUI := runTUI
	t.Cleanup(func() {
		runCLI = origRunCLI
		runTUI = origRunTUI
	})

	runCLI = func(cli.Config) error {
		return errors.New("unexpected CLI invocation")
	}
	runTUI = func(bool, string) error {
		return errors.New("unexpected TUI invocation")
	}

	err := executeRoot(t)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "at least one model is required") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRootCmd_TUIFlagLaunchesTUI(t *testing.T) {
	origRunCLI := runCLI
	origRunTUI := runTUI
	t.Cleanup(func() {
		runCLI = origRunCLI
		runTUI = origRunTUI
	})

	runCLI = func(cli.Config) error {
		return errors.New("unexpected CLI invocation")
	}

	called := false
	runTUI = func(dry bool, base string) error {
		called = true
		if !dry {
			t.Fatalf("dry = false, want true")
		}
		if base != "http://localhost:11434" {
			t.Fatalf("base = %q, want %q", base, "http://localhost:11434")
		}
		return nil
	}

	if err := executeRoot(t, "--tui", "--dry-run"); err != nil {
		t.Fatalf("Execute() error: %v", err)
	}
	if !called {
		t.Fatal("expected TUI runner to be called")
	}
}

func TestRootCmd_TUIRejectsExportFlags(t *testing.T) {
	err := executeRoot(t, "--tui", "--json", "out.json")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "only valid in CLI mode") {
		t.Fatalf("unexpected error: %v", err)
	}
}
