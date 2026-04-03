#!/usr/bin/env python3
"""ollama-bench TUI — Interactive Ollama model benchmarking with a Claude Code-inspired interface."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import (
    Button, DataTable, Footer, Header, Input, Label,
    ProgressBar, RichLog, Rule, Select, SelectionList,
    Static, Switch, TabbedContent, TabPane, TextArea,
)
from textual.containers import VerticalScroll
from textual.worker import get_current_worker

# Import benchmark logic from the CLI module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_bench import benchmark_once, build_schedule, stats, fmt, METRICS, PER_RUN_METRICS


# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_ollama_models(base_url: str = "http://localhost:11434") -> list[dict]:
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read()).get("models", [])
    except Exception:
        return []


def model_label(m: dict) -> str:
    name = m["name"]
    d = m.get("details", {})
    parts = [name]
    if ps := d.get("parameter_size"):
        parts.append(f"— {ps}")
    if q := d.get("quantization_level"):
        parts.append(q)
    size = m.get("size", 0)
    if size:
        parts.append(f"({size / 1e9:.1f} GB)")
    return "  ".join(parts)


# ── Config Screen ────────────────────────────────────────────────────────────

class ConfigScreen(Screen):
    BINDINGS = [Binding("escape", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="config-body"):
            yield Static("Models", classes="section-title")
            yield SelectionList[str](id="model-list")
            yield Input(placeholder="Or type model names, comma-separated", id="in-manual")

            yield Rule()
            yield Static("Settings", classes="section-title")

            with Horizontal(classes="form-row"):
                with Vertical(classes="form-col"):
                    yield Label("Runs / round")
                    yield Input(value="3", type="integer", id="in-runs", max_length=3)
                with Vertical(classes="form-col"):
                    yield Label("Schedule")
                    yield Select[str](
                        [
                            ("Sequential", "sequential"),
                            ("Round Robin", "round-robin"),
                            ("Rounds — Random", "rounds-random"),
                            ("Rounds — Balanced", "rounds-balanced"),
                        ],
                        value="sequential",
                        allow_blank=False,
                        id="sel-mode",
                    )

            with Horizontal(classes="form-row", id="rounds-row"):
                with Vertical(classes="form-col"):
                    yield Label("Rounds")
                    yield Input(value="3", type="integer", id="in-rounds", max_length=3)
                with Vertical(classes="form-col"):
                    yield Label("Cooldown (sec)")
                    yield Input(value="0", type="integer", id="in-cooldown", max_length=4)

            with Horizontal(classes="form-row"):
                yield Switch(id="sw-warmup", value=False)
                yield Static("  Warmup (one uncounted run per model)", id="warmup-label")

            yield Rule()
            yield Static("Prompt", classes="section-title")
            yield TextArea(
                "Write a 200 word explanation of transformers in ML.",
                id="ta-prompt",
            )

            with Horizontal(classes="btn-row"):
                yield Button("Start Benchmark", variant="primary", id="btn-start")
                yield Button("Quit", variant="error", id="btn-quit")

        yield Footer()

    def on_mount(self) -> None:
        models = fetch_ollama_models()
        sel = self.query_one("#model-list", SelectionList)
        if models:
            for m in models:
                sel.add_option((model_label(m), m["name"]))
        else:
            self.notify(
                "Could not connect to Ollama. Type model names below.",
                severity="warning",
            )
        self._toggle_rounds()

    @on(Select.Changed, "#sel-mode")
    def _on_mode_change(self, event: Select.Changed) -> None:
        self._toggle_rounds()

    def _toggle_rounds(self) -> None:
        mode = self.query_one("#sel-mode", Select).value
        self.query_one("#rounds-row").display = mode in (
            "rounds-random", "rounds-balanced",
        )

    @on(Button.Pressed, "#btn-start")
    def _on_start(self) -> None:
        # Gather selected models
        selected = list(self.query_one("#model-list", SelectionList).selected)
        manual = self.query_one("#in-manual", Input).value.strip()
        if manual:
            for name in (n.strip() for n in manual.split(",")):
                if name and name not in selected:
                    selected.append(name)
        if not selected:
            self.notify("Select or type at least one model", severity="warning")
            return

        prompt = self.query_one("#ta-prompt", TextArea).text.strip()
        if not prompt:
            self.notify("Enter a prompt", severity="warning")
            return

        mode = self.query_one("#sel-mode", Select).value
        runs = max(1, int(self.query_one("#in-runs", Input).value or "3"))
        n_rounds = max(1, int(self.query_one("#in-rounds", Input).value or "1"))
        cooldown = max(0, int(self.query_one("#in-cooldown", Input).value or "0"))

        if mode in ("rounds-random", "rounds-balanced") and n_rounds <= 1:
            n_rounds = 3

        self.app.push_screen(BenchmarkScreen({
            "models": selected,
            "runs": runs,
            "round_robin": mode == "round-robin",
            "rounds": n_rounds if mode in ("rounds-random", "rounds-balanced") else 1,
            "balanced": mode == "rounds-balanced",
            "cooldown": cooldown,
            "warmup": self.query_one("#sw-warmup", Switch).value,
            "prompt": prompt,
            "base_url": "http://localhost:11434",
        }))

    @on(Button.Pressed, "#btn-quit")
    def _on_quit(self) -> None:
        self.app.exit()


# ── Benchmark Screen ─────────────────────────────────────────────────────────

class BenchmarkScreen(Screen):
    BINDINGS = [Binding("c", "cancel", "Cancel")]

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.results: defaultdict[str, list] = defaultdict(list)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="bench-body"):
            yield Static("Initializing...", id="lbl-status")
            yield ProgressBar(id="pb-main", show_eta=True)
            yield Static("Live Results", classes="section-title")
            yield DataTable(id="dt-live")
            yield Static("Log", classes="section-title")
            yield RichLog(id="log", auto_scroll=True, max_lines=200)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#dt-live", DataTable)
        table.add_columns("Model", "Rnd", "Run", "TTFT", "Gen TPS", "Prompt TPS", "Total")
        table.cursor_type = "none"

        self.schedule = build_schedule(
            self.config["models"],
            self.config["runs"],
            self.config["round_robin"],
            self.config["rounds"],
            self.config["balanced"],
        )
        total = sum(len(r) for r in self.schedule)
        if self.config["warmup"]:
            total += len(self.config["models"])
        self.query_one("#pb-main", ProgressBar).update(total=total)

        self._run()

    @work(thread=True, exclusive=True)
    def _run(self) -> None:
        worker = get_current_worker()
        cfg = self.config
        models, prompt, url = cfg["models"], cfg["prompt"], cfg["base_url"]
        n_rounds = len(self.schedule)

        # warmup
        if cfg["warmup"]:
            for m in models:
                if worker.is_cancelled:
                    return
                self.app.call_from_thread(self._status, f"Warmup: {m}")
                try:
                    benchmark_once(m, prompt, url)
                    self.app.call_from_thread(self._log, f"  Warmup {m} done")
                except Exception as e:
                    self.app.call_from_thread(self._log, f"  ⚠ Warmup {m}: {e}", "yellow")
                self.app.call_from_thread(self._advance)

        # benchmark rounds
        for ri, sched in enumerate(self.schedule):
            if worker.is_cancelled:
                break

            # cooldown between rounds
            if ri > 0 and cfg["cooldown"] > 0:
                for s in range(cfg["cooldown"], 0, -1):
                    if worker.is_cancelled:
                        break
                    self.app.call_from_thread(self._status, f"Cooldown {s}s...")
                    time.sleep(1)
                self.app.call_from_thread(self._log, f"  Cooldown complete")

            cnt: defaultdict[str, int] = defaultdict(int)
            per: defaultdict[str, int] = defaultdict(int)
            for m in sched:
                per[m] += 1

            for m in sched:
                if worker.is_cancelled:
                    break
                cnt[m] += 1
                rn, rnd = cnt[m], ri + 1
                lbl = (
                    f"R{rnd}/{n_rounds}  {m}  run {rn}/{per[m]}"
                    if n_rounds > 1
                    else f"{m}  run {rn}/{per[m]}"
                )
                self.app.call_from_thread(self._status, lbl)

                try:
                    res = benchmark_once(m, prompt, url)
                    res["_round"] = rnd
                    self.app.call_from_thread(self._record, m, rnd, rn, res)
                except Exception as e:
                    self.app.call_from_thread(self._log, f"  ✗ {m} R{rnd}/run{rn}: {e}", "red")
                    self.results[m].append(None)
                self.app.call_from_thread(self._advance)

        self.app.call_from_thread(self._done)

    # ── UI helpers (called on main thread) ───────────────────────────────

    def _status(self, text: str) -> None:
        self.query_one("#lbl-status", Static).update(text)

    def _advance(self) -> None:
        self.query_one("#pb-main", ProgressBar).advance(1)

    def _log(self, msg: str, style: str = "white") -> None:
        self.query_one("#log", RichLog).write(f"[{style}]{msg}[/]")

    def _record(self, model: str, rnd: int, run: int, res: dict) -> None:
        self.results[model].append(res)
        self.query_one("#dt-live", DataTable).add_row(
            model, str(rnd), str(run),
            fmt(res["ttft_ms"], "ms"),
            fmt(res["eval_rate"], "tok/s"),
            fmt(res["prompt_eval_rate"], "tok/s"),
            fmt(res["total_duration_ms"], "ms"),
        )
        self._log(f"  ✓ {model} R{rnd}/run{run} — {res['eval_rate']:.1f} tok/s", "green")

    def _done(self) -> None:
        self._status("Complete!")
        self.app.push_screen(ResultsScreen(dict(self.results), self.config))

    def action_cancel(self) -> None:
        for w in self.workers:
            w.cancel()
        self._log("  Cancelled — showing partial results", "yellow")
        self._status("Cancelled")
        self.set_timer(0.5, self._done)


# ── Results Screen ───────────────────────────────────────────────────────────

class ResultsScreen(Screen):
    BINDINGS = [
        Binding("r", "restart", "Restart"),
        Binding("e", "export", "Export JSON"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, results: dict, config: dict) -> None:
        super().__init__()
        self.results = results
        self.config = config
        self.models = config["models"]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(id="result-tabs"):
            with TabPane("Summary", id="tab-summary"):
                yield DataTable(id="dt-summary")
            with TabPane("Per-Run", id="tab-perrun"):
                yield DataTable(id="dt-perrun")
            with TabPane("Relative", id="tab-relative"):
                yield DataTable(id="dt-relative")
            with TabPane("Charts", id="tab-charts"):
                with VerticalScroll(id="chart-scroll"):
                    yield Static(id="chart-content")
        yield Footer()

    def on_mount(self) -> None:
        self._build_summary()
        self._build_perrun()
        self._build_relative()
        self._build_charts()

    def _build_summary(self) -> None:
        t = self.query_one("#dt-summary", DataTable)
        t.cursor_type = "row"
        t.add_column("Metric", key="metric")
        for m in self.models:
            t.add_column(m, key=m)

        for key, label, unit, lower_better in METRICS:
            avgs: dict[str, float] = {}
            cells: dict[str, str] = {}
            for m in self.models:
                valid = [r for r in self.results.get(m, []) if r is not None]
                if not valid:
                    cells[m] = "n/a"
                    continue
                avg, sd, *_ = stats([r[key] for r in valid])
                if math.isnan(avg):
                    cells[m] = "n/a"
                    continue
                avgs[m] = avg
                cells[m] = (
                    f"{fmt(avg, unit)} ±{fmt(sd, unit)}"
                    if len(valid) > 1 and sd > 0
                    else fmt(avg, unit)
                )

            best = None
            if avgs and lower_better is not None:
                best = (min if lower_better else max)(avgs, key=avgs.get)

            row = [label]
            for m in self.models:
                c = cells.get(m, "n/a")
                row.append(f"★ {c}" if m == best else c)
            t.add_row(*row)

    def _build_perrun(self) -> None:
        t = self.query_one("#dt-perrun", DataTable)
        t.cursor_type = "row"
        t.add_columns("Model", "Rnd", "Run")
        for _, label, _ in PER_RUN_METRICS:
            t.add_column(label)

        for m in self.models:
            cur_rnd, rn = None, 0
            for r in self.results.get(m, []):
                if r is None:
                    t.add_row(m, "?", "?", *["err"] * len(PER_RUN_METRICS))
                    continue
                rnd = r.get("_round", 1)
                if rnd != cur_rnd:
                    cur_rnd, rn = rnd, 0
                rn += 1
                t.add_row(m, str(rnd), str(rn), *[fmt(r[k], u) for k, _, u in PER_RUN_METRICS])

    def _build_relative(self) -> None:
        t = self.query_one("#dt-relative", DataTable)
        t.cursor_type = "row"
        if len(self.models) < 2:
            t.add_column("Info")
            t.add_row("Need 2+ models for comparison")
            return

        t.add_column("Metric", key="metric")
        for m in self.models:
            t.add_column(m, key=m)

        for key, label, lower_better in [
            ("ttft_ms",           "TTFT",            True),
            ("eval_rate",         "Generation TPS",  False),
            ("prompt_eval_rate",  "Prompt Eval TPS", False),
            ("total_duration_ms", "Total Duration",  True),
        ]:
            avgs: dict[str, float] = {}
            for m in self.models:
                valid = [r for r in self.results.get(m, []) if r is not None]
                avg, *_ = stats([r[key] for r in valid]) if valid else (float("nan"),)
                if not math.isnan(avg):
                    avgs[m] = avg
            if not avgs:
                continue

            best = (min if lower_better else max)(avgs.values())
            row = [label]
            for m in self.models:
                if m not in avgs or best == 0:
                    row.append("n/a")
                elif lower_better:
                    pct = ((avgs[m] - best) / best) * 100
                    row.append("★ best" if abs(pct) < 0.5 else f"+{pct:.1f}%")
                else:
                    pct = ((best - avgs[m]) / best) * 100
                    row.append("★ best" if abs(pct) < 0.5 else f"+{pct:.1f}%")
            t.add_row(*row)

    def _build_charts(self) -> None:
        CHART_METRICS = [
            ("ttft_ms",           "TTFT",            "ms",    True),
            ("eval_rate",         "Generation TPS",  "tok/s", False),
            ("prompt_eval_rate",  "Prompt Eval TPS", "tok/s", False),
            ("total_duration_ms", "Total Duration",  "ms",    True),
            ("eval_duration_ms",  "Eval Time",       "ms",    True),
            ("eval_count",        "Generated Tokens", "tok",  None),
        ]

        # Assign each model a color from a palette
        palette = ["#c9a0ff", "#f78166", "#79c0ff", "#d29922", "#f0883e", "#a5d6ff"]
        model_colors = {m: palette[i % len(palette)] for i, m in enumerate(self.models)}

        BAR_WIDTH = 40
        BLOCKS = " ▏▎▍▌▋▊▉█"
        lines: list[str] = []

        for key, label, unit, lower_better in CHART_METRICS:
            # Compute per-model averages
            avgs: dict[str, float] = {}
            for m in self.models:
                valid = [r for r in self.results.get(m, []) if r is not None]
                if valid:
                    avg, *_ = stats([r[key] for r in valid])
                    if not math.isnan(avg):
                        avgs[m] = avg
            if not avgs:
                continue

            max_val = max(avgs.values())
            if max_val == 0:
                continue

            # Find best
            best = None
            if lower_better is not None:
                best = (min if lower_better else max)(avgs, key=avgs.get)

            direction = "lower is better" if lower_better else "higher is better" if lower_better is False else ""
            lines.append(f"  [bold]{label}[/bold]  [dim]{direction}[/dim]")
            lines.append("")

            max_name_len = max(len(m) for m in avgs)

            for m in self.models:
                if m not in avgs:
                    continue
                val = avgs[m]
                fraction = val / max_val

                # Build smooth bar
                full = int(fraction * BAR_WIDTH)
                remainder = (fraction * BAR_WIDTH) - full
                partial_idx = int(remainder * 8)
                bar = "█" * full
                if partial_idx > 0 and full < BAR_WIDTH:
                    bar += BLOCKS[partial_idx]
                bar = bar.ljust(BAR_WIDTH)

                color = "#3fb950" if m == best else model_colors[m]
                name = m.ljust(max_name_len)
                val_str = fmt(val, unit)
                star = "  ★" if m == best else ""

                lines.append(f"  {name}  [{color}]{bar}[/{color}]  {val_str}{star}")

            lines.append("")
            lines.append("")

        self.query_one("#chart-content", Static).update("\n".join(lines))

    def action_restart(self) -> None:
        self.app.pop_screen()   # ResultsScreen
        self.app.pop_screen()   # BenchmarkScreen
        # Back to ConfigScreen

    def action_export(self) -> None:
        sanitize = lambda v: None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
        export = {
            m: [
                {k: sanitize(v) for k, v in r.items() if not k.startswith("_")} if r else None
                for r in self.results.get(m, [])
            ]
            for m in self.models
        }
        path = Path.cwd() / "ollama-bench-results.json"
        path.write_text(json.dumps(export, indent=2))
        self.notify(f"Exported to {path}")

    def action_quit(self) -> None:
        self.app.exit()


# ── App ──────────────────────────────────────────────────────────────────────

class BenchApp(App):
    TITLE = "Ollama Bench"
    SUB_TITLE = "Model Performance Comparison"

    CSS = """
    Screen {
        overflow-y: auto;
    }

    #config-body {
        padding: 1 2;
    }

    .section-title {
        text-style: bold;
        color: $primary;
        margin: 1 0 0 0;
    }

    .form-row {
        height: auto;
        margin: 0 0 1 0;
    }

    .form-col {
        width: 1fr;
        height: auto;
        margin: 0 1 0 0;
    }

    .btn-row {
        height: auto;
        margin: 2 0 1 0;
        align-horizontal: center;
    }

    SelectionList {
        height: auto;
        max-height: 8;
        margin: 0 0 0 0;
    }

    #in-manual {
        margin: 0 0 1 0;
    }

    #ta-prompt {
        height: 4;
        margin: 0 0 1 0;
    }

    #warmup-label {
        margin: 1 0 0 0;
    }

    Button {
        margin: 0 1;
    }

    /* ── Benchmark Screen ── */

    #bench-body {
        padding: 1 2;
    }

    #lbl-status {
        text-style: bold;
        color: $primary;
        margin: 0 0 1 0;
    }

    #pb-main {
        margin: 0 0 1 0;
    }

    #dt-live {
        height: 1fr;
        min-height: 6;
    }

    #log {
        height: 8;
        min-height: 4;
        margin: 1 0 0 0;
    }

    /* ── Results Screen ── */

    TabbedContent {
        height: 1fr;
    }

    TabPane {
        padding: 1 2;
    }

    DataTable {
        height: 1fr;
    }

    #chart-scroll {
        height: 1fr;
    }

    #chart-content {
        padding: 1 0;
    }
    """

    def on_mount(self) -> None:
        self.register_theme(
            Theme(
                name="bench",
                primary="#c9a0ff",
                secondary="#7c65a9",
                accent="#f78166",
                background="#0d1117",
                surface="#161b22",
                panel="#21262d",
                foreground="#e6edf3",
                error="#f85149",
                success="#3fb950",
                warning="#d29922",
                dark=True,
            )
        )
        self.theme = "bench"
        self.push_screen(ConfigScreen())


def main():
    BenchApp().run()


if __name__ == "__main__":
    main()
