#!/usr/bin/env python3
"""
ollama-bench — Compare Ollama model performance side-by-side.

Usage:
    ollama-bench gemma4:e4b gemma4:26b gemma4:31b -n 4
    ollama-bench gemma4:e4b gemma4:26b -n 3 --rounds 4 --cooldown 30
    ollama-bench gemma4:e4b gemma4:26b -n 3 --rounds 3 --balanced --warmup
    ollama-bench gemma4:e4b gemma4:26b -n 4 --round-robin
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
import urllib.request
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )
    from rich import box
    from rich.rule import Rule
except ImportError:
    print("This tool requires the 'rich' library. Install with: pip install rich")
    sys.exit(1)

console = Console()

# (key, display_label, unit, lower_is_better)  — None = neutral
METRICS = [
    ("ttft_ms",                "TTFT",              "ms",    True),
    ("eval_rate",              "Generation TPS",    "tok/s", False),
    ("prompt_eval_rate",       "Prompt Eval TPS",   "tok/s", False),
    ("total_duration_ms",      "Total Duration",    "ms",    True),
    ("load_duration_ms",       "Load Duration",     "ms",    True),
    ("prompt_eval_duration_ms","Prompt Eval Time",  "ms",    True),
    ("eval_duration_ms",       "Eval Time",         "ms",    True),
    ("prompt_eval_count",      "Prompt Tokens",     "tok",   None),
    ("eval_count",             "Generated Tokens",  "tok",   None),
]

PER_RUN_METRICS = [
    ("ttft_ms",           "TTFT",       "ms"),
    ("eval_rate",         "Gen TPS",    "tok/s"),
    ("prompt_eval_rate",  "Prompt TPS", "tok/s"),
    ("total_duration_ms", "Total",      "ms"),
    ("load_duration_ms",  "Load",       "ms"),
    ("eval_duration_ms",  "Eval Time",  "ms"),
    ("eval_count",        "Tokens",     "tok"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt(val, unit):
    """Format a single metric value."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "n/a"
    if unit == "tok":
        return f"{int(val)}"
    if unit == "tok/s":
        return f"{val:.1f}"
    if unit == "ms":
        return f"{val / 1000:.2f}s" if val >= 10_000 else f"{val:.1f}ms"
    return f"{val:.2f}"


def stats(values):
    """Return (mean, stdev, min, max) filtering out NaN."""
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan"), float("nan"), float("nan")
    avg = statistics.mean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return avg, sd, min(clean), max(clean)


# ── benchmark ────────────────────────────────────────────────────────────────

def benchmark_once(model, prompt, base_url):
    """Run one inference pass and return a metrics dict."""
    url = f"{base_url}/api/generate"
    payload = json.dumps({"model": model, "prompt": prompt}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    start = time.perf_counter()
    first_token_time = None
    final = None

    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            tok = obj.get("response", "")
            if first_token_time is None and (tok or obj.get("thinking")):
                first_token_time = time.perf_counter()
            if obj.get("done"):
                final = obj

    if final is None:
        raise RuntimeError(f"No final chunk received from {model}")

    td  = final.get("total_duration", 0)
    ld  = final.get("load_duration", 0)
    pec = final.get("prompt_eval_count", 0)
    ped = final.get("prompt_eval_duration", 0)
    ec  = final.get("eval_count", 0)
    ed  = final.get("eval_duration", 0)

    return {
        "ttft_ms":                (first_token_time - start) * 1000 if first_token_time else float("nan"),
        "total_duration_ms":      td / 1e6,
        "load_duration_ms":       ld / 1e6,
        "prompt_eval_count":      pec,
        "prompt_eval_duration_ms":ped / 1e6,
        "prompt_eval_rate":       (pec * 1e9 / ped) if ped else 0.0,
        "eval_count":             ec,
        "eval_duration_ms":       ed / 1e6,
        "eval_rate":              (ec * 1e9 / ed) if ed else 0.0,
    }


# ── scheduling ───────────────────────────────────────────────────────────────

def build_schedule(models, n_runs, round_robin, n_rounds, balanced):
    """Build execution schedule as a list of rounds.

    Each round is a list of model names in execution order.
    Within a round, each model's runs are grouped together (sequential)
    so the model stays loaded in VRAM.

    Returns: list[list[str]]
    """
    if round_robin:
        # Single round, interleaved: A,B,C, A,B,C, A,B,C ...
        sched = []
        for _ in range(n_runs):
            sched.extend(models)
        return [sched]

    if n_rounds > 1:
        rounds = []
        if balanced:
            # Latin square rotation for positional fairness.
            # Shuffle the base order so it's not just CLI arg order.
            base = models[:]
            random.shuffle(base)
            for r in range(n_rounds):
                offset = r % len(base)
                order = base[offset:] + base[:offset]
                sched = []
                for model in order:
                    sched.extend([model] * n_runs)
                rounds.append(sched)
        else:
            # Fully random order per round
            for _ in range(n_rounds):
                order = models[:]
                random.shuffle(order)
                sched = []
                for model in order:
                    sched.extend([model] * n_runs)
                rounds.append(sched)
        return rounds

    # Default: sequential, single round — all runs of A, then B, then C
    sched = []
    for model in models:
        sched.extend([model] * n_runs)
    return [sched]


def run_benchmarks(models, schedule, prompt, base_url, warmup, cooldown):
    """Execute the schedule and return {model: [result_dict, ...]}."""
    results = defaultdict(list)
    n_rounds = len(schedule)
    total_runs = sum(len(r) for r in schedule)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # warmup: one uncounted run per model
        if warmup:
            wt = progress.add_task("Warming up…", total=len(models))
            for model in models:
                progress.update(wt, description=f"Warmup: {model}")
                try:
                    benchmark_once(model, prompt, base_url)
                except Exception as e:
                    console.print(f"[yellow]⚠ warmup failed for {model}: {e}[/yellow]")
                progress.advance(wt)
            progress.remove_task(wt)

        bt = progress.add_task("Benchmarking…", total=total_runs)

        for round_i, round_sched in enumerate(schedule):
            # cooldown between rounds
            if round_i > 0 and cooldown > 0:
                for remaining in range(cooldown, 0, -1):
                    progress.update(bt, description=f"Cooldown {remaining}s…")
                    time.sleep(1)

            # run the round
            model_run_count = defaultdict(int)
            # count expected runs per model this round
            runs_expected = defaultdict(int)
            for m in round_sched:
                runs_expected[m] += 1

            for model in round_sched:
                model_run_count[model] += 1
                run_num = model_run_count[model]
                n_for_model = runs_expected[model]

                if n_rounds > 1:
                    desc = f"R{round_i + 1}/{n_rounds}  {model}  run {run_num}/{n_for_model}"
                else:
                    desc = f"{model}  run {run_num}/{n_for_model}"
                progress.update(bt, description=desc)

                try:
                    result = benchmark_once(model, prompt, base_url)
                    result["_round"] = round_i + 1
                    results[model].append(result)
                except Exception as e:
                    console.print(f"[red]✗ {model}: {e}[/red]")
                    results[model].append(None)
                progress.advance(bt)

    return results


# ── display ──────────────────────────────────────────────────────────────────

def show_per_run(results, models, n_rounds):
    """Per-model detail tables showing every run."""
    for model in models:
        runs = results[model]
        if not runs or all(r is None for r in runs):
            console.print(f"[red]No successful runs for {model}[/red]\n")
            continue

        table = Table(
            title=model, box=box.ROUNDED,
            title_style="bold cyan", header_style="bold",
        )
        if n_rounds > 1:
            table.add_column("Rnd", style="dim", justify="right", width=4)
        table.add_column("Run", style="dim", justify="right", width=4)
        for _, label, _ in PER_RUN_METRICS:
            table.add_column(label, justify="right")

        current_round = None
        run_in_round = 0

        for run in runs:
            if run is None:
                cells = ["[red]err[/red]"] * len(PER_RUN_METRICS)
                if n_rounds > 1:
                    table.add_row("?", "?", *cells)
                else:
                    table.add_row("?", *cells)
                continue

            r = run.get("_round", 1)
            if n_rounds > 1 and current_round is not None and r != current_round:
                table.add_section()
                run_in_round = 0
            current_round = r
            run_in_round += 1

            cells = [fmt(run[k], u) for k, _, u in PER_RUN_METRICS]
            if n_rounds > 1:
                table.add_row(str(r), str(run_in_round), *cells)
            else:
                table.add_row(str(run_in_round), *cells)

        # averages row
        valid = [r for r in runs if r is not None]
        if len(valid) > 1:
            table.add_section()
            avg_row = []
            if n_rounds > 1:
                avg_row.append("")
            avg_row.append("avg")
            for key, _, unit in PER_RUN_METRICS:
                vals = [r[key] for r in valid]
                avg, *_ = stats(vals)
                avg_row.append(f"[bold]{fmt(avg, unit)}[/bold]")
            table.add_row(*avg_row, style="on grey15")

        console.print(table)
        console.print()


def show_summary(results, models):
    """Side-by-side comparison table with best values highlighted."""
    table = Table(
        title="Comparison Summary",
        box=box.HEAVY_EDGE,
        title_style="bold white on dark_blue",
        header_style="bold",
        padding=(0, 1),
    )
    table.add_column("Metric", style="bold", width=18)
    for model in models:
        table.add_column(model, justify="right", min_width=16)

    for key, label, unit, lower_is_better in METRICS:
        model_avgs = {}
        cells = {}

        for model in models:
            valid = [r for r in results[model] if r is not None]
            if not valid:
                cells[model] = "[dim]n/a[/dim]"
                continue
            vals = [r[key] for r in valid]
            avg, sd, lo, hi = stats(vals)
            if math.isnan(avg):
                cells[model] = "[dim]n/a[/dim]"
                continue
            model_avgs[model] = avg
            if len(valid) > 1 and sd > 0:
                cells[model] = f"{fmt(avg, unit)}  [dim]±{fmt(sd, unit)}[/dim]"
            else:
                cells[model] = fmt(avg, unit)

        # highlight best
        best = None
        if model_avgs and lower_is_better is not None:
            best = (min if lower_is_better else max)(model_avgs, key=model_avgs.get)

        row = [label]
        for model in models:
            cell = cells.get(model, "[dim]n/a[/dim]")
            if model == best:
                cell = f"[bold green]{cell}[/bold green]"
            row.append(cell)
        table.add_row(*row)

    console.print(table)


def show_relative(results, models):
    """Show how each model compares to the best as a percentage."""
    if len(models) < 2:
        return

    key_metrics = [
        ("ttft_ms",           "TTFT",              True),
        ("eval_rate",         "Generation TPS",    False),
        ("prompt_eval_rate",  "Prompt Eval TPS",   False),
        ("total_duration_ms", "Total Duration",     True),
    ]

    table = Table(
        title="Relative Performance (vs best)",
        box=box.SIMPLE_HEAVY,
        title_style="bold",
        header_style="bold",
    )
    table.add_column("Metric", style="bold", width=18)
    for model in models:
        table.add_column(model, justify="right", min_width=16)

    for key, label, lower_is_better in key_metrics:
        model_avgs = {}
        for model in models:
            valid = [r for r in results[model] if r is not None]
            vals = [r[key] for r in valid] if valid else []
            avg, *_ = stats(vals)
            if not math.isnan(avg):
                model_avgs[model] = avg

        if not model_avgs:
            continue

        best_val = (min if lower_is_better else max)(model_avgs.values())
        row = [label]
        for model in models:
            if model not in model_avgs or best_val == 0:
                row.append("[dim]n/a[/dim]")
                continue
            val = model_avgs[model]
            if lower_is_better:
                pct = ((val - best_val) / best_val) * 100
            else:
                pct = ((best_val - val) / best_val) * 100

            if abs(pct) < 0.5:
                row.append("[bold green]★ best[/bold green]")
            else:
                color = "red" if pct > 30 else "yellow" if pct > 10 else "white"
                row.append(f"[{color}]+{pct:.1f}%[/{color}]")
        table.add_row(*row)

    console.print(table)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and compare Ollama models side-by-side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
scheduling modes:
  (default)       Sequential — all runs of model A, then B, then C.
                  Best for measuring steady-state performance (model stays loaded).
  --round-robin   Interleave — cycle through models each run (A,B,C,A,B,C…).
                  Includes load/unload overhead in measurements.
  --rounds R      Multiple rounds — each round runs all models sequentially,
                  but model order is randomized per round to control for
                  thermal throttling and positional bias.
  --balanced      With --rounds, use Latin-square rotation so each model
                  occupies each position (1st, 2nd, …) an equal number of times.
                  For perfect balance, set rounds to a multiple of model count.

examples:
  ollama-bench gemma4:e4b gemma4:26b gemma4:31b -n 4
  ollama-bench gemma4:e4b gemma4:26b -n 3 --rounds 4 --cooldown 30
  ollama-bench gemma4:e4b gemma4:26b -n 3 --rounds 3 --balanced --warmup
  ollama-bench gemma4:e4b gemma4:26b -n 4 --round-robin
  ollama-bench gemma4:e4b gemma4:26b --json results.json --no-per-run
""",
    )
    parser.add_argument("models", nargs="*", help="Model name(s) to benchmark")
    parser.add_argument("--tui", action="store_true",
                        help="Launch interactive TUI instead of CLI")
    parser.add_argument("-n", "--runs", type=int, default=3,
                        help="Runs per model per round (default: 3)")
    parser.add_argument("-p", "--prompt",
                        default="Write a 200 word explanation of transformers in ML.",
                        help="Prompt to send to each model")
    parser.add_argument("--prompt-file", metavar="FILE",
                        help="Read prompt from a file instead of -p")
    parser.add_argument("--url", default="http://localhost:11434",
                        help="Ollama API base URL (default: localhost:11434)")
    parser.add_argument("--warmup", action="store_true",
                        help="Run each model once before benchmarking (not counted)")
    parser.add_argument("--round-robin", action="store_true",
                        help="Interleave models each run instead of sequential")
    parser.add_argument("--rounds", type=int, default=1, metavar="R",
                        help="Number of rounds; model order randomized per round")
    parser.add_argument("--balanced", action="store_true",
                        help="With --rounds, use Latin-square positional balancing")
    parser.add_argument("--cooldown", type=int, default=0, metavar="SEC",
                        help="Seconds to sleep between rounds (thermal recovery)")
    parser.add_argument("--json", metavar="FILE",
                        help="Export raw results to a JSON file")
    parser.add_argument("--no-per-run", action="store_true",
                        help="Skip per-run detail tables, show summary only")
    args = parser.parse_args()

    # Launch TUI if requested
    if args.tui:
        from ollama_bench_tui import BenchApp
        BenchApp().run()
        return

    if not args.models:
        parser.error("at least one model is required (or use --tui for interactive mode)")

    if args.round_robin and args.rounds > 1:
        parser.error("--round-robin and --rounds are mutually exclusive")
    if args.balanced and args.rounds <= 1:
        parser.error("--balanced requires --rounds > 1")
    if args.cooldown > 0 and args.rounds <= 1 and not args.round_robin:
        parser.error("--cooldown requires --rounds > 1")

    if args.prompt_file:
        with open(args.prompt_file) as f:
            args.prompt = f.read().strip()

    # build schedule
    schedule = build_schedule(
        args.models, args.runs, args.round_robin, args.rounds, args.balanced
    )
    n_rounds = len(schedule)
    total_per_model = args.runs * n_rounds

    # header
    console.print()
    console.print(Rule("[bold]Ollama Model Benchmark[/bold]"))
    console.print(f"  [dim]Models:[/dim]    {', '.join(args.models)}")

    if args.round_robin:
        mode_str = "round-robin"
    elif n_rounds > 1:
        mode_str = f"{n_rounds} rounds ({'balanced' if args.balanced else 'random order'})"
    else:
        mode_str = "sequential"
    console.print(f"  [dim]Mode:[/dim]      {mode_str}")
    console.print(f"  [dim]Runs:[/dim]      {args.runs}/model/round x {n_rounds} round(s) = {total_per_model} total/model")

    if args.cooldown > 0:
        console.print(f"  [dim]Cooldown:[/dim]  {args.cooldown}s between rounds")
    if args.warmup:
        console.print(f"  [dim]Warmup:[/dim]    enabled")

    prompt_display = args.prompt[:80] + ("…" if len(args.prompt) > 80 else "")
    console.print(f"  [dim]Prompt:[/dim]    {prompt_display}")

    # show schedule for multi-round
    if n_rounds > 1:
        console.print()
        for r_i, r_sched in enumerate(schedule):
            seen = []
            for m in r_sched:
                if m not in seen:
                    seen.append(m)
            console.print(f"  [dim]Round {r_i + 1}:[/dim]  {' -> '.join(seen)}")

    console.print()

    # benchmark
    results = run_benchmarks(
        args.models, schedule, args.prompt, args.url, args.warmup, args.cooldown
    )
    console.print()

    # display
    if not args.no_per_run:
        show_per_run(results, args.models, n_rounds)

    show_summary(results, args.models)
    console.print()
    show_relative(results, args.models)

    # export
    if args.json:
        def sanitize(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        export = {}
        for model in args.models:
            export[model] = [
                {k: sanitize(v) for k, v in r.items()} if r else None
                for r in results[model]
            ]
        with open(args.json, "w") as f:
            json.dump(export, f, indent=2)
        console.print(f"\n[dim]Results exported to {args.json}[/dim]")

    console.print()


if __name__ == "__main__":
    main()
