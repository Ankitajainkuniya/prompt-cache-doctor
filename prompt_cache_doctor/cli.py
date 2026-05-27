"""prompt-cache-doctor CLI."""

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

from prompt_cache_doctor import __version__
from prompt_cache_doctor.analyzer import (
    AnalysisReport,
    analyze,
    load_records,
)
from prompt_cache_doctor.pricing import PRICING_PER_MTOK

app = typer.Typer(
    help="Diagnose your Anthropic prompt cache hit rate and find the exact reason you're missing it.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _load_pricing(path: Optional[Path]):
    if not path:
        return PRICING_PER_MTOK
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise typer.BadParameter("pricing file must be a YAML mapping")
    return data


def _render(report: AnalysisReport, verbose: bool = False) -> None:
    console.print("[bold cyan]Prompt Cache Doctor[/bold cyan]\n")
    console.rule(" Overall ", style="dim")
    console.print(f"  Calls analyzed            [bold]{report.total_calls}[/bold]")
    console.print(
        f"  Cache hit rate            [bold]{report.overall_hit_rate*100:5.1f}%[/bold]   "
        "(target: 80%+ for chat agents)"
    )
    console.print(f"  Cost paid              [green]${report.total_paid_usd:8.2f}[/green]")
    console.print(f"  Cost without caching   [yellow]${report.total_no_cache_usd:8.2f}[/yellow]")
    pct_saved = (
        (report.total_savings_so_far / report.total_no_cache_usd * 100)
        if report.total_no_cache_usd > 0
        else 0.0
    )
    console.print(
        f"  Savings so far         [green]${report.total_savings_so_far:8.2f}[/green]  ({pct_saved:.0f}% off)"
    )
    console.print(
        f"  Potential savings      [magenta]${report.total_max_possible_savings:8.2f}[/magenta]  "
        f"(you're leaving [magenta]${report.total_potential_remaining:.2f}[/magenta] on the table)"
    )

    flagged = [(r, m) for r in report.routes for m in r.miss_reasons]
    if not flagged:
        console.print("\n[green]No cache anti-patterns detected. Nice work.[/green]")
        return

    console.print("")
    console.rule(" Top miss reasons ", style="dim")
    flagged.sort(key=lambda x: x[1].est_savings_usd_per_month, reverse=True)
    for i, (route, miss) in enumerate(flagged, 1):
        console.print(
            f"  [bold]{i}.[/bold] [cyan]route={route.route}[/cyan]   [bold]{miss.label}[/bold]"
        )
        console.print(f"        [dim]{miss.explanation}[/dim]")
        console.print(f"        [yellow]fix:[/yellow] {miss.fix}")
        if miss.est_savings_usd_per_month > 0:
            console.print(
                f"        [green]est. savings: ${miss.est_savings_usd_per_month:.2f}/mo[/green]"
            )
        console.print("")

    if verbose:
        console.rule(" Per-route detail ", style="dim")
        for r in report.routes:
            console.print(
                f"  [cyan]{r.route}[/cyan]  calls={r.call_count}  hit_rate={r.hit_rate*100:.1f}%  "
                f"paid=${r.paid_usd:.2f}  no_cache=${r.no_cache_usd:.2f}"
            )


def _report_to_dict(report: AnalysisReport) -> dict:
    return {
        "total_calls": report.total_calls,
        "overall_hit_rate": report.overall_hit_rate,
        "total_paid_usd": report.total_paid_usd,
        "total_no_cache_usd": report.total_no_cache_usd,
        "total_savings_so_far": report.total_savings_so_far,
        "total_max_possible_savings": report.total_max_possible_savings,
        "routes": [
            {
                "route": r.route,
                "call_count": r.call_count,
                "hit_rate": r.hit_rate,
                "paid_usd": r.paid_usd,
                "no_cache_usd": r.no_cache_usd,
                "miss_reasons": [
                    {
                        "label": m.label,
                        "explanation": m.explanation,
                        "fix": m.fix,
                        "est_savings_usd_per_month": m.est_savings_usd_per_month,
                    }
                    for m in r.miss_reasons
                ],
            }
            for r in report.routes
        ],
    }


def analyze_cmd(
    source: str = typer.Argument(
        ..., metavar="LOG_OR_DEMO", help="Path to a JSONL log file, or the literal word 'demo'."
    ),
    pricing: Optional[Path] = typer.Option(None, "--pricing", help="Custom pricing YAML."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-route detail."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write machine-readable JSON."),
) -> None:
    """Analyze a JSONL log of Anthropic API calls and report cache health."""
    if source == "demo":
        path = Path(__file__).parent / "fixtures" / "demo_logs.jsonl"
        console.print("[dim]Demo mode — analyzing bundled sample log.[/dim]\n")
    else:
        path = Path(source)
        if not path.exists():
            raise typer.BadParameter(f"Log file not found: {path}")

    records = load_records(str(path))
    if not records:
        console.print("[yellow]No records found in log.[/yellow]")
        raise typer.Exit(1)

    pricing_data = _load_pricing(pricing)
    report = analyze(records, pricing_data)
    _render(report, verbose=verbose)

    if json_out:
        json_out.write_text(json.dumps(_report_to_dict(report), indent=2))
        console.print(f"\n[green]Wrote JSON report:[/green] {json_out}")


app.command(name="analyze")(analyze_cmd)


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"prompt-cache-doctor {__version__}")


if __name__ == "__main__":
    app()
