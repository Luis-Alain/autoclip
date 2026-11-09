"""``autoclip raw`` — the command for the raw-clips pipeline."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .. import config

console = Console()


def raw(
    target: str = typer.Argument(..., help="Path to a video file, or a YouTube link."),
    lang: str = typer.Option("es", "--lang", help="Spoken language: es, en, ... or auto."),
    chat: str = typer.Option(
        "none", "--chat", help="Chat source. Only 'none' for now; chat spikes arrive in M4."
    ),
    top: int = typer.Option(20, "--top", "-n", min=1, help="Maximum number of clips."),
    min_s: float = typer.Option(15.0, "--min", help="Minimum clip length in seconds."),
    max_s: float = typer.Option(60.0, "--max", help="Maximum clip length in seconds."),
    provider: str = typer.Option(
        "", "--provider", "-p", help="LLM provider (ollama, gemini, openai, anthropic)."
    ),
    style: str = typer.Option("", "--style", "-s", help="Caption style (see `autoclip styles`)."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Output directory."),
) -> None:
    """Find the best moments in a VOD and export them as 9:16 clips with captions."""
    from .runner import RawOptions, RawRunner
    from .source import SourceError, resolve

    if chat != "none":
        console.print(
            "[yellow]--chat is not implemented yet (M4); continuing without chat.[/yellow]"
        )
    if min_s >= max_s:
        console.print("[red]--min must be smaller than --max.[/red]")
        raise typer.Exit(2)

    settings = config.load()
    # Absolute: upstream's caption export runs ffmpeg from another working
    # directory, where a relative output path would point somewhere else.
    out = out.expanduser().resolve()

    try:
        with console.status("[cyan]Reading source...", spinner="dots"):
            source, folder = resolve(target, out, settings)
    except SourceError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    log_path = _setup_logging(folder / "logs")
    minutes = source.duration_s / 60
    console.print(
        f"[green]Source:[/green] {source.title} ({minutes:.1f} min, "
        f"{source.width}x{source.height} @ {source.fps or 0:.2f} fps)"
    )
    console.print(f"[green]Output:[/green] {folder}")

    options = RawOptions(
        lang=lang,
        top=top,
        min_s=min_s,
        max_s=max_s,
        provider=provider or None,
        style=style,
    )

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting", total=1.0)
        runner = RawRunner(
            source,
            folder,
            options,
            settings,
            on_progress=lambda overall, msg: progress.update(
                task, completed=overall, description=msg
            ),
        )
        try:
            result = asyncio.run(runner.run())
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow] Run the same command to resume.")
            raise typer.Exit(130) from None
        except Exception as exc:
            progress.stop()
            logging.getLogger("autoclip").exception("Raw run failed.")
            console.print(f"\n[red]Failed:[/red] {exc}")
            console.print(
                f"Details in {log_path}. Run the same command again to resume — finished "
                "stages are reused."
            )
            raise typer.Exit(1) from exc

    table = Table(title=f"{len(result.clips)} clips", header_style="bold", title_justify="left")
    table.add_column("#", width=3)
    table.add_column("Score", width=5)
    table.add_column("Length", width=6)
    table.add_column("At", width=8)
    table.add_column("Title")
    for clip in result.clips:
        title = clip.hook_title + (" [dim](needs context)[/dim]" if clip.needs_context else "")
        table.add_row(
            str(clip.rank),
            str(clip.score),
            f"{clip.duration_s:.0f}s",
            _clock(clip.start_s),
            title,
        )
    console.print()
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]Note:[/yellow] {warning}")
    console.print(f"\n[green]Clips (9:16):[/green] {result.folder / 'clips'}")
    console.print(f"[green]Manifest:[/green]     {result.folder / 'manifest.csv'}")
    console.print(
        f"[green]Premiere:[/green]     {result.folder / 'timeline.xml'}  [dim](File → Import)[/dim]"
    )


def _setup_logging(directory: Path) -> Path:
    """Full DEBUG log (incl. every ffmpeg command) to a file; warnings to the console."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"run-{dt.datetime.now():%Y%m%d-%H%M%S}.log"

    logger = logging.getLogger("autoclip")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)

    console_handler = RichHandler(console=console, show_path=False, show_time=False)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)
    return path


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
