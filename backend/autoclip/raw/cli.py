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


def _default_provider() -> str:
    """Gemma on the M-series GPU when this machine can run it; otherwise the saved provider."""
    import platform
    import sys

    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    return ""


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
        "",
        "--provider",
        "-p",
        help="Who scores the moments: mlx (Gemma on the Apple GPU, the default on "
        "this Mac), openai, ollama, gemini, or anthropic.",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Model id for this run. mlx defaults to "
        "lmstudio-community/gemma-4-12B-it-MLX-8bit. openai takes any id your key can call.",
    ),
    style: str = typer.Option("", "--style", "-s", help="Caption style (see `autoclip styles`)."),
    layout: str = typer.Option(
        "auto",
        "--layout",
        help="auto: split (webcam over gameplay) where the webcam overlay is on screen, "
        "blur (whole game over a blurred background) for gameplay without it, face "
        "(speaker-tracked crop) for talking-head scenes. Or force split, blur or face.",
    ),
    cam: str = typer.Option(
        "", "--cam", help="Webcam rectangle as x,y,w,h in source pixels (skips detection)."
    ),
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
    if layout not in ("auto", "split", "blur", "face"):
        console.print("[red]--layout must be auto, split, blur or face.[/red]")
        raise typer.Exit(2)
    from .layout import Rect

    try:
        cam_rect = Rect.parse(cam) if cam else None
    except ValueError as exc:
        console.print(f"[red]--cam: {exc}[/red]")
        raise typer.Exit(2) from exc

    settings = config.load()
    if not provider:
        provider = _default_provider()
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
        model=model,
        style=style,
        layout=layout,  # type: ignore[arg-type]  # validated above
        cam=cam_rect,
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
    table.add_column("Layout", width=6)
    table.add_column("Title")
    for clip, row in zip(result.clips, result.rows, strict=True):
        title = clip.hook_title + (" [dim](needs context)[/dim]" if clip.needs_context else "")
        table.add_row(
            str(clip.rank),
            str(clip.score),
            f"{clip.duration_s:.0f}s",
            _clock(clip.start_s),
            row.layout,
            title,
        )
    console.print()
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]Note:[/yellow] {warning}")
    console.print(f"\n[green]Clips (9:16):[/green] {result.folder / 'clips'}")
    if (result.folder / "layout_preview.jpg").exists():
        console.print(
            f"[green]Webcam:[/green]       {result.folder / 'layout_detect.jpg'}  "
            "[dim](check the green box; fix with --cam x,y,w,h)[/dim]"
        )
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
