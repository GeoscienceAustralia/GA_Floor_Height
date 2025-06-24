"""Common progress bar utilities for the floor heights pipeline.

This module provides reusable progress bar configurations and helpers
for consistent progress reporting across all pipeline stages.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

T = TypeVar("T")

console = Console(stderr=True, force_terminal=True, force_interactive=True)


class PipelineProgress:
    """Standard progress bar for pipeline stages with common counter fields."""

    def __init__(
        self, description: str, total: int, show_elapsed: bool = False, custom_fields: dict[str, Any] | None = None
    ):
        """Initialize pipeline progress bar.

        Args:
            description: Task description
            total: Total number of items to process
            show_elapsed: Whether to show elapsed time
            custom_fields: Additional custom fields to track
        """
        self.description = description
        self.total = total
        self.show_elapsed = show_elapsed

        self.fields = {
            "suc": 0,
            "skp": 0,
            "fail": 0,
            "writes": 0,
        }

        if custom_fields:
            self.fields.update(custom_fields)

        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(compact=True),
        ]

        if show_elapsed:
            columns.append(TimeElapsedColumn())

        columns.extend(
            [
                TextColumn(" • [green]✓{task.fields[suc]}"),
                TextColumn(" • [yellow]➟{task.fields[skp]}"),
                TextColumn(" • [red]✗{task.fields[fail]}"),
                TextColumn(" • [blue]↓{task.fields[writes]}"),
            ]
        )

        for field in custom_fields or {}:
            if field not in ["suc", "skp", "fail"]:
                if field == "nt":
                    label = "NT"
                elif field == "mt":
                    label = "MT"
                elif field.startswith("avg_"):
                    columns.append(TextColumn(" • [magenta]" + field[4:].upper() + ":{task.fields[" + field + "]:.2f}"))
                    continue
                elif field.startswith("pct_"):
                    columns.append(TextColumn(" • [cyan]" + field[4:].upper() + ":{task.fields[" + field + "]:.1f}%"))
                    continue
                elif field == "sig_basic_var":
                    columns.append(TextColumn(" • [yellow]SIG_VAR:{task.fields[sig_basic_var]:.4f}"))
                    continue
                elif field == "clips":
                    columns.append(TextColumn(" • [blue]CLIPS:{task.fields[clips]}"))
                    continue
                elif field == "c/b":
                    columns.append(TextColumn(" • [cyan]C/B:{task.fields[c/b]:.1f}"))
                    continue
                elif field == "missing":
                    columns.append(TextColumn(" • [yellow]⊘{task.fields[missing]}"))
                    continue
                else:
                    label = field.upper()
                columns.append(TextColumn(" • [blue]" + label + ":{task.fields[" + field + "]}"))

        self.progress = Progress(*columns, console=console, refresh_per_second=10)
        self.task_id: TaskID | None = None
        self.lock = threading.Lock()

    def __enter__(self):
        self.progress.__enter__()
        self.task_id = self.progress.add_task(self.description, total=self.total, **self.fields)
        return self

    def __exit__(self, *args):
        self.progress.__exit__(*args)

    def update(self, field: str, increment: int = 1) -> None:
        """Update a counter field and advance progress.

        Args:
            field: Field name to update (e.g., 'suc', 'skp', 'fail')
            increment: Amount to increment by
        """
        with self.lock:
            if field in self.fields:
                self.fields[field] += increment
            if self.task_id is not None:
                self.progress.advance(self.task_id)
                self.progress.update(self.task_id, **self.fields)

    def update_multiple(self, updates: dict[str, int]) -> None:
        """Update multiple fields at once.

        Args:
            updates: Dictionary of field names to increment values
        """
        delta = sum(updates.values())
        with self.lock:
            for field, increment in updates.items():
                if field in self.fields:
                    self.fields[field] += increment
            if self.task_id is not None:
                self.progress.advance(self.task_id, delta)
                self.progress.update(self.task_id, **self.fields)

    def advance(self) -> None:
        """Advance the progress bar by one step."""
        if self.task_id is not None:
            self.progress.advance(self.task_id)

    def start(self) -> None:
        """Start the progress bar manually."""
        self.__enter__()

    def stop(self) -> None:
        """Stop the progress bar manually."""
        self.__exit__(None, None, None)

    def get_summary(self) -> str:
        """Get a summary string of the current progress."""
        parts = []
        if self.fields["suc"] > 0:
            parts.append(f"✓{self.fields['suc']}")
        if self.fields["skp"] > 0:
            parts.append(f"➟{self.fields['skp']}")
        if self.fields["fail"] > 0:
            parts.append(f"✗{self.fields['fail']}")

        for field, value in self.fields.items():
            if field not in ["suc", "skp", "fail"] and value > 0:
                parts.append(f"{field}:{value}")

        return " ".join(parts)


@contextmanager
def stage_progress(description: str, total: int, **kwargs) -> Iterator[PipelineProgress]:
    """Context manager for standard stage progress bar.

    Args:
        description: Task description
        total: Total number of items
        **kwargs: Additional arguments for PipelineProgress

    Yields:
        PipelineProgress instance
    """
    progress = PipelineProgress(description, total, **kwargs)
    with progress:
        yield progress


def clip_progress(region: str, total: int) -> PipelineProgress:
    """Create progress bar for clipping operations with standard fields."""
    return PipelineProgress(
        f"Clipping {region}",
        total,
        custom_fields={
            "nt": 0,
            "mt": 0,
        },
    )


def download_progress(description: str, total: int) -> PipelineProgress:
    """Create progress bar for download operations."""
    return PipelineProgress(
        description,
        total,
        show_elapsed=True,
        custom_fields={
            "retry": 0,
        },
    )


def processing_progress(description: str, total: int) -> PipelineProgress:
    """Create standard processing progress bar."""
    return PipelineProgress(description, total, show_elapsed=True)


def detection_progress(description: str, total: int) -> PipelineProgress:
    """Create progress bar for object detection with average detections per class."""
    return PipelineProgress(
        description,
        total,
        show_elapsed=True,
        custom_fields={
            "avg_door": 0,
            "avg_found": 0,
            "avg_garage": 0,
            "avg_stairs": 0,
            "avg_window": 0,
        },
    )


def best_view_progress(description: str, total: int) -> PipelineProgress:
    """Create progress bar for best view selection with average scores."""
    return PipelineProgress(
        description,
        total,
        show_elapsed=True,
        custom_fields={
            "avg_det": 0,
            "avg_sig": 0,
            "avg_comb": 0,
            "avg_gnd": 0,
            "pct_door": 0,
            "sig_basic_var": 0,
            "clips": 0,
            "c/b": 0,
        },
    )


def projection_progress(description: str, total: int) -> PipelineProgress:
    """Create progress bar for point cloud projection operations."""
    return PipelineProgress(
        description,
        total,
        show_elapsed=True,
        custom_fields={
            "missing": 0,
        },
    )
