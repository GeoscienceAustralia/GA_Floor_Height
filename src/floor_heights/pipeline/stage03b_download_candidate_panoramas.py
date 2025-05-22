#!/usr/bin/env python
"""Stage-03b: Download panoramas flagged is_chosen."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Tuple, cast

import argparse
import os
import sys
import warnings

import boto3
import yaml
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.engine import Engine


load_dotenv()
warnings.filterwarnings("ignore", message="Did not recognize type 'geometry'")

DB_URL = os.getenv("DB_CONNECTION_STRING")
if not DB_URL:
    raise ValueError("DB_CONNECTION_STRING environment variable is required")

CFG_PATH = Path(__file__).resolve().parents[3] / "config" / "common.yaml"
_common_cfg = yaml.safe_load(CFG_PATH.read_text()) if CFG_PATH.exists() else {}
REGIONS: Tuple[str, ...] = tuple(_common_cfg.get("regions", ()))
OUTPUT_ROOT = _common_cfg.get("output_root", "output")

AWS_REGION = "ap-southeast-2"
S3_BUCKET = "frontiersi-p127-floor-height-woolpert"


@dataclass(frozen=True)
class RegionSpec:
    prefix: str


SPEC: Mapping[str, RegionSpec] = {
    "tweed": RegionSpec("02_TweedHeads/01_StreetViewImagery/"),
    "wagga": RegionSpec("01_WaggaWagga/01_StreetViewImagery/"),
    "launceston": RegionSpec("03_Launceston/01_StreetViewImagery/"),
}


def extract_ucid(pano_id: str) -> str:
    import re

    stem = pano_id.removesuffix(".jpg")
    match = re.search(r"_(\d+)-", stem)
    if match:
        return match.group(1)
    raise ValueError(f"No UCID found in '{pano_id}'.")


@lru_cache(maxsize=None)
def _s3_head(client: Any, key: str) -> int:
    return cast(int, client.head_object(Bucket=S3_BUCKET, Key=key)["ContentLength"])


def resolve_s3_key(client: Any, region: str, pano_id: str) -> str:
    spec = SPEC[region]
    ucid = extract_ucid(pano_id)
    prefix = spec.prefix

    def candidate(name: str) -> str:
        return f"{prefix}{ucid}/Panoramas/{name}"

    names: Iterable[str] = (
        pano_id.replace("NoRoad.jpg", "OutsideAOI.jpg")
        if region == "wagga" and "NoRoad" in pano_id
        else pano_id,
        pano_id,
    )

    for name in names:
        key = candidate(name)
        try:
            _s3_head(client, key)
            return key
        except ClientError:
            continue

    raise FileNotFoundError(f"No object found for '{pano_id}' in {region}.")


def chosen_map(engine: Engine, region: str) -> Mapping[int, Tuple[str, ...]]:
    meta = MetaData()
    tbl = Table("panorama_candidate_views", meta, autoload_with=engine)
    stmt = select(tbl.c.building_id, tbl.c.pano_id).where(
        tbl.c.region == region, tbl.c.is_chosen.is_(True)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt)
        out: dict[int, list[str]] = {}
        for bid, pid in rows:
            out.setdefault(bid, []).append(pid)
    return {bid: tuple(pids) for bid, pids in out.items()}


def download_once(client: Any, key: str, dest: Path) -> Tuple[bool, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return False, dest.stat().st_size

    size = _s3_head(client, key)
    with open(dest, "wb") as fh:
        client.download_fileobj(S3_BUCKET, key, fh)
    return True, size


def _format_bytes(n: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    val = float(n)
    for unit in units:
        if val < 1024:
            return f"{val:.1f}{unit}"
        val = val / 1024
    return f"{val:.1f}EB"


def main() -> None:  # noqa: C901 – CLI glue is unavoidably imperative
    ap = argparse.ArgumentParser(
        description="Strict download of panoramas flagged is_chosen = TRUE",
    )
    ap.add_argument("--region", choices=REGIONS, help="Restrict to a single region")
    ap.add_argument("--output-root", default=str(OUTPUT_ROOT))
    ap.add_argument(
        "--log-level",
        default="info",
        choices=["trace", "debug", "info", "warning", "error", "critical"],
    )
    ap.add_argument("--check-existing", action="store_true")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    engine = create_engine(cast(str, DB_URL), future=True, pool_pre_ping=True)
    s3 = boto3.Session(region_name=AWS_REGION).client("s3")

    for region in (args.region,) if args.region else REGIONS:
        if region not in SPEC:
            logger.error(f"Unknown region '{region}'.")
            sys.exit(1)

        mapping = chosen_map(engine, region)
        if not mapping:
            logger.warning(f"{region}: nothing flagged is_chosen – skipping.")
            continue

        total = sum(len(pids) for pids in mapping.values())
        root = Path(args.output_root).expanduser().resolve()

        if args.check_existing:
            cap = root / region.capitalize() / "panoramas"
            low = root / region.lower() / "panoramas"
            existing = sum(
                1 for d in (cap, low) if d.exists() for _ in d.rglob("*.jpg")
            )
            logger.info(
                f"{region}: {existing}/{total} files present (check-only mode)."
            )
            continue

        cols = [
            SpinnerColumn(),
            TextColumn("[bold blue]Downloading…"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeRemainingColumn(compact=True),
            TextColumn(" • [green]OK:[/green] {task.fields[ok]}", justify="right"),
            TextColumn(
                " • [yellow]Skip:[/yellow] {task.fields[skip]}", justify="right"
            ),
            TextColumn(" • [red]Fail:[/red] {task.fields[fail]}", justify="right"),
            TextColumn(" • [cyan]Size:[/cyan] {task.fields[size]}", justify="right"),
        ]

        ok = skip = fail = bytes_total = 0
        futures: dict[Future[tuple[bool, int]], str] = {}

        def _worker(bid: int, pid: str) -> tuple[bool, int]:
            name = pid if pid.endswith(".jpg") else f"{pid}.jpg"
            dest = root / region.capitalize() / "panoramas" / str(bid) / name
            key = resolve_s3_key(s3, region, pid)
            return download_once(s3, key, dest)

        def submit(
            pool: ThreadPoolExecutor, bid: int, pid: str
        ) -> Future[tuple[bool, int]]:
            return pool.submit(_worker, bid, pid)

        with (
            ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as pool,
            Progress(*cols) as prog,
        ):
            tid = prog.add_task(region, total=total, ok=0, skip=0, fail=0, size="0B")

            for bid, pids in mapping.items():
                futures.update({submit(pool, bid, pid): pid for pid in pids})

            for fut in as_completed(futures):
                try:
                    fresh, sz = fut.result()
                    ok, skip = (ok + 1, skip) if fresh else (ok, skip + 1)
                    bytes_total += sz
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"{exc!s}")
                    fail += 1
                prog.update(
                    tid,
                    advance=1,
                    ok=ok,
                    skip=skip,
                    fail=fail,
                    size=_format_bytes(bytes_total),
                )

        logger.success(
            f"{region}: OK={ok}, Skip={skip}, Fail={fail}, Files={total}, Size={_format_bytes(bytes_total)}"
        )


if __name__ == "__main__":
    main()
