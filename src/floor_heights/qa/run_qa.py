"""Run LiDAR QA on tileset data."""

import argparse
import multiprocessing as mp
from pathlib import Path

from loguru import logger

from floor_heights.qa import LidarQAPipeline


def main():
    parser = argparse.ArgumentParser(description="Run LiDAR Quality Assurance on tileset data")
    parser.add_argument("input_dirs", nargs="+", type=Path, help="Input directories containing LiDAR tiles")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/exports"),
        help="Output directory for QA reports (default: ./data/exports)",
    )
    parser.add_argument("--pattern", default="*.las", help="File pattern to match (default: *.las)")
    parser.add_argument(
        "--workers",
        type=int,
        default=mp.cpu_count(),
        help=f"Number of parallel workers (default: all {mp.cpu_count()} cores)",
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    pipeline = LidarQAPipeline(args.output)

    logger.info(f"Using {args.workers} workers for processing")

    for input_dir in args.input_dirs:
        if input_dir.exists():
            logger.info(f"Processing {input_dir}...")

            df = pipeline.process_directory(input_dir, pattern=args.pattern, max_workers=args.workers)

            if not df.empty:
                summary = pipeline.generate_summary_report(df)

                logger.info(f"\n=== {input_dir.name.upper()} SUMMARY ===")
                logger.info(f"Total tiles: {summary['total_tiles']}")
                logger.info(f"Tiles needing correction: {summary['tiles_needing_correction']}")
                logger.info(f"Average density: {summary['average_density']:.2f} pts/m²")
                logger.info(f"Issue types found: {list(summary['issue_type_frequency'].keys())}")

                region_report = args.output / f"qa_report_{input_dir.name}.parquet"
                df.to_parquet(region_report, index=False)
                logger.info(f"Saved region report to {region_report}")
        else:
            logger.warning(f"Directory not found: {input_dir}")

    logger.info("\nQA processing complete!")
    logger.info(f"Check {args.output} for detailed reports")


if __name__ == "__main__":
    main()
