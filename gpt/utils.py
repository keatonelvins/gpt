from pathlib import Path

from loguru import logger


def setup_logger(run_dir: Path | None = None) -> None:
    """Setup logging to console and run directory."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="{time:HH:mm:ss} | {level} | {file}:{line} | {message}",
    )
    if run_dir is not None:
        logger.add(
            run_dir / "run.log",
            format="{time:HH:mm:ss} | {level} | {file}:{line} | {message}",
        )
