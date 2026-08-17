import sys
import logging
import io
from pathlib import Path
from rich.console import Console


class BarbetLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if not isinstance(msg, str):
            try:
                buf = io.StringIO()
                Console(file=buf, force_terminal=False).print(msg)
                msg = buf.getvalue().rstrip()
            except Exception:
                msg = str(msg)
        super()._log(level, msg, args, exc_info=exc_info, extra=extra, stack_info=stack_info, stacklevel=stacklevel)


logging.setLoggerClass(BarbetLogger)


def setup_logger(output_dir: Path = None) -> logging.Logger:
    """Set up and return a logger configured with stdout and file handlers."""
    if output_dir is None:
        output_dir = Path("output")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "barbet.log"

    logger = logging.getLogger("barbet")
    logger.setLevel(logging.INFO)

    file_handler_exists = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_file.resolve()
        for h in logger.handlers
    )

    if not file_handler_exists:
        logger.handlers.clear()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # File handler
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Console handler writing to stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
