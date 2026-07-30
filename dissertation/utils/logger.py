"""utils/logger.py — CSV training log writer."""

import csv
from pathlib import Path


class Logger:
    """Appends training metrics to a CSV file each epoch."""

    def __init__(self, path: "str | Path"):
        self.path    = Path(path)
        self._header = False

    def log(self, metrics: dict):
        write_header = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
