import csv
import time
from datetime import datetime


class EventLog:
    def __init__(self, config):
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = config.logs_dir / f"session_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
        self.config = config
        self.started = time.monotonic()
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            (
                "timestamp",
                "event",
                "state",
                "faces",
                "similarity",
                "threshold",
                "processing_ms",
                "confirmation_ms",
                "pad_v2",
                "pad_v1se",
                "pad_cutoff",
                "pad_ready",
                "presentation",
                "elapsed_ms",
                "quality_issue",
            )
        )

    def write(self, event, controller, packet=None, confirmation_ms=None):
        p = packet or {}
        score = p.get("score")
        self.writer.writerow(
            (
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
                event,
                controller.state,
                p.get("count", ""),
                "" if score is None else round(score, 4),
                self.config.threshold,
                p.get("processing_ms", ""),
                "" if confirmation_ms is None else round(confirmation_ms, 1),
                *(p.get("pad_scores") or ("", "")),
                self.config.pad_threshold,
                p.get("pad_ok", False),
                self.config.label,
                round((time.monotonic() - self.started) * 1000, 1),
                p.get("problem") or "",
            )
        )
        self.file.flush()

    def close(self):
        self.file.close()
