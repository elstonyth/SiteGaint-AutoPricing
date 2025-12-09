"""
Processing statistics storage module.

Tracks daily processing runs and provides summary statistics.
"""

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default path for stats CSV
DEFAULT_STATS_PATH = "data/cache/processing_stats.csv"

# CSV columns
COLUMNS = [
    "date",
    "timestamp",
    "products_processed",
    "products_updated",
    "ok_count",
    "warning_count",
    "blocked_count",
    "no_data_count",
    "unmapped_count",
    "avg_price_change_pct",
    "fx_rate",
]


@dataclass
class ProcessingRun:
    """Record of a single processing run."""

    date: str
    timestamp: str
    products_processed: int
    products_updated: int
    ok_count: int
    warning_count: int
    blocked_count: int
    no_data_count: int
    unmapped_count: int
    avg_price_change_pct: float
    fx_rate: float


@dataclass
class DailyStats:
    """Aggregated daily statistics."""

    date: str
    runs_count: int
    products_processed: int
    products_updated: int
    ok_count: int
    warning_count: int
    blocked_count: int
    avg_price_change_pct: float
    last_run: str


@dataclass
class DashboardStats:
    """Statistics for dashboard display."""

    today_processed: int = 0
    today_updated: int = 0
    today_runs: int = 0
    avg_change_pct: float = 0.0
    last_run: str | None = None
    last_run_ago: str = "Never"
    week_processed: int = 0
    week_updated: int = 0


class StatsStore:
    """Manages storage and retrieval of processing statistics."""

    def __init__(self, stats_path: str | None = None) -> None:
        """Initialize the stats store."""
        self.stats_path = Path(stats_path or DEFAULT_STATS_PATH)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create the stats file with headers if it doesn't exist."""
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.stats_path.exists():
            with open(self.stats_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
            logger.info(f"Created stats file: {self.stats_path}")

    def record_run(
        self,
        products_processed: int,
        products_updated: int,
        ok_count: int = 0,
        warning_count: int = 0,
        blocked_count: int = 0,
        no_data_count: int = 0,
        unmapped_count: int = 0,
        avg_price_change_pct: float = 0.0,
        fx_rate: float = 0.0,
    ) -> ProcessingRun:
        """Record a processing run."""
        now = datetime.now()
        run = ProcessingRun(
            date=now.strftime("%Y-%m-%d"),
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
            products_processed=products_processed,
            products_updated=products_updated,
            ok_count=ok_count,
            warning_count=warning_count,
            blocked_count=blocked_count,
            no_data_count=no_data_count,
            unmapped_count=unmapped_count,
            avg_price_change_pct=round(avg_price_change_pct, 2),
            fx_rate=round(fx_rate, 4),
        )

        # Append to CSV
        with open(self.stats_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    run.date,
                    run.timestamp,
                    run.products_processed,
                    run.products_updated,
                    run.ok_count,
                    run.warning_count,
                    run.blocked_count,
                    run.no_data_count,
                    run.unmapped_count,
                    run.avg_price_change_pct,
                    run.fx_rate,
                ]
            )

        logger.info(
            f"Recorded processing run: {products_processed} processed, {products_updated} updated"
        )
        return run

    def get_all_runs(self) -> list[ProcessingRun]:
        """Get all processing runs."""
        runs = []

        if not self.stats_path.exists():
            return runs

        try:
            with open(self.stats_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        runs.append(
                            ProcessingRun(
                                date=row["date"],
                                timestamp=row["timestamp"],
                                products_processed=int(row["products_processed"]),
                                products_updated=int(row["products_updated"]),
                                ok_count=int(row.get("ok_count", 0)),
                                warning_count=int(row.get("warning_count", 0)),
                                blocked_count=int(row.get("blocked_count", 0)),
                                no_data_count=int(row.get("no_data_count", 0)),
                                unmapped_count=int(row.get("unmapped_count", 0)),
                                avg_price_change_pct=float(row.get("avg_price_change_pct", 0)),
                                fx_rate=float(row.get("fx_rate", 0)),
                            )
                        )
                    except (ValueError, KeyError) as e:
                        logger.warning(f"Invalid stats row: {row}, error: {e}")
        except Exception as e:
            logger.error(f"Failed to read stats file: {e}")

        return runs

    def get_today_runs(self) -> list[ProcessingRun]:
        """Get all runs from today."""
        today = date.today().strftime("%Y-%m-%d")
        return [r for r in self.get_all_runs() if r.date == today]

    def get_dashboard_stats(self) -> DashboardStats:
        """Get aggregated stats for dashboard display."""
        all_runs = self.get_all_runs()
        today = date.today().strftime("%Y-%m-%d")

        # Filter today's runs
        today_runs = [r for r in all_runs if r.date == today]

        # Calculate week stats (last 7 days)
        from datetime import timedelta

        week_ago = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
        week_runs = [r for r in all_runs if r.date >= week_ago]

        stats = DashboardStats()

        # Today's stats
        if today_runs:
            stats.today_runs = len(today_runs)
            stats.today_processed = sum(r.products_processed for r in today_runs)
            stats.today_updated = sum(r.products_updated for r in today_runs)
            avg_changes = [r.avg_price_change_pct for r in today_runs if r.avg_price_change_pct]
            stats.avg_change_pct = (
                round(sum(avg_changes) / len(avg_changes), 1) if avg_changes else 0.0
            )

        # Week stats
        if week_runs:
            stats.week_processed = sum(r.products_processed for r in week_runs)
            stats.week_updated = sum(r.products_updated for r in week_runs)

        # Last run info
        if all_runs:
            last_run = max(all_runs, key=lambda r: r.timestamp)
            stats.last_run = last_run.timestamp

            # Calculate time ago
            try:
                last_dt = datetime.strptime(last_run.timestamp, "%Y-%m-%d %H:%M:%S")
                delta = datetime.now() - last_dt

                if delta.days > 0:
                    stats.last_run_ago = f"{delta.days}d ago"
                elif delta.seconds >= 3600:
                    stats.last_run_ago = f"{delta.seconds // 3600}h ago"
                elif delta.seconds >= 60:
                    stats.last_run_ago = f"{delta.seconds // 60}m ago"
                else:
                    stats.last_run_ago = "Just now"
            except ValueError:
                stats.last_run_ago = "Unknown"

        return stats


# Module-level singleton
_store: StatsStore | None = None


def _get_store() -> StatsStore:
    """Get or create the singleton store instance."""
    global _store
    if _store is None:
        _store = StatsStore()
    return _store


def record_processing_run(
    products_processed: int,
    products_updated: int,
    ok_count: int = 0,
    warning_count: int = 0,
    blocked_count: int = 0,
    no_data_count: int = 0,
    unmapped_count: int = 0,
    avg_price_change_pct: float = 0.0,
    fx_rate: float = 0.0,
) -> ProcessingRun:
    """Record a processing run (convenience function)."""
    return _get_store().record_run(
        products_processed=products_processed,
        products_updated=products_updated,
        ok_count=ok_count,
        warning_count=warning_count,
        blocked_count=blocked_count,
        no_data_count=no_data_count,
        unmapped_count=unmapped_count,
        avg_price_change_pct=avg_price_change_pct,
        fx_rate=fx_rate,
    )


def get_dashboard_stats() -> DashboardStats:
    """Get dashboard statistics (convenience function)."""
    return _get_store().get_dashboard_stats()
