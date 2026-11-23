"""The radar command line."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from .models import Store, atomic_write, web_url, yaml_read
from .pipeline import collect
from .report import report, week_bounds, week_id


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("天数必须大于 0")
    return number


def add_lead(path, url, note, excerpt):
    entry = {"url": web_url(url)}
    if note:
        entry["note"] = note
    if excerpt:
        entry["excerpt"] = excerpt
    config = yaml_read(path) if path.exists() else {}
    existing = config.setdefault("leads", [])
    if entry not in existing:
        existing.append(entry)
        atomic_write(path, yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print("线索已登记。提交 config/leads.yaml 并推送到 main，或经 PR 合入，定时任务即可读取。")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="radar")
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--data-dir", type=Path, default=Path(".radar-data"))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--lookback-days", type=positive_int)
    run.add_argument("--channel", action="append", default=[], help="仅采集指定已启用渠道，可重复")
    weekly = commands.add_parser("report")
    dates = weekly.add_mutually_exclusive_group()
    dates.add_argument("--week")
    dates.add_argument("--previous-week", action="store_true")
    weekly.add_argument("--backend", choices=("none", "deepseek", "openai"), default="none")
    weekly.add_argument("--output-dir", type=Path, default=Path("reports/weekly"))
    weekly.add_argument("--prompt", type=Path, default=Path("prompts/weekly_digest.md"))
    lead = commands.add_parser("add")
    lead.add_argument("url")
    lead.add_argument("--note", default="")
    lead.add_argument("--excerpt", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    now = datetime.now(UTC)
    try:
        if args.command == "add":
            add_lead(args.config_dir / "leads.yaml", args.url, args.note, args.excerpt)
        elif args.command == "collect":
            stats = collect(args.config_dir, Store(args.data_dir), now, args.lookback_days, channels=args.channel)
            print(f"本轮新增 {sum(stats.values())} 条记录。")
        else:
            current_start, _ = week_bounds(week_id(now))
            week = week_id(current_start - timedelta(days=1)) if args.previous_week else args.week or week_id(now)
            print(report(Store(args.data_dir), args.output_dir, week, now, args.backend, args.prompt, args.config_dir))
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"radar: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
