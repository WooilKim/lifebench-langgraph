#!/usr/bin/env python3
"""CLI entry point for the full LifeBench pipeline.

Usage:
    python run_full.py --count 2 --provider claude
    python run_full.py --count 1 --provider gpt --test

This module is a thin wrapper around :mod:`server.pipeline_runner`, which is
also used by the web server so both paths share the exact same execution and
output-serialization logic.
"""
import argparse

from server.pipeline_runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the full LifeBench LangGraph pipeline")
    parser.add_argument("--count", type=int, default=1, help="Number of personas to generate")
    parser.add_argument("--provider", type=str, default="claude",
                        choices=["claude", "gpt", "glm"], help="LLM provider")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Test mode: limit draft events to 5 per persona")
    args = parser.parse_args()

    print("=== Full LifeBench Pipeline ===")
    print(f"Personas: {args.count} | Provider: {args.provider} | Output: {args.output}")
    if args.test:
        print("[TEST MODE] draft events capped at 5 per persona, faster run")
    print()

    summary = run_pipeline(
        count=args.count,
        provider=args.provider,
        output_dir=args.output,
        test_mode=args.test,
        log=print,
    )

    print(f"\nDone! {len(summary['personas'])} persona(s) saved to {summary['output_dir']}/")
    for p in summary["personas"]:
        c = p["counts"]
        print(f"  Saved {p['event_count']:4d} events → {p['output_dir']}/ "
              f"[call={c['call']} sms={c['sms']} noti={c['noti']} gmail={c['gmail']}]")


if __name__ == "__main__":
    main()
