#!/usr/bin/env python3
"""CLI entry point for the full LifeBench pipeline.

Usage:
    python run_full.py --count 2 --provider claude
    python run_full.py --count 1 --provider gpt
"""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the full LifeBench LangGraph pipeline")
    parser.add_argument("--count", type=int, default=1, help="Number of personas to generate")
    parser.add_argument("--provider", type=str, default="claude",
                        choices=["claude", "gpt", "glm"], help="LLM provider")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Test mode: limit draft events to 5 per persona")
    args = parser.parse_args()

    print(f"=== Full LifeBench Pipeline ===")
    print(f"Personas: {args.count} | Provider: {args.provider} | Output: {args.output}")
    print()

    from pipeline.full_graph import build_full_graph

    graph = build_full_graph()

    if args.test:
        print("[TEST MODE] draft events capped at 5 per persona, faster run")

    initial_state = {
        "count": args.count,
        "provider": args.provider,
        "test_mode": args.test,
        "persons": [],   # populated by person_gen
        "personas": [],
        "daily_drafts": {},
        "daily_events_map": {},
        "raw_calls": [],
        "raw_sms": [],
        "raw_push": [],
        "persona_event_id_map": {},
        "generated_calls": [],
        "generated_sms": [],
        "generated_noti": [],
        "behavior_events": [],
        "behavior_events_map": {},
        "metadata": {"start_time": time.time(), "provider": args.provider},
    }

    print("Running pipeline...")
    t0 = time.time()
    result = graph.invoke(initial_state)
    elapsed = time.time() - t0
    print(f"\nPipeline completed in {elapsed:.1f}s")

    # ── Save per-persona output ────────────────────────────────────────────────
    output_root = Path(args.output)
    behavior_events_map: dict = result.get("behavior_events_map", {})
    personas: list = result.get("personas", [])

    if not behavior_events_map:
        print("[WARNING] behavior_events_map is empty — saving combined output instead")
        behavior_events_map = {"all": result.get("behavior_events", [])}

    saved = []
    for persona in personas:
        name = persona.get("name", "unknown")
        events = behavior_events_map.get(name, [])
        out_dir = output_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        events_path = out_dir / "behavior_events.json"
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        persona_path = out_dir / "persona.json"
        with open(persona_path, "w", encoding="utf-8") as f:
            json.dump(persona, f, ensure_ascii=False, indent=2)

        saved.append((name, len(events), str(events_path)))
        print(f"  Saved {len(events):4d} events → {events_path}")

    # ── Save metadata ──────────────────────────────────────────────────────────
    metadata = result.get("metadata", {})
    metadata["elapsed_seconds"] = elapsed
    meta_path = output_root / "metadata.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(saved)} persona(s) saved to {output_root}/")
    for name, n_events, path in saved:
        print(f"  {name}: {n_events} events → {path}")


if __name__ == "__main__":
    main()
