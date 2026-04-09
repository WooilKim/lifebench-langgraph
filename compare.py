#!/usr/bin/env python3
"""
Run two graphs (base + variant) for a single user and produce a comparison report.

Usage:
    python compare.py --user fenghaoran --variant llm_enriched
    python compare.py --user fenghaoran --variant korean_local
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


VARIANT_REGISTRY = {
    "llm_enriched": None,   # loaded lazily
    "korean_local": None,
}


def _get_data_dir(user_id: str) -> str:
    base = os.getenv(
        "LIFEBENCH_DATA_DIR",
        str(Path.home() / "source" / "LifeBench" / "life_bench_data" / "data_en"),
    )
    return str(Path(base) / user_id)


def _load_variant(name: str):
    if name == "llm_enriched":
        from variants.llm_enriched import build_llm_enriched_graph, export_graph_structure
        return build_llm_enriched_graph(), export_graph_structure()
    if name == "korean_local":
        from variants.korean_local import build_korean_local_graph, export_graph_structure
        return build_korean_local_graph(), export_graph_structure()
    raise ValueError(f"Unknown variant: {name!r}. Choose llm_enriched | korean_local.")


def _sample_events(events: list, n: int = 5) -> list:
    """Return up to n events evenly distributed across the list."""
    if len(events) <= n:
        return events
    step = len(events) // n
    return [events[i * step] for i in range(n)]


def _diff_summary(base_events: list, variant_events: list) -> dict:
    base_by_id = {e["identifier"]: e for e in base_events}
    variant_by_id = {e["identifier"]: e for e in variant_events}

    common = set(base_by_id) & set(variant_by_id)
    changed = []
    for eid in common:
        if base_by_id[eid]["payload"] != variant_by_id[eid]["payload"]:
            changed.append(eid)

    return {
        "base_count": len(base_events),
        "variant_count": len(variant_events),
        "common_identifiers": len(common),
        "changed_payloads": len(changed),
        "only_in_base": len(set(base_by_id) - set(variant_by_id)),
        "only_in_variant": len(set(variant_by_id) - set(base_by_id)),
        "changed_sample": changed[:10],
    }


def run_comparison(user_id: str, variant_name: str, output_dir: Path) -> dict:
    from pipeline.graph import build_base_graph, export_graph_structure as base_structure

    data_dir = _get_data_dir(user_id)
    initial_state = {
        "user_id": user_id,
        "data_dir": data_dir,
        "persona": {},
        "daily_events": [],
        "raw_calls": [],
        "raw_sms": [],
        "raw_push": [],
        "generated_calls": [],
        "generated_sms": [],
        "generated_noti": [],
        "behavior_events": [],
        "metadata": {},
    }

    print(f"[compare] Running base graph for {user_id}…")
    t0 = time.time()
    base_graph = build_base_graph()
    base_result = base_graph.invoke(initial_state)
    base_elapsed = time.time() - t0
    print(f"[compare] Base done in {base_elapsed:.1f}s — {len(base_result['behavior_events'])} events")

    print(f"[compare] Running '{variant_name}' graph for {user_id}…")
    t0 = time.time()
    variant_graph, variant_struct = _load_variant(variant_name)
    variant_result = variant_graph.invoke(initial_state)
    variant_elapsed = time.time() - t0
    print(f"[compare] Variant done in {variant_elapsed:.1f}s — {len(variant_result['behavior_events'])} events")

    comparison = {
        "user_id": user_id,
        "variant": variant_name,
        "base_graph": base_structure(),
        "variant_graph": variant_struct,
        "base_metadata": base_result.get("metadata", {}),
        "variant_metadata": variant_result.get("metadata", {}),
        "timing": {"base_s": round(base_elapsed, 2), "variant_s": round(variant_elapsed, 2)},
        "diff": _diff_summary(
            base_result["behavior_events"],
            variant_result["behavior_events"],
        ),
        "base_sample": _sample_events(base_result["behavior_events"], 5),
        "variant_sample": _sample_events(variant_result["behavior_events"], 5),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "comparison_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"[compare] Saved comparison report → {out_path}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Compare base vs variant LangGraph pipeline")
    parser.add_argument("--user", required=True, help="User ID (e.g. fenghaoran)")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["llm_enriched", "korean_local"],
        help="Variant to compare against base",
    )
    parser.add_argument(
        "--output",
        default="viz",
        help="Output directory for comparison_result.json (default: viz/)",
    )
    args = parser.parse_args()

    report = run_comparison(args.user, args.variant, Path(args.output))

    print("\n=== Comparison Summary ===")
    diff = report["diff"]
    print(f"  Base events:    {diff['base_count']}")
    print(f"  Variant events: {diff['variant_count']}")
    print(f"  Changed payloads: {diff['changed_payloads']}")
    print(f"  Timing: base={report['timing']['base_s']}s  variant={report['timing']['variant_s']}s")


if __name__ == "__main__":
    main()
