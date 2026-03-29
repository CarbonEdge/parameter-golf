"""
Evaluator: parse experiment logs and compare BPB scores.

Usage:
    python experiments/compare_results.py                    # scan all experiment dirs
    python experiments/compare_results.py --logs dir1 dir2   # specific dirs
    python experiments/compare_results.py --log file.log     # single log file
"""

import argparse
import re
import sys
from pathlib import Path


# Patterns to extract from logs
PAT_VAL_BPB = re.compile(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)")
PAT_FINAL_BPB = re.compile(r"final.*val_bpb[:\s=]+([0-9]+\.[0-9]+)", re.IGNORECASE)
PAT_POST_TTT = re.compile(r"post[_-]?ttt.*bpb[:\s=]+([0-9]+\.[0-9]+)", re.IGNORECASE)
PAT_PRE_TTT = re.compile(r"pre[_-]?ttt.*bpb[:\s=]+([0-9]+\.[0-9]+)", re.IGNORECASE)
PAT_ARTIFACT_SIZE = re.compile(r"artifact_size[:\s=]+([0-9]+)")
PAT_COMPRESSED_MB = re.compile(r"compressed[:\s=]+([0-9]+\.[0-9]+)\s*MB", re.IGNORECASE)
PAT_STEPS = re.compile(r"step[:\s=]+([0-9]+)")
PAT_WALLCLOCK = re.compile(r"wallclock[:\s=]+([0-9]+\.[0-9]+)")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    result = {"file": str(path), "bpb_history": []}

    # Collect all val_bpb values (chronological)
    for line in lines:
        m = PAT_VAL_BPB.search(line)
        if m:
            result["bpb_history"].append(float(m.group(1)))

    # Final val_bpb (last occurrence wins)
    if result["bpb_history"]:
        result["final_bpb"] = result["bpb_history"][-1]

    # Pre/post TTT
    for line in lines:
        if PAT_POST_TTT.search(line):
            result["post_ttt_bpb"] = float(PAT_POST_TTT.search(line).group(1))
        if PAT_PRE_TTT.search(line):
            result["pre_ttt_bpb"] = float(PAT_PRE_TTT.search(line).group(1))

    # Artifact size
    for line in lines:
        m = PAT_ARTIFACT_SIZE.search(line)
        if m:
            result["artifact_bytes"] = int(m.group(1))
            result["artifact_mb"] = result["artifact_bytes"] / 1_000_000
            break
        m = PAT_COMPRESSED_MB.search(line)
        if m:
            result["artifact_mb"] = float(m.group(1))
            break

    # Total steps
    step_nums = []
    for line in lines:
        m = PAT_STEPS.search(line)
        if m:
            step_nums.append(int(m.group(1)))
    if step_nums:
        result["max_step"] = max(step_nums)

    return result


def scan_experiment_dirs(base: Path) -> list[tuple[str, Path]]:
    """Find all log files under experiment directories."""
    found = []
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        for log_file in sorted(exp_dir.glob("*.log")):
            found.append((exp_dir.name, log_file))
    return found


def best_bpb(r: dict) -> float | None:
    """Return the best (lowest) BPB from a result dict."""
    candidates = []
    if "post_ttt_bpb" in r:
        candidates.append(r["post_ttt_bpb"])
    if "final_bpb" in r:
        candidates.append(r["final_bpb"])
    return min(candidates) if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Compare experiment BPB results")
    parser.add_argument("--logs", nargs="+", type=Path, help="Specific log files or dirs to parse")
    parser.add_argument("--log", type=Path, help="Single log file")
    parser.add_argument("--base", type=Path, default=Path(__file__).parent,
                        help="Base experiments directory (default: this script's directory)")
    args = parser.parse_args()

    log_files: list[tuple[str, Path]] = []

    if args.log:
        log_files.append((args.log.stem, args.log))
    elif args.logs:
        for p in args.logs:
            if p.is_dir():
                for lf in sorted(p.glob("*.log")):
                    log_files.append((p.name, lf))
            else:
                log_files.append((p.stem, p))
    else:
        log_files = scan_experiment_dirs(args.base)
        # Also check SOTA baseline
        sota = Path(__file__).parent.parent / "records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon"
        for lf in sorted(sota.glob("*.log")):
            log_files.insert(0, ("SOTA_baseline", lf))

    if not log_files:
        print("No log files found. Run experiments first, then rerun this script.")
        sys.exit(0)

    results = []
    for name, lf in log_files:
        r = parse_log(lf)
        r["name"] = name
        r["log"] = lf.name
        results.append(r)

    # Sort by best BPB
    results.sort(key=lambda r: best_bpb(r) or float("inf"))

    # Print table
    print(f"\n{'Experiment':<30} {'Log':<35} {'Best BPB':>10} {'Pre-TTT':>10} {'Post-TTT':>10} {'Size MB':>8} {'Steps':>7}")
    print("-" * 120)
    for r in results:
        bpb = best_bpb(r)
        bpb_str = f"{bpb:.4f}" if bpb else "N/A"
        pre = f"{r['pre_ttt_bpb']:.4f}" if "pre_ttt_bpb" in r else "—"
        post = f"{r['post_ttt_bpb']:.4f}" if "post_ttt_bpb" in r else "—"
        mb = f"{r['artifact_mb']:.2f}" if "artifact_mb" in r else "—"
        steps = str(r.get("max_step", "—"))
        print(f"{r['name']:<30} {r['log']:<35} {bpb_str:>10} {pre:>10} {post:>10} {mb:>8} {steps:>7}")

    print()

    # Delta vs SOTA
    sota_results = [r for r in results if r["name"] == "SOTA_baseline"]
    if sota_results:
        sota_bpb = best_bpb(sota_results[0])
        if sota_bpb:
            print(f"SOTA baseline: {sota_bpb:.4f}")
            for r in results:
                if r["name"] == "SOTA_baseline":
                    continue
                b = best_bpb(r)
                if b:
                    delta = b - sota_bpb
                    sign = "" if delta >= 0 else ""
                    print(f"  {r['name']:<28} {sign}{delta:+.4f} BPB vs SOTA")
            print()


if __name__ == "__main__":
    main()
