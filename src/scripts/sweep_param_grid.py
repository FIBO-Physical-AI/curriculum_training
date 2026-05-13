from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "unitree_rl_lab" / "logs" / "rsl_rl" / "curriculum_go2_velocity_uniform"
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(REPO_ROOT / "src" / "results")))
SWEEP_CSV = RESULTS_DIR / "sweep_v4.csv"
SWEEP_MD = RESULTS_DIR / "sweep_v4_comparison.md"
FIGURES_DIR = RESULTS_DIR / "figures"

NUM_BINS = 8
V_MAX = 4.0
SEED = 0
ROLLOUTS_PER_BIN = 100
MAX_ITERATIONS = 3000
NUM_ENVS = 4096
MAX_WALLCLOCK_MIN = 45

SIGMA_Z_RATIO = 0.5

CSV_COLS = (
    ["run_id", "sigma_en_x", "std", "alpha_en", "mean_reward_final", "mean_ep_length_final"]
    + [f"err_b{b}" for b in range(NUM_BINS)]
    + [f"v_act_b{b}" for b in range(NUM_BINS)]
    + [f"duty_b{b}" for b in range(NUM_BINS)]
    + [f"freq_b{b}" for b in range(NUM_BINS)]
    + [f"stride_b{b}" for b in range(NUM_BINS)]
    + [f"flight_b{b}" for b in range(NUM_BINS)]
    + ["wall_clock_min"]
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sweep driver. Three modes:\n"
            "  (default)    Full 9-run orchestration: requires --sigma-low/mid/high\n"
            "  --append-run Append one run to CSV after shell-orchestrated train+eval\n"
            "  --compare    Regenerate sweep_v4_comparison.md from existing CSV\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sigma-low",  type=float, default=None, help="Low σ_en_x (required in default mode)")
    p.add_argument("--sigma-mid",  type=float, default=None, help="Mid σ_en_x (required in default mode)")
    p.add_argument("--sigma-high", type=float, default=None, help="High σ_en_x (required in default mode)")
    p.add_argument("--resume", action="store_true", help="(default mode) Skip runs already in CSV")

    p.add_argument("--append-run", action="store_true", help="Append one run's results to CSV")
    p.add_argument("--compare",    action="store_true", help="Regenerate comparison markdown only")

    p.add_argument("--run-id",    type=int,   default=None, help="(--append-run) 1-based run index")
    p.add_argument("--sigma",     type=float, default=None, help="(--append-run) σ_en_x for this run")
    p.add_argument("--std",       type=float, default=None, help="(--append-run) tracking std for this run")
    p.add_argument("--alpha-en",  type=float, default=None, help="(--append-run) energy weight for this run")
    p.add_argument("--eval-csv",  type=Path,  default=None, help="(--append-run) per-run eval CSV from eval_epte.py")
    p.add_argument("--trace-npz", type=Path,  default=None, help="(--append-run) per-run trace npz from eval_epte.py")
    p.add_argument("--wall-sec",  type=float, default=None, help="(--append-run) training wall-clock in seconds")
    p.add_argument("--ckpt",      type=Path,  default=None, help="(--append-run) checkpoint path (for log.txt lookup)")
    return p


def get_configs(sigma_low: float, sigma_mid: float, sigma_high: float) -> list[tuple[int, float, float, float]]:
    combos = [
        (sigma_low,  0.5, 1.0),
        (sigma_mid,  0.5, 1.0),
        (sigma_high, 0.5, 1.0),
        (sigma_low,  1.0, 1.0),
        (sigma_mid,  1.0, 1.0),
        (sigma_high, 1.0, 1.0),
        (sigma_low,  1.0, 0.5),
        (sigma_mid,  1.0, 0.5),
        (sigma_high, 1.0, 0.5),
    ]
    return [(i + 1, *c) for i, c in enumerate(combos)]


def get_completed_run_ids() -> set[int]:
    if not SWEEP_CSV.exists() or SWEEP_CSV.stat().st_size == 0:
        return set()
    completed = set()
    with SWEEP_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rid = int(row["run_id"])
                if row.get("err_b0") not in ("", "nan"):
                    completed.add(rid)
            except (KeyError, ValueError):
                pass
    return completed


def find_latest_checkpoint(after_time: float) -> Path | None:
    if not LOG_ROOT.exists():
        return None
    candidates = [d for d in LOG_ROOT.iterdir() if d.is_dir() and d.stat().st_mtime > after_time]
    if not candidates:
        return None
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    ckpts = sorted(latest.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def parse_mean_reward(ckpt: Path) -> tuple[float, float]:
    log_txt = ckpt.parent / "log.txt"
    if not log_txt.exists():
        return float("nan"), float("nan")
    text = log_txt.read_text(errors="replace")
    reward_matches = re.findall(r"[Mm]ean\s+(?:episode\s+)?reward\s*[:\-]\s*([\d.eE+\-]+)", text)
    eplen_matches  = re.findall(r"[Mm]ean\s+(?:episode\s+)?(?:ep\s+)?length\s*[:\-]\s*([\d.eE+\-]+)", text)
    mean_r = float(reward_matches[-1]) if reward_matches else float("nan")
    mean_l = float(eplen_matches[-1])  if eplen_matches  else float("nan")
    return mean_r, mean_l


def train_run(run_id: int, sigma: float, std: float, alpha_en: float) -> tuple[Path | None, float]:
    sigma_z = sigma * SIGMA_Z_RATIO
    env = {
        **os.environ,
        "SWEEP_SIGMA_EN_X":   str(sigma),
        "SWEEP_SIGMA_EN_Z":   str(sigma_z),
        "SWEEP_TRACK_STD":    str(std),
        "SWEEP_ENERGY_WEIGHT": str(alpha_en),
    }
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "scripts" / "train.py"),
        "--condition", "uniform",
        "--seed", str(SEED),
        "--headless",
        f"--num_envs={NUM_ENVS}",
        f"--max_iterations={MAX_ITERATIONS}",
    ]
    print(f"\n[sweep] === run {run_id}/9 === σ={sigma} σ_z={sigma_z} std={std} α_en={alpha_en}")

    before_time = time.time()
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, cwd=REPO_ROOT)
    wall_min = (time.time() - t0) / 60.0

    if proc.returncode != 0:
        print(f"[sweep] run {run_id}: train FAILED (code={proc.returncode}) wall={wall_min:.1f}min")
        return None, wall_min

    ckpt = find_latest_checkpoint(before_time)
    if ckpt is None:
        print(f"[sweep] run {run_id}: no checkpoint found after train")
    else:
        print(f"[sweep] run {run_id}: train OK  ckpt={ckpt}  wall={wall_min:.1f}min")
    return ckpt, wall_min


def eval_run(run_id: int, ckpt: Path) -> tuple[Path | None, Path | None]:
    eval_csv  = RESULTS_DIR / f"sweep_v4_run{run_id}_eval.csv"
    traces_dir = RESULTS_DIR / f"sweep_v4_run{run_id}_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "scripts" / "eval_epte.py"),
        "--condition", "uniform",
        "--seed", str(SEED),
        "--checkpoint", str(ckpt),
        "--rollouts-per-bin", str(ROLLOUTS_PER_BIN),
        "--out-csv", str(eval_csv),
        "--traces-dir", str(traces_dir),
    ]
    print(f"[sweep] run {run_id}: evaluating...")
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        print(f"[sweep] run {run_id}: eval FAILED (code={proc.returncode})")
        return None, None
    trace_npz = traces_dir / f"uniform_seed{SEED}.npz"
    return eval_csv, (trace_npz if trace_npz.exists() else None)


def parse_eval_csv(eval_csv: Path) -> dict[str, float]:
    bin_data: dict[int, dict[str, list[float]]] = {
        b: {"err": [], "v_act": [], "duty": [], "freq": [], "stride": []}
        for b in range(NUM_BINS)
    }
    with eval_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                b = int(row["bin_idx"])
            except (KeyError, ValueError):
                continue
            if 0 <= b < NUM_BINS:
                bin_data[b]["err"].append(float(row["tracking_error"]))
                bin_data[b]["v_act"].append(float(row["v_x_signed_mean"]))
                bin_data[b]["duty"].append(float(row["duty_factor"]))
                bin_data[b]["freq"].append(float(row["stride_freq_hz"]))
                bin_data[b]["stride"].append(float(row["stride_length_m"]))
    metrics: dict[str, float] = {}
    for b in range(NUM_BINS):
        d = bin_data[b]
        metrics[f"err_b{b}"]    = float(np.mean(d["err"]))    if d["err"]    else float("nan")
        metrics[f"v_act_b{b}"]  = float(np.mean(d["v_act"]))  if d["v_act"]  else float("nan")
        metrics[f"duty_b{b}"]   = float(np.mean(d["duty"]))   if d["duty"]   else float("nan")
        metrics[f"freq_b{b}"]   = float(np.mean(d["freq"]))   if d["freq"]   else float("nan")
        metrics[f"stride_b{b}"] = float(np.mean(d["stride"])) if d["stride"] else float("nan")
    return metrics


def parse_flight(trace_npz: Path) -> dict[str, float]:
    data = np.load(trace_npz)
    result: dict[str, float] = {}
    for b in range(NUM_BINS):
        key = f"contact_b{b}"
        if key not in data.files:
            result[f"flight_b{b}"] = float("nan")
            continue
        contact = data[key]
        all_airborne = ~contact.any(axis=-1)
        result[f"flight_b{b}"] = 1.0 if all_airborne.any() else 0.0
    return result


def nan_metrics() -> dict[str, float]:
    m: dict[str, float] = {}
    for b in range(NUM_BINS):
        for prefix in ("err", "v_act", "duty", "freq", "stride", "flight"):
            m[f"{prefix}_b{b}"] = float("nan")
    return m


def append_row(run_id: int, sigma: float, std: float, alpha_en: float,
               metrics: dict[str, float], mean_r: float, mean_l: float, wall_min: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not SWEEP_CSV.exists() or SWEEP_CSV.stat().st_size == 0
    row = {
        "run_id": run_id, "sigma_en_x": sigma, "std": std, "alpha_en": alpha_en,
        "mean_reward_final": mean_r, "mean_ep_length_final": mean_l,
        "wall_clock_min": round(wall_min, 2),
    }
    row.update(metrics)
    with SWEEP_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def generate_gait_diagram(traces_dir: Path, out_path: Path) -> None:
    try:
        from curriculum_rl.figures.plot_gait_diagram import plot_gait_diagram
        plot_gait_diagram(traces_dir, out_path)
        print(f"[sweep] gait diagram -> {out_path}")
    except Exception as exc:
        print(f"[sweep] gait diagram skipped: {exc}")


def generate_comparison_md() -> None:
    if not SWEEP_CSV.exists():
        return

    rows: list[dict[str, str]] = []
    with SWEEP_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        return

    lines: list[str] = ["# Sweep V4 — Comparison Table\n"]

    summary_cols = [
        "run_id", "sigma_en_x", "std", "alpha_en", "mean_reward_final",
        "err_b0", "err_b4", "err_b7",
        "duty_b0", "duty_b4", "duty_b7",
        "flight_b5", "flight_b6", "flight_b7",
        "wall_clock_min",
    ]
    lines.append("| " + " | ".join(summary_cols) + " |")
    lines.append("| " + " | ".join("---" for _ in summary_cols) + " |")
    for r in rows:
        lines.append("| " + " | ".join(_fmt_cell(r.get(c, "")) for c in summary_cols) + " |")
    lines.append("")

    lines.append("## Per-Bin Tracking Error\n")
    lines.append("| run | " + " | ".join(f"b{b}" for b in range(NUM_BINS)) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in range(NUM_BINS)) + " |")
    for r in rows:
        vals = [r.get("run_id", "")]
        for b in range(NUM_BINS):
            vals.append(_fmt_cell(r.get(f"err_b{b}", "")))
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    lines.append("")

    lines.append("## Per-Bin Duty Factor\n")
    lines.append("| run | " + " | ".join(f"b{b}" for b in range(NUM_BINS)) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in range(NUM_BINS)) + " |")
    for r in rows:
        vals = [r.get("run_id", "")]
        for b in range(NUM_BINS):
            vals.append(_fmt_cell(r.get(f"duty_b{b}", "")))
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    lines.append("")

    valid = [r for r in rows if _is_valid(r)]
    if valid:
        def sum_err(r: dict) -> float:
            return sum(_f(r.get(f"err_b{b}", "nan")) for b in range(NUM_BINS))

        def duty_diff(r: dict) -> float:
            return _f(r.get("duty_b0", "nan")) - _f(r.get("duty_b4", "nan"))

        top_track = sorted(valid, key=sum_err)[:3]
        lines.append("## Top 3 by Tracking Quality (sum tracking error across bins)\n")
        lines.append("| run | sigma_en_x | std | alpha_en | sum_err |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in top_track:
            lines.append(
                f"| {r['run_id']} | {r['sigma_en_x']} | {r['std']} | {r['alpha_en']} "
                f"| {sum_err(r):.3f} |"
            )
        lines.append("")

        top_gait = sorted(valid, key=duty_diff, reverse=True)[:3]
        lines.append("## Top 3 by Gait Differentiation (duty_b0 - duty_b4)\n")
        lines.append("| run | sigma_en_x | std | alpha_en | duty_diff |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in top_gait:
            lines.append(
                f"| {r['run_id']} | {r['sigma_en_x']} | {r['std']} | {r['alpha_en']} "
                f"| {duty_diff(r):.3f} |"
            )
        lines.append("")

        lines.append("## Winner\n")
        winners = [r for r in valid if _meets_criteria(r)]
        if winners:
            w = min(winners, key=sum_err)
            lines.append(
                f"**Run {w['run_id']}** "
                f"(σ_en_x={w['sigma_en_x']}, std={w['std']}, α_en={w['alpha_en']})"
            )
            lines.append(
                f"\nsum_err={sum_err(w):.3f}  err_b7={_fmt_cell(w.get('err_b7', ''))}  "
                f"duty_b0-duty_b4={duty_diff(w):.3f}"
            )
        else:
            best = min(valid, key=sum_err)
            lines.append("No run met all three acceptance criteria.")
            lines.append(
                f"\nBest tracking-only run: **Run {best['run_id']}** "
                f"(sum_err={sum_err(best):.3f})"
            )
    lines.append("")

    SWEEP_MD.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_MD.write_text("\n".join(lines))
    print(f"[sweep] comparison table -> {SWEEP_MD}")


def _fmt_cell(v: str) -> str:
    try:
        f = float(v)
        return "NaN" if (f != f) else f"{f:.3f}"
    except (ValueError, TypeError):
        return str(v)


def _f(v: str) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return float("nan")


def _is_valid(r: dict) -> bool:
    return r.get("err_b0", "") not in ("", "nan")


def _meets_criteria(r: dict) -> bool:
    err_b6   = _f(r.get("err_b6",   "nan"))
    err_b7   = _f(r.get("err_b7",   "nan"))
    low_errs = [_f(r.get(f"err_b{b}", "nan")) for b in range(5)]
    duty_b0  = _f(r.get("duty_b0",  "nan"))
    duty_b4  = _f(r.get("duty_b4",  "nan"))
    flight_b6 = _f(r.get("flight_b6", "0"))
    flight_b7 = _f(r.get("flight_b7", "0"))
    A = err_b7 <= 0.5 and err_b6 <= 0.5
    B = all(e <= 0.4 for e in low_errs if e == e)
    C = (duty_b0 - duty_b4 >= 0.05) or flight_b6 > 0 or flight_b7 > 0
    return A and B and C


def _mode_append(args: argparse.Namespace) -> int:
    missing = [
        f for f in ("run_id", "sigma", "std", "alpha_en")
        if getattr(args, f.replace("-", "_"), None) is None
    ]
    if missing:
        print(f"ERROR --append-run requires: {', '.join('--' + m.replace('_','-') for m in missing)}")
        return 1

    eval_csv  = args.eval_csv
    trace_npz = args.trace_npz
    ckpt      = args.ckpt
    wall_min  = (args.wall_sec / 60.0) if args.wall_sec is not None else float("nan")

    if eval_csv is not None and eval_csv.exists() and eval_csv.stat().st_size > 0:
        metrics = parse_eval_csv(eval_csv)
        if trace_npz is not None and trace_npz.exists():
            metrics.update(parse_flight(trace_npz))
        else:
            for b in range(NUM_BINS):
                metrics[f"flight_b{b}"] = float("nan")
    else:
        metrics = nan_metrics()

    mean_r, mean_l = parse_mean_reward(ckpt) if ckpt is not None else (float("nan"), float("nan"))

    append_row(args.run_id, args.sigma, args.std, args.alpha_en, metrics, mean_r, mean_l, wall_min)
    print(f"[sweep] run {args.run_id} appended to {SWEEP_CSV}")

    if trace_npz is not None and trace_npz.exists():
        traces_dir = trace_npz.parent
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        gait_out = FIGURES_DIR / f"sweep_v4_run_{args.run_id}_gait.png"
        generate_gait_diagram(traces_dir, gait_out)

    return 0


def _mode_run(args: argparse.Namespace) -> int:
    if args.sigma_low is None or args.sigma_mid is None or args.sigma_high is None:
        print("ERROR: --sigma-low, --sigma-mid, --sigma-high are required in default (run) mode")
        return 1

    configs   = get_configs(args.sigma_low, args.sigma_mid, args.sigma_high)
    completed = get_completed_run_ids() if args.resume else set()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for run_id, sigma, std, alpha_en in configs:
        if run_id in completed:
            print(f"[sweep] run {run_id}: already completed, skipping")
            continue

        try:
            ckpt, wall_min = train_run(run_id, sigma, std, alpha_en)

            if wall_min > MAX_WALLCLOCK_MIN:
                print(
                    f"[sweep] ABORT: run {run_id} took {wall_min:.1f} min "
                    f"(> {MAX_WALLCLOCK_MIN} min threshold). "
                    "Diagnose throughput before continuing."
                )
                break

            if ckpt is None:
                append_row(run_id, sigma, std, alpha_en, nan_metrics(), float("nan"), float("nan"), wall_min)
                continue

            mean_r, mean_l = parse_mean_reward(ckpt)

            eval_csv, trace_npz = eval_run(run_id, ckpt)
            if eval_csv is None:
                append_row(run_id, sigma, std, alpha_en, nan_metrics(), mean_r, mean_l, wall_min)
                continue

            metrics = parse_eval_csv(eval_csv)
            flight  = parse_flight(trace_npz) if trace_npz else {f"flight_b{b}": float("nan") for b in range(NUM_BINS)}
            metrics.update(flight)

            append_row(run_id, sigma, std, alpha_en, metrics, mean_r, mean_l, wall_min)

            traces_dir = RESULTS_DIR / f"sweep_v4_run{run_id}_traces"
            gait_out   = FIGURES_DIR / f"sweep_v4_run_{run_id}_gait.png"
            generate_gait_diagram(traces_dir, gait_out)

            print(
                f"[sweep] run {run_id} recorded  wall={wall_min:.1f}min  "
                f"err_b0={metrics.get('err_b0', float('nan')):.3f}  "
                f"err_b7={metrics.get('err_b7', float('nan')):.3f}"
            )

        except KeyboardInterrupt:
            print("[sweep] interrupted by user")
            break
        except Exception as exc:
            print(f"[sweep] run {run_id} unexpected error: {type(exc).__name__}: {exc}")
            append_row(run_id, sigma, std, alpha_en, nan_metrics(), float("nan"), float("nan"), 0.0)

    generate_comparison_md()
    return 0


def main() -> int:
    args = build_argparser().parse_args()

    if args.compare:
        generate_comparison_md()
        return 0

    if args.append_run:
        return _mode_append(args)

    return _mode_run(args)


if __name__ == "__main__":
    sys.exit(main())
