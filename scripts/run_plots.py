#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_csv(base: Path, name: str) -> pd.DataFrame:
    p = base / name
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser(description="Plot calibration CSVs to PNG.")
    ap.add_argument("--calib-dir", type=str, required=True, help="Diretório artifacts/calibration_<ts>")
    args = ap.parse_args()
    base = Path(args.calib_dir)

    # F1 grid
    df_f1 = _load_csv(base, "token_f1_grid.csv")
    df_f1["class_cut"] = df_f1["class_cut"].astype(float)
    df_f1["F1"] = df_f1["F1"].astype(float)

    plt.figure()
    plt.plot(df_f1["class_cut"], df_f1["F1"], marker="o")
    plt.xlabel("class_cut")
    plt.ylabel("F1")
    plt.title("F1 vs class_cut")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base / "plot_f1_vs_cut.png", dpi=150)
    plt.close()

    # TTR grids
    for fname, title in [
        ("ttr_grid_last_step.csv", "TTR vs wait_threshold (last_step)"),
        ("ttr_grid_any_step.csv", "TTR vs wait_threshold (any_step)"),
        ("ttr_grid_tail_mL.csv", "TTR vs wait_threshold (tail_m_of_last_L)"),
    ]:
        df = _load_csv(base, fname)
        for col in ["wait_thr", "rate", "TTR_P50", "TTR_P90", "TTR_avg"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        # rate
        plt.figure()
        plt.plot(df["wait_thr"], df["rate"], marker="o")
        plt.xlabel("wait_threshold")
        plt.ylabel("Detection rate")
        plt.title(title + " — rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(base / f"plot_rate_{fname.replace('.csv','')}.png", dpi=150)
        plt.close()
        # p50/p90/avg
        for metric in ["TTR_P50", "TTR_P90", "TTR_avg"]:
            plt.figure()
            plt.plot(df["wait_thr"], df[metric], marker="o")
            plt.xlabel("wait_threshold")
            plt.ylabel(metric + " (time units)")
            plt.title(title + f" — {metric}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(base / f"plot_{metric}_{fname.replace('.csv','')}.png", dpi=150)
            plt.close()

    print(f"[OUT] PNGs salvos em: {base}")


if __name__ == "__main__":
    main()
