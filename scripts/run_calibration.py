#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# === Imports do projeto ===
from im12dt.data.dataset_seq import SeqDatasetConfig, UNSWSequenceDataset, seq_collate
from im12dt.models.model_dt import DecisionTransformer
from im12dt.models.temporal_embed import TimeEncodingFourier
from im12dt.models.tokens import (
    ActionTokenizer,
    CatSpec,
    RTGTokenizer,
    StateTokenizer,
    _rule_embed_dim,
)

# ------------------------- Helpers genéricos -------------------------


def _interpolate_templates(obj, ctx):
    if isinstance(obj, dict):
        return {k: _interpolate_templates(v, ctx) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_templates(v, ctx) for v in obj]
    if isinstance(obj, str):

        def repl(m):
            path = m.group(1).split(".")
            cur = ctx
            for key in path:
                cur = cur[key]
            return str(cur)

        return re.sub(r"\$\{([^}]+)\}", repl, obj)
    return obj


def _parse_grid(arg: str) -> list[float]:
    """
    Aceita:
      - lista: '0.3,0.4,0.5'
      - faixa com passo: '0.3:0.9:0.05'
    """
    arg = arg.strip()
    if ":" in arg:
        a, b, s = arg.split(":")
        a, b, s = float(a), float(b), float(s)
        n = int(math.floor((b - a) / s + 0.5)) + 1
        vals = [round(a + i * s, 10) for i in range(n)]
        # garante inclusão do ponto final (com tolerância)
        if abs(vals[-1] - b) > 1e-9:
            vals.append(b)
        return vals
    return [float(x) for x in arg.split(",") if x.strip()]


def _latest_ckpt(art_dir: Path) -> Path | None:
    cands = sorted(art_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _safe_load_ckpt(path: Path, device: torch.device) -> dict:
    """
    PyTorch 2.6+: weights_only=True por padrão. Permitimos objetos do numpy necessários
    e caímos para weights_only=False se preciso.
    """
    try:
        import numpy as _np  # noqa
        from torch.serialization import add_safe_globals

        # allowlist de funções/classes do numpy usadas em pickles antigos
        add_safe_globals(
            [
                _np.dtype,
                _np._core.multiarray.scalar,
                _np._core.multiarray._reconstruct,
            ]
        )
    except Exception:
        pass

    try:
        return torch.load(path, map_location=device)
    except Exception:
        # fallback seguro: apenas se o checkpoint é seu (fonte confiável)
        return torch.load(path, map_location=device, weights_only=False)


def _build_cat_specs(ds, cat_cols, rule, fixed_dim):
    specs = []
    for c in cat_cols:
        if hasattr(ds, "_cat_maps") and c in ds._cat_maps:
            n_tokens = len(ds._cat_maps[c])
            d = _rule_embed_dim(n_tokens, rule=rule, fixed=fixed_dim)
            specs.append(CatSpec(c, n_tokens, d))
    return specs


# ------------------------- Núcleo de calibração -------------------------


@torch.no_grad()
def collect_eval_tensors(
    model_cfg,
    trainer_cfg,
    data_cfg,
    ckpt: dict,
    device: torch.device,
    max_rows_val: int | None = None,
    batch_size: int = 128,
):
    """
    Faz 1 passagem no conjunto de validação (teste) e retorna buffers para calcular
    métricas com diferentes thresholds sem novo forward.
    """
    # -------- Dataset (usa stats/cat_maps do checkpoint se disponíveis) --------
    test_csv = str(Path(data_cfg["paths"]["test_csv"]))
    conf_val = SeqDatasetConfig(
        csv_path=test_csv,
        flow_keys=data_cfg["processing"]["flow_keys"],
        time_col=data_cfg["processing"]["time_col"],
        context_length=data_cfg["sequence"]["context_length"],
        start_action=data_cfg["sequence"]["start_action"],
        pad_token=data_cfg["sequence"]["pad_token"],
        normalize=data_cfg["processing"]["normalize"],
        label_col=data_cfg["labels"]["label_col"],
        attack_cat_col=data_cfg["labels"]["attack_cat_col"],
        categorical_cols=model_cfg["categorical"]["cols"],
    )

    stats_override = ckpt.get("norm_stats", None)
    cat_maps_override = ckpt.get("cat_maps", None)

    ds_val = UNSWSequenceDataset(
        conf_val,
        max_rows=max_rows_val,
        stats_override=stats_override,
        cat_maps_override=cat_maps_override,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=seq_collate,
    )

    # -------- Tokenizers e Modelo --------
    Dnum = ds_val.examples[0].states.shape[1]
    specs = _build_cat_specs(
        ds_val,
        model_cfg["categorical"]["cols"],
        model_cfg["categorical"]["embed_rule"],
        model_cfg["categorical"].get("fixed_dim", 16),
    )

    state_tok = StateTokenizer(Dnum, specs, model_cfg["embeddings"]["state_embed_dim"]).to(device)
    action_tok = ActionTokenizer(4, model_cfg["embeddings"]["state_embed_dim"]).to(device)
    rtg_tok = RTGTokenizer(model_cfg["embeddings"]["rtg_dim"]).to(device)
    time_tok = TimeEncodingFourier(model_cfg["embeddings"]["time_dim"], n_freq=16, use_log1p=True).to(device)

    # Carrega pesos dos tokenizers (se existirem)
    if "state_tok" in ckpt:
        state_tok.load_state_dict(ckpt["state_tok"], strict=False)
    if "action_tok" in ckpt:
        action_tok.load_state_dict(ckpt["action_tok"], strict=False)
    if "rtg_tok" in ckpt:
        rtg_tok.load_state_dict(ckpt["rtg_tok"], strict=False)
    if "time_tok" in ckpt:
        time_tok.load_state_dict(ckpt["time_tok"], strict=False)

    model = DecisionTransformer(
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg.get("Dropout", 0.1),
        n_actions=model_cfg["vocab"]["n_actions"],
    ).to(device)
    model.ensure_projections(
        model_cfg["embeddings"]["state_embed_dim"],
        model_cfg["embeddings"]["state_embed_dim"],
        model_cfg["embeddings"]["rtg_dim"],
        model_cfg["embeddings"]["time_dim"],
        device,
    )
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # -------- Buffers agregados --------
    token_y = []
    token_p = []
    token_m = []
    # Para TTR por flow:
    flows_attack_t = {}  # fid -> t_attack (min t com y==1)
    flows_last_stats = {}  # fid -> list of (t_last, argmax_last, conf_last)
    flows_any_tokens = {}  # fid -> list of (t, argmax, conf) para quaisquer passos válidos
    flows_tail_tokens = {}  # fid -> list of (t, argmax, conf, is_tail) para últimos m de cada janela
    # P/ tail_m_of_last_L precisamos reconhecer “últimos m tokens” de cada janela
    tail_m_default = int(trainer_cfg.get("inference", {}).get("tail_m", 1))
    tail_L_default = int(trainer_cfg.get("inference", {}).get("tail_L", 3))

    for batch in dl_val:
        # move para device
        for k, v in list(batch.items()):
            batch[k] = v.to(device) if torch.is_tensor(v) else v

        cats = {c: batch.get(f"cat_{c}") for c in model_cfg["categorical"]["cols"] if f"cat_{c}" in batch}

        s_emb = state_tok(batch["states"].float(), cats)
        a_emb = action_tok(batch["actions_in"])
        r_emb = rtg_tok(batch["rtg"].float())
        t_emb = time_tok(batch["delta_t"].float())
        logits = model(s_emb, a_emb, r_emb, t_emb, batch["attn_mask"])
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        probs1 = torch.softmax(logits, dim=-1)[..., 1]  # (B,K)
        conf_all, argmax_all = torch.softmax(logits, dim=-1).max(dim=-1)  # (B,K)

        # token-level buffers
        token_y.append(batch["actions_out"].reshape(-1).detach().cpu().numpy())
        token_p.append(probs1.reshape(-1).detach().cpu().numpy())
        token_m.append(batch["attn_mask"].reshape(-1).detach().cpu().numpy())

        # flow-level stats
        B, K = batch["actions_out"].shape
        lengths = batch["length"]  # (B,)
        abs_t = batch["abs_time"]  # (B,K)
        y = batch["actions_out"]  # (B,K)
        fid = batch["flow_id"]  # (B,)

        for i in range(B):
            f = int(fid[i].item())
            L = int(lengths[i].item())
            j_last = L - 1
            # earliest attack time no recorte (opcionalmente agregando por fluxo)
            y_i = y[i, :L]
            t_i = abs_t[i, :L]
            pos_mask = y_i == 1
            if pos_mask.any():
                t_pos_min = float(t_i[pos_mask].min().item())
                if f not in flows_attack_t:
                    flows_attack_t[f] = t_pos_min
                else:
                    flows_attack_t[f] = min(flows_attack_t[f], t_pos_min)

            # last-step stats
            t_last = float(abs_t[i, j_last].item())
            a_last = int(argmax_all[i, j_last].item())
            c_last = float(conf_all[i, j_last].item())
            flows_last_stats.setdefault(f, []).append((t_last, a_last, c_last))

            # any-step tokens válidos
            for j in range(L):
                flows_any_tokens.setdefault(f, []).append(
                    (
                        float(abs_t[i, j].item()),
                        int(argmax_all[i, j].item()),
                        float(conf_all[i, j].item()),
                    )
                )

            # tail_m_of_last_L: últimos m tokens desta janela
            m_tail = tail_m_default
            idx0 = max(0, L - m_tail)
            for j in range(idx0, L):
                flows_tail_tokens.setdefault(f, []).append(
                    (
                        float(abs_t[i, j].item()),
                        int(argmax_all[i, j].item()),
                        float(conf_all[i, j].item()),
                        True,
                    )
                )

    # Concatena token-level
    y_np = np.concatenate(token_y) if token_y else np.array([])
    p_np = np.concatenate(token_p) if token_p else np.array([])
    m_np = (np.concatenate(token_m) > 0.5) if token_m else np.array([], dtype=bool)

    return {
        "y": y_np,
        "p": p_np,
        "m": m_np,
        "flows_attack_t": flows_attack_t,
        "flows_last_stats": flows_last_stats,
        "flows_any_tokens": flows_any_tokens,
        "flows_tail_tokens": flows_tail_tokens,
        "tail_defaults": (tail_m_default, tail_L_default),
    }


def token_metrics_grid(y_np, p_np, m_np, grid_class_cut: list[float]):
    """F1 por limiar de probabilidade. PR-AUC é único (independe do corte)."""
    from sklearn.metrics import average_precision_score, f1_score

    # filtra máscaras válidas
    y = y_np[m_np]
    p = p_np[m_np]
    p = np.clip(np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    pr_auc = 0.0 if (y.size == 0 or y.max() == 0) else float(average_precision_score(y, p))
    rows = []
    best = (None, -1.0)  # (thr, f1)
    for thr in grid_class_cut:
        f1 = float(f1_score(y, (p >= thr).astype(int), zero_division=0))
        rows.append((thr, f1))
        if f1 > best[1]:
            best = (thr, f1)
    return pr_auc, rows, best


def summarize_ttr_last_step(flows_attack_t, flows_last_stats, thr: float):
    detected = 0
    ttrs = []
    malicious_flows = len(flows_attack_t)
    for f, t_attack in flows_attack_t.items():
        # ordenar por tempo e pegar earliest last-step que dispara
        cand = [t for (t, a, c) in sorted(flows_last_stats.get(f, [])) if (a == 1 and c >= thr)]
        if len(cand) > 0:
            t_detect = cand[0]
            # TTR >= 0; se detectar "antes", cravamos 0
            ttrs.append(max(0.0, t_detect - t_attack))
            detected += 1
    return malicious_flows, detected, ttrs


def summarize_ttr_any_step(flows_attack_t, flows_any_tokens, thr: float):
    detected = 0
    ttrs = []
    malicious_flows = len(flows_attack_t)
    for f, t_attack in flows_attack_t.items():
        cand = [t for (t, a, c) in sorted(flows_any_tokens.get(f, [])) if (a == 1 and c >= thr)]
        if len(cand) > 0:
            t_detect = cand[0]
            ttrs.append(max(0.0, t_detect - t_attack))
            detected += 1
    return malicious_flows, detected, ttrs


def summarize_ttr_tail(flows_attack_t, flows_last_stats, flows_tail_tokens, thr: float, m: int, L: int):
    """
    Considera as últimas L janelas do fluxo e, dentro de cada janela, os últimos m tokens.
    Dispara se algum desses tokens indicar a classe 1 com conf >= thr.
    """
    detected = 0
    ttrs = []
    malicious_flows = len(flows_attack_t)

    # Precisamos do "tempo" das últimas L janelas → use flows_last_stats ordenado
    for f, t_attack in flows_attack_t.items():
        last_list = sorted(flows_last_stats.get(f, []))  # (t_last, a_last, c_last)
        last_L = last_list[-L:] if L > 0 and len(last_list) > L else last_list

        # Recolhe tempos "limiares" de cada uma dessas últimas janelas a partir dos tail tokens
        # (usamos flows_tail_tokens e filtramos pelos tempos pertencentes aos últimos L last-steps)
        # Mais simples: pegue todos os tokens tail e mantenha apenas aqueles cujo tempo <= t_last da última janela
        # e estejam dentro do conjunto dos últimos L last-steps. Para robustez, aceitaremos qualquer token tail
        # cujo tempo seja >= min(t_last desses L) (proxy para "estar na cauda").
        if not last_L:
            continue
        t_min_tail = min(t for (t, _, _) in last_L)

        cand = [
            t
            for (t, a, c, is_tail) in sorted(flows_tail_tokens.get(f, []))
            if (is_tail and t >= t_min_tail and a == 1 and c >= thr)
        ]
        if len(cand) > 0:
            t_detect = cand[0]
            ttrs.append(max(0.0, t_detect - t_attack))
            detected += 1

    return malicious_flows, detected, ttrs


def ttr_grid(
    flows_attack_t,
    flows_last_stats,
    flows_any_tokens,
    flows_tail_tokens,
    grid_wait: list[float],
    tail_m: int,
    tail_L: int,
):
    """
    Retorna dicionário: modo -> lista de linhas (thr, flows+, detected, rate, p50, p90, avg, max)
    """

    def _summarize(ttrs: list[float], flows_plus: int, detected: int):
        rate = detected / max(flows_plus, 1)
        if len(ttrs) > 0:
            p50 = float(np.percentile(ttrs, 50))
            p90 = float(np.percentile(ttrs, 90))
            avg = float(np.mean(ttrs))
            worst = float(np.max(ttrs))
        else:
            p50 = p90 = avg = worst = float("nan")
        return rate, p50, p90, avg, worst

    out = {"last_step": [], "any_step": [], "tail_m_of_last_L": []}
    flows_plus = len(flows_attack_t)

    for thr in grid_wait:
        # last-step
        mf, det, ttrs = summarize_ttr_last_step(flows_attack_t, flows_last_stats, thr)
        rate, p50, p90, avg, worst = _summarize(ttrs, mf, det)
        out["last_step"].append((thr, mf, det, rate, p50, p90, avg, worst))

        # any-step
        mf, det, ttrs = summarize_ttr_any_step(flows_attack_t, flows_any_tokens, thr)
        rate, p50, p90, avg, worst = _summarize(ttrs, mf, det)
        out["any_step"].append((thr, mf, det, rate, p50, p90, avg, worst))

        # tail
        mf, det, ttrs = summarize_ttr_tail(flows_attack_t, flows_last_stats, flows_tail_tokens, thr, tail_m, tail_L)
        rate, p50, p90, avg, worst = _summarize(ttrs, mf, det)
        out["tail_m_of_last_L"].append((thr, mf, det, rate, p50, p90, avg, worst))

    return out


# ------------------------- Script principal -------------------------


def main():
    parser = argparse.ArgumentParser(description="Calibration sweep for class_cut and wait_threshold with TTR modes.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Caminho do checkpoint (.pt). Se vazio, usa o mais recente em artifacts/",
    )
    parser.add_argument(
        "--grid-cut",
        type=str,
        default="0.30:0.80:0.05",
        help="Grade de class_cut (ex: '0.3,0.4,0.5' ou '0.3:0.9:0.05').",
    )
    parser.add_argument(
        "--grid-wait",
        type=str,
        default="0.30,0.40,0.50,0.55,0.60,0.70,0.80",
        help="Grade de wait_threshold (ex: '0.5,0.6' ou '0.4:0.9:0.05').",
    )
    parser.add_argument("--max-rows-val", type=int, default=None, help="Limite de linhas do CSV de teste (debug).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Diretório para salvar CSVs. Default: artifacts/calibration_<ts>",
    )
    parser.add_argument(
        "--tail-m",
        type=int,
        default=None,
        help="Override de m para tail_m_of_last_L (default: trainer.yaml).",
    )
    parser.add_argument(
        "--tail-L",
        type=int,
        default=None,
        help="Override de L para tail_m_of_last_L (default: trainer.yaml).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] torch={torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_build={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[DEVICE] gpu={torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}")

    # --- Configs ---
    cfg_data = yaml.safe_load(Path("configs/data.yaml").read_text())
    cfg_data = _interpolate_templates(cfg_data, cfg_data)
    cfg_model = yaml.safe_load(Path("configs/model_dt.yaml").read_text())
    cfg_trn = yaml.safe_load(Path("configs/trainer.yaml").read_text())

    # --- Checkpoint ---
    ckpt_path = Path(args.ckpt) if args.ckpt else None
    if not ckpt_path:
        art = Path("artifacts")
        latest = _latest_ckpt(art)
        if latest is None:
            raise FileNotFoundError("Nenhum checkpoint encontrado em 'artifacts/'. Treine o modelo antes.")
        print(f"[INFO] Using latest checkpoint: {latest}")
        ckpt_path = latest
    else:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

    ckpt = _safe_load_ckpt(ckpt_path, device=device)

    # --- Coleta única dos tensores para varrer thresholds ---
    buffers = collect_eval_tensors(
        cfg_model,
        cfg_trn,
        cfg_data,
        ckpt,
        device,
        max_rows_val=args.max_rows_val,
        batch_size=args.batch_size,
    )

    # --- Grades ---
    grid_cut = _parse_grid(args.grid_cut)
    grid_wait = _parse_grid(args.grid_wait)
    tail_m_default, tail_L_default = buffers["tail_defaults"]
    tail_m = args.tail_m if args.tail_m is not None else tail_m_default
    tail_L = args.tail_L if args.tail_L is not None else tail_L_default

    # --- Métricas por token (F1 vs class_cut) ---
    pr_auc, f1_rows, best_cut = token_metrics_grid(buffers["y"], buffers["p"], buffers["m"], grid_cut)

    # --- TTR por wait_threshold (3 modos) ---
    ttr = ttr_grid(
        buffers["flows_attack_t"],
        buffers["flows_last_stats"],
        buffers["flows_any_tokens"],
        buffers["flows_tail_tokens"],
        grid_wait,
        tail_m,
        tail_L,
    )

    # --- Saída: CSVs ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.outdir) if args.outdir else Path("artifacts") / f"calibration_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    # token-level
    import csv

    with (outdir / "token_f1_grid.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_cut", "F1"])
        for thr, f1 in f1_rows:
            w.writerow([f"{thr:.4f}", f"{f1:.6f}"])

    with (outdir / "ttr_grid_last_step.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "wait_thr",
                "flows_plus",
                "detected",
                "rate",
                "TTR_P50",
                "TTR_P90",
                "TTR_avg",
                "TTR_max",
            ]
        )
        for row in ttr["last_step"]:
            thr, mf, det, rate, p50, p90, avg, worst = row
            w.writerow(
                [
                    f"{thr:.4f}",
                    mf,
                    det,
                    f"{rate:.6f}",
                    f"{p50:.6f}",
                    f"{p90:.6f}",
                    f"{avg:.6f}",
                    f"{worst:.6f}",
                ]
            )

    with (outdir / "ttr_grid_any_step.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "wait_thr",
                "flows_plus",
                "detected",
                "rate",
                "TTR_P50",
                "TTR_P90",
                "TTR_avg",
                "TTR_max",
            ]
        )
        for row in ttr["any_step"]:
            thr, mf, det, rate, p50, p90, avg, worst = row
            w.writerow(
                [
                    f"{thr:.4f}",
                    mf,
                    det,
                    f"{rate:.6f}",
                    f"{p50:.6f}",
                    f"{p90:.6f}",
                    f"{avg:.6f}",
                    f"{worst:.6f}",
                ]
            )

    with (outdir / "ttr_grid_tail_mL.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "wait_thr",
                "flows_plus",
                "detected",
                "rate",
                "TTR_P50",
                "TTR_P90",
                "TTR_avg",
                "TTR_max",
                "m",
                "L",
            ]
        )
        for row in ttr["tail_m_of_last_L"]:
            thr, mf, det, rate, p50, p90, avg, worst = row
            w.writerow(
                [
                    f"{thr:.4f}",
                    mf,
                    det,
                    f"{rate:.6f}",
                    f"{p50:.6f}",
                    f"{p90:.6f}",
                    f"{avg:.6f}",
                    f"{worst:.6f}",
                    tail_m,
                    tail_L,
                ]
            )

    # --- Resumo no console ---
    print(f"\n[SUMMARY] Checkpoint: {ckpt_path}")
    print(f"Token-level PR-AUC={pr_auc:.6f}")
    print("Best F1 by class_cut:")
    print(f"  class_cut={best_cut[0]:.4f}  F1={best_cut[1]:.6f}")

    def _best_wait(criteria_rows):
        # escolhe por menor TTR_P90 entre aqueles com rate >= 0.99; se nenhum, maior rate; empate: menor P90
        best = None
        for thr, mf, det, rate, p50, p90, avg, worst in criteria_rows:
            tup = None
            if rate >= 0.99:
                tup = (0, p90, -rate, thr)  # prioridade P90 baixo
            else:
                tup = (1, -rate, p90, thr)  # prioridade maior rate
            if best is None or tup < best[0]:
                best = (tup, thr, rate, p50, p90, avg, worst)
        return None if best is None else best[1:]

    best_last = _best_wait(ttr["last_step"])
    best_any = _best_wait(ttr["any_step"])
    best_tail = _best_wait(ttr["tail_m_of_last_L"])

    def _print_best(name, best):
        if best is None:
            print(f"{name}: no candidates.")
            return
        thr, rate, p50, p90, avg, worst = best
        print(
            f"{name}: wait_thr={thr:.4f} rate={rate:.4f} TTR_P50={p50:.6f} TTR_P90={p90:.6f} TTR_avg={avg:.6f} TTR_max={worst:.6f}"
        )

    print("\nBest wait_threshold per TTR mode (rule: minimize P90 if rate>=0.99, else maximize rate):")
    _print_best("last_step", best_last)
    _print_best("any_step", best_any)
    _print_best("tail_m_of_last_L", best_tail)

    # dump também um pequeno README de calibração
    with (outdir / "calibration_summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        f.write(f"Token-level PR-AUC={pr_auc:.6f}\n")
        f.write(f"Best F1 @ class_cut={best_cut[0]:.4f} → {best_cut[1]:.6f}\n\n")
        f.write("Best wait_threshold per TTR mode (rule: minimize P90 if rate>=0.99, else maximize rate):\n")
        for name, best in [
            ("last_step", best_last),
            ("any_step", best_any),
            ("tail_m_of_last_L", best_tail),
        ]:
            if best is None:
                f.write(f"{name}: no candidates\n")
            else:
                thr, rate, p50, p90, avg, worst = best
                f.write(
                    f"{name}: wait_thr={thr:.4f} rate={rate:.4f} TTR_P50={p50:.6f} TTR_P90={p90:.6f} TTR_avg={avg:.6f} TTR_max={worst:.6f}\n"
                )

    print(f"\n[OUT] CSVs salvos em: {outdir.resolve()}")


if __name__ == "__main__":
    main()
