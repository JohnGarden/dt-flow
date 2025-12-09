#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="numpy.core is deprecated", category=DeprecationWarning)

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ====== Projeto: datasets e modelo (fallback) ======
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


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
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


def _latest_ckpt(artifacts_dir: Path) -> Optional[Path]:
    if not artifacts_dir.exists():
        return None
    cands = sorted(artifacts_dir.glob("dt_baseline_*.pt"))
    return cands[-1] if cands else None


def _print_device():
    print(f"[DEVICE] torch={torch.__version__} | cuda_available={torch.cuda.is_available()} | cuda_build={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[DEVICE] gpu={torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}")


# ---------------------------------------------------
# Engine de Streaming - construção robusta
# ---------------------------------------------------
def _make_stream_engine(ckpt_path: Path, cfg_model: dict, device: torch.device):
    """
    Tenta construir o engine de streaming usando im12dt.runtime.StreamingDT,
    cobrindo variações de assinatura de from_checkpoint.

    Importante: passamos **o caminho** (str/Path) para evitar o erro de 'seek'
    que ocorre quando se passa um dict já carregado.
    """
    try:
        from im12dt.runtime.stream_dt import StreamingDT  # preferido
    except Exception:
        StreamingDT = None

    if StreamingDT is not None:
        # Tentar várias assinaturas
        # 1) (ckpt_path, cfg_model, device)
        try:
            return StreamingDT.from_checkpoint(str(ckpt_path), cfg_model, device)
        except TypeError:
            pass
        except Exception:
            pass

        # 2) keywords: (ckpt_path, model_cfg=..., device=...)
        for kws in (
            dict(model_cfg=cfg_model, device=device),
            dict(cfg_model=cfg_model, device=device),
            dict(device=device),
            dict(model_cfg=cfg_model),
            dict(cfg_model=cfg_model),
            {},
        ):
            try:
                return StreamingDT.from_checkpoint(str(ckpt_path), **kws)
            except Exception:
                continue

        # 3) Construtor direto (pouco provável)
        for ctor in (
            lambda: StreamingDT(str(ckpt_path), device=device),
            lambda: StreamingDT(str(ckpt_path)),
        ):
            try:
                return ctor()
            except Exception:
                pass

    # Fallback: engine mínimo usando os módulos do projeto
    print("[WARN] StreamingDT indisponível/assinatura incompatível; usando fallback offline.")
    return _FallbackStreamingDT(ckpt_path=ckpt_path, device=device, cfg_model=cfg_model)


# ---------------------------------------------------
# Fallback simples (forward offline por janela)
# ---------------------------------------------------
class _FallbackStreamingDT:
    """
    Carrega modelo + tokenizers do checkpoint e expõe uma API mínima:
      - infer_batch(states, actions_in, rtg, delta_t, attn_mask, cats_dict) -> logits
    Isso nos permite simular a lógica de "stream" varrendo janelas do dataset.
    """

    def __init__(self, ckpt_path: Path, device: torch.device, cfg_model: dict):
        self.device = device

        # Carrega o checkpoint com allowlist para numpy antigos (PyTorch 2.6+)
        import numpy as _np
        from torch.serialization import add_safe_globals

        add_safe_globals(
            [
                _np.core.multiarray._reconstruct,
                _np.core.multiarray.scalar,
            ]
        )
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

        self.norm_stats = ckpt.get("norm_stats", None)
        self.cat_maps = ckpt.get("cat_maps", None)
        self.cfg_model = cfg_model

        # Tokenizers
        # Precisamos conhecer Dnum em tempo de execução → inferimos do checkpoint se existir,
        # senão será obtido na primeira chamada (late-bind).
        self._lazy_tokenizers_ready = False
        self._tok_dims = None  # tuple(E_state, E_action, E_rtg, E_time)

        # Modelo
        self.model = DecisionTransformer(
            d_model=cfg_model["d_model"],
            n_layers=cfg_model["n_layers"],
            n_heads=cfg_model["n_heads"],
            d_ff=cfg_model["d_ff"],
            dropout=cfg_model.get("Dropout", 0.1),
            n_actions=cfg_model["vocab"]["n_actions"],
        ).to(self.device)

        # Projeções
        E_state = cfg_model["embeddings"]["state_embed_dim"]
        E_action = cfg_model["embeddings"]["state_embed_dim"]
        E_rtg = cfg_model["embeddings"]["rtg_dim"]
        E_time = cfg_model["embeddings"]["time_dim"]
        self.model.ensure_projections(E_state, E_action, E_rtg, E_time, device=self.device)

        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.eval()

        # Tente restaurar tokenizers (se presentes)
        self.state_tok = None
        self.action_tok = ActionTokenizer(4, E_state).to(self.device)
        self.rtg_tok = RTGTokenizer(E_rtg).to(self.device)
        self.time_tok = TimeEncodingFourier(E_time, n_freq=16, use_log1p=True).to(self.device)

        st = ckpt.get("state_tok", None)
        if st is not None:
            # Não temos Dnum salvo; criaremos sob demanda no primeiro batch
            pass

        at = ckpt.get("action_tok", None)
        if at is not None:
            self.action_tok.load_state_dict(at)

        rt = ckpt.get("rtg_tok", None)
        if rt is not None:
            self.rtg_tok.load_state_dict(rt)

        tt = ckpt.get("time_tok", None)
        if tt is not None:
            self.time_tok.load_state_dict(tt)

    def _maybe_build_state_tok(self, Dnum: int):
        if self._lazy_tokenizers_ready:
            return
        cat_cols = self.cfg_model["categorical"]["cols"]
        specs: List[CatSpec] = []
        if self.cat_maps:
            rule = self.cfg_model["categorical"].get("embed_rule", "sqrt")
            fixed = self.cfg_model["categorical"].get("fixed_dim", 16)
            for c in cat_cols:
                if c in self.cat_maps:
                    n_tokens = len(self.cat_maps[c])
                    d = _rule_embed_dim(n_tokens, rule=rule, fixed=fixed)
                    specs.append(CatSpec(c, n_tokens, d))
        self.state_tok = StateTokenizer(
            numeric_dim=Dnum,
            cat_specs=specs,
            state_embed_dim=self.cfg_model["embeddings"]["state_embed_dim"],
        ).to(self.device)
        self._lazy_tokenizers_ready = True

    @torch.no_grad()
    def infer_batch(self, batch: Dict[str, torch.Tensor], cat_cols: List[str]) -> torch.Tensor:
        """
        Recebe um batch do DataLoader (seq_collate) e retorna logits (B,K,C).
        """
        for k, v in list(batch.items()):
            if torch.is_tensor(v):
                batch[k] = v.to(self.device)

        Dnum = int(batch["states"].shape[-1])
        self._maybe_build_state_tok(Dnum)

        cats = {c: batch[f"cat_{c}"] for c in cat_cols if f"cat_{c}" in batch}

        s_emb = self.state_tok(batch["states"].float(), cats)
        a_emb = self.action_tok(batch["actions_in"])
        r_emb = self.rtg_tok(batch["rtg"].float())
        t_emb = self.time_tok(batch["delta_t"].float())

        logits = self.model(s_emb, a_emb, r_emb, t_emb, batch["attn_mask"])
        return torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)


# ---------------------------------------------------
# Detecção de eventos em “modo streaming” (simulado)
# ---------------------------------------------------
def _detect_events_from_logits(
    logits: torch.Tensor,  # (B,K,C)
    attn_mask: torch.Tensor,  # (B,K)
    wait_thr: float,
    class_cut: float,
    mode: str = "any_step",  # "any_step" | "last_step" | "tail_m_of_last_L"
    tail_m: int = 1,
    tail_L: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (event_mask, conf_max) por janela (B,), onde event_mask=1 indica detecção.

    Critério:
      - prob = softmax(logits)[...,1]
      - conf = max(prob, 1-prob)  (confiança do argmax)
      - ação=1 se prob>=class_cut e conf>=wait_thr (senão wait/benign)
    """
    probs = torch.softmax(logits, dim=-1)[..., 1]  # (B,K)
    conf, argmax = torch.softmax(logits, dim=-1).max(dim=-1)  # (B,K)

    valid = attn_mask.bool()
    prob1 = probs.masked_fill(~valid, 0.0)
    conf = conf.masked_fill(~valid, 0.0)
    argmx = argmax.masked_fill(~valid, 0)  # 0 or 1

    # condição de "ação 1" (malicioso)
    cond = (argmx == 1) & (conf >= wait_thr) & (prob1 >= class_cut)

    B, K = cond.shape
    event = torch.zeros(B, dtype=torch.bool, device=cond.device)
    conf_out = torch.zeros(B, dtype=torch.float32, device=cond.device)

    if mode == "any_step":
        event = cond.any(dim=1)
        conf_out = prob1.max(dim=1).values

    elif mode == "last_step":
        # só olha o último passo válido
        lengths = valid.long().sum(dim=1)  # (B,)
        idx = (lengths - 1).clamp_min(0)
        gather = torch.arange(B, device=cond.device)
        event = cond[gather, idx]
        conf_out = prob1[gather, idx]

    elif mode == "tail_m_of_last_L":
        # últimos L (válidos); evento se >=m passos satisfazem a condição
        lengths = valid.long().sum(dim=1)
        event = torch.zeros(B, dtype=torch.bool, device=cond.device)
        conf_out = torch.zeros(B, dtype=torch.float32, device=cond.device)
        for i in range(B):
            L = int(lengths[i].item())
            if L == 0:
                continue
            j0 = max(0, L - tail_L)
            tail = cond[i, j0:L]
            tail_prob = prob1[i, j0:L]
            event[i] = tail.sum() >= tail_m
            conf_out[i] = float(tail_prob.max().item()) if tail_prob.numel() > 0 else 0.0
    else:
        raise ValueError(f"Unknown mode={mode}")

    return event.detach().cpu().numpy().astype(np.int32), conf_out.detach().cpu().numpy()


# ---------------------------------------------------
# Main
# ---------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Streaming DT inference (simulada) sobre UNSW_NB15.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Caminho do checkpoint (.pt). Se vazio, usa o mais recente em artifacts/",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Limite de linhas do CSV de teste")
    parser.add_argument(
        "--mode",
        type=str,
        default="any_step",
        choices=["any_step", "last_step", "tail_m_of_last_L"],
    )
    parser.add_argument("--tail-m", type=int, default=1)
    parser.add_argument("--tail-L", type=int, default=3)
    parser.add_argument("--wait-thr", type=float, default=0.55)
    parser.add_argument("--class-cut", type=float, default=0.50)
    parser.add_argument("--print-top", type=int, default=10)
    parser.add_argument("--save-csv", type=str, default="", help="Se fornecido, salva CSV com eventos detectados")
    args = parser.parse_args()

    _print_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Configs
    cfg_data = yaml.safe_load(Path("configs/data.yaml").read_text())
    cfg_data = _interpolate_templates(cfg_data, cfg_data)
    cfg_model = yaml.safe_load(Path("configs/model_dt.yaml").read_text())
    cfg_trn = yaml.safe_load(Path("configs/trainer.yaml").read_text())

    # ---- Checkpoint path (NÃO carregar agora!)
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = _latest_ckpt(Path("artifacts"))
        if ckpt_path is None:
            raise FileNotFoundError("Nenhum checkpoint encontrado em artifacts/*.pt")
    print(f"[INFO] Using checkpoint: {ckpt_path.resolve()}")

    # ---- Constrói engine (passando **caminho**, não dict)
    engine = _make_stream_engine(ckpt_path, cfg_model, device)

    # ---- Dataset de avaliação (ordem determinística, sem shuffle)
    conf_val = SeqDatasetConfig(
        csv_path=str(Path(cfg_data["paths"]["test_csv"])),
        flow_keys=cfg_data["processing"]["flow_keys"],
        time_col=cfg_data["processing"]["time_col"],
        context_length=cfg_data["sequence"]["context_length"],
        start_action=cfg_data["sequence"]["start_action"],
        pad_token=cfg_data["sequence"]["pad_token"],
        normalize=cfg_data["processing"]["normalize"],
        label_col=cfg_data["labels"]["label_col"],
        attack_cat_col=cfg_data["labels"]["attack_cat_col"],
        categorical_cols=cfg_model["categorical"]["cols"],
    )
    ds_val = UNSWSequenceDataset(conf_val, max_rows=args.max_rows)
    dl_val = DataLoader(
        ds_val,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=seq_collate,
    )

    # ---- Varredura “streaming” por janelas
    cat_cols = cfg_model["categorical"]["cols"]
    rows_out: List[Tuple[int, float, int]] = []  # (idx_global, conf, y_true)

    idx_global = 0
    with torch.no_grad():
        for batch in dl_val:
            # Inferência
            if hasattr(engine, "infer_batch"):
                logits = engine.infer_batch(batch, cat_cols=cat_cols)  # fallback
            else:
                # Caso StreamingDT possua método forward_batch/logits
                # Tentativas comuns:
                tried = False
                for fn in ("infer_batch", "forward_batch", "logits"):
                    if hasattr(engine, fn):
                        logits = getattr(engine, fn)(batch, cat_cols=cat_cols)
                        tried = True
                        break
                if not tried:
                    # último recurso: reconstruir via fallback local
                    logits = _FallbackStreamingDT(ckpt_path, device, cfg_model).infer_batch(batch, cat_cols=cat_cols)

            # Decisão de evento
            event_mask, conf_max = _detect_events_from_logits(
                logits,
                batch["attn_mask"].to(device),
                wait_thr=float(args.wait_thr),
                class_cut=float(args.class_cut),
                mode=args.mode,
                tail_m=int(args.tail_m),
                tail_L=int(args.tail_L),
            )

            y_last = batch["actions_out"][:, -1].detach().cpu().numpy()  # rótulo do último token (apenas para referência)
            B = event_mask.shape[0]
            for i in range(B):
                rows_out.append((idx_global, float(conf_max[i]), int(y_last[i])))
                idx_global += 1

    # ---- Ordena por confiança e imprime top-N
    rows_out.sort(key=lambda t: t[1], reverse=True)
    topN = rows_out[: max(0, int(args.print_top))]
    if topN:
        print("\nTop events (by confidence):")
        print("idx\tconf\tlabel_last")
        for idx, conf, y in topN:
            print(f"{idx}\t{conf:.6f}\t{y}")

    # ---- Salvar CSV (opcional)
    if args.save_csv:
        import csv

        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "confidence", "y_last"])
            for r in rows_out:
                w.writerow(r)
        print(f"[OUT] saved events CSV to: {outp.resolve()}")


if __name__ == "__main__":
    main()
