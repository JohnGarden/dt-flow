from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from im12dt.models.model_dt import DecisionTransformer
from im12dt.models.temporal_embed import TimeEncodingFourier

# Importa componentes do seu projeto
from im12dt.models.tokens import (
    ActionTokenizer,
    CatSpec,
    RTGTokenizer,
    StateTokenizer,
    _rule_embed_dim,
)

# ---------- util: política 'wait' ----------


@torch.no_grad()
def decode_with_wait(logits: torch.Tensor, wait_threshold: float) -> torch.Tensor:
    """
    Converte logits (B,K,2) em ações {0,1,2} com 'wait' (2) quando a confiança máxima < wait_threshold.
    """
    probs = torch.softmax(logits, dim=-1)  # (B,K,2)
    conf, argmax = probs.max(dim=-1)  # (B,K)
    pred = argmax.clone()
    pred[conf < wait_threshold] = 2
    return pred


# ---------- util: load seguro do checkpoint (PyTorch 2.6+) ----------


def _safe_load_ckpt(path: str, map_location: torch.device) -> dict:
    """
    Carrega um checkpoint salvo no projeto (dict com 'model', tokenizers e stats),
    contornando o default `weights_only=True` e a allowlist de pickle do PyTorch 2.6.
    """
    # allowlist de tipos do NumPy que aparecem em stats
    import numpy as _np

    try:
        from torch.serialization import add_safe_globals

        add_safe_globals(
            [
                _np.core.multiarray.scalar,  # tipos escalares numpy
                _np.core.multiarray._reconstruct,  # reconstrução de arrays
            ]
        )
    except Exception:
        # Em versões sem add_safe_globals, ignoramos (raramente necessário)
        pass

    # Precisamos do dicionário completo → weights_only=False
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint inesperado: esperava dict com chaves ('model', 'state_tok', ...).")
    return ckpt


# ---------- configuração mínima para reconstrução ----------


@dataclass
class StreamingConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    n_actions: int
    # dims de embeddings
    state_embed_dim: int
    rtg_dim: int
    time_dim: int
    # categóricas
    cat_cols: List[str]
    cat_rule: str = "sqrt"
    cat_fixed_dim: int = 16


# ---------- classe principal de inferência streaming ----------


class StreamingDT(nn.Module):
    """
    Contém modelo + tokenizers e expõe um método de inferência por lote (batch dict
    com as mesmas chaves do collate do dataset: states, actions_in, rtg, delta_t,
    attn_mask e cat_* quando existir).
    """

    def __init__(
        self,
        model: DecisionTransformer,
        state_tok: StateTokenizer,
        action_tok: ActionTokenizer,
        rtg_tok: RTGTokenizer,
        time_tok: TimeEncodingFourier,
        device: torch.device,
        cat_cols: List[str],
    ):
        super().__init__()
        self.model = model
        self.state_tok = state_tok
        self.action_tok = action_tok
        self.rtg_tok = rtg_tok
        self.time_tok = time_tok
        self.device = device
        self.cat_cols = cat_cols

        self.to(device)
        self.eval()

    # ---------- fábrica a partir do checkpoint do treinamento ----------
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, model_cfg: dict, device: Optional[torch.device] = None) -> "StreamingDT":
        """
        Reconstrói modelo e tokenizers a partir de um checkpoint salvo pelo treinamento.
        Requer o mesmo `configs/model_dt.yaml` (passado em model_cfg).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = _safe_load_ckpt(ckpt_path, map_location=device)

        # Extrai config do modelo (usamos o que está em model_cfg para evitar drift)
        d_model = model_cfg["d_model"]
        n_layers = model_cfg["n_layers"]
        n_heads = model_cfg["n_heads"]
        d_ff = model_cfg["d_ff"]
        dropout = model_cfg.get("Dropout", 0.1)
        n_actions = model_cfg["vocab"]["n_actions"]

        E_state = model_cfg["embeddings"]["state_embed_dim"]
        E_rtg = model_cfg["embeddings"]["rtg_dim"]
        E_time = model_cfg["embeddings"]["time_dim"]

        cat_cols = list(model_cfg["categorical"]["cols"])
        cat_rule = model_cfg["categorical"].get("embed_rule", "sqrt")
        cat_fix = int(model_cfg["categorical"].get("fixed_dim", 16))

        # Número de atributos numéricos: usamos as stats do checkpoint
        norm_stats = ckpt.get("norm_stats", None)
        if norm_stats is None or "mean" not in norm_stats:
            raise ValueError("Checkpoint não contém 'norm_stats'. Treine e salve novamente (modelo + tokenizers + stats).")
        Dnum = int(norm_stats["mean"].shape[-1])

        # Cat maps salvos no treino (stoi)
        cat_maps = ckpt.get("cat_maps", None) or {}
        # monta CatSpec usando o stoi do treino
        cat_specs: List[CatSpec] = []
        for c in cat_cols:
            if c in cat_maps and isinstance(cat_maps[c], dict):
                n_tok = max(1, len(cat_maps[c]))
                d = _rule_embed_dim(n_tok, rule=cat_rule, fixed=cat_fix)
                cat_specs.append(CatSpec(name=c, n_tokens=n_tok, embed_dim=d))

        # Constrói tokenizers e modelo
        state_tok = StateTokenizer(Dnum, cat_specs, E_state)
        action_tok = ActionTokenizer(n_actions_plus_start=4, embed_dim=E_state)
        rtg_tok = RTGTokenizer(embed_dim=E_rtg)
        time_tok = TimeEncodingFourier(d_model=E_time, n_freq=16, use_log1p=True)

        model = DecisionTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            n_actions=n_actions,
        )

        # projeta embeddings heterogêneos para d_model
        model.ensure_projections(E_state, E_state, E_rtg, E_time, device=device)

        # Carrega pesos
        model.load_state_dict(ckpt["model"], strict=True)
        try:
            state_tok.load_state_dict(ckpt["state_tok"], strict=False)
            action_tok.load_state_dict(ckpt["action_tok"], strict=False)
            rtg_tok.load_state_dict(ckpt["rtg_tok"], strict=False)
            time_tok.load_state_dict(ckpt["time_tok"], strict=False)
        except Exception:
            # Em caso de pequenas diferenças no dict, seguimos com pesos do init
            pass

        inst = cls(
            model=model,
            state_tok=state_tok,
            action_tok=action_tok,
            rtg_tok=rtg_tok,
            time_tok=time_tok,
            device=device,
            cat_cols=cat_cols,
        )
        return inst

    # ---------- helpers ----------
    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in batch.items():
            out[k] = v.to(self.device) if torch.is_tensor(v) else v
        return out

    def _cats_from_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {c: batch[f"cat_{c}"] for c in self.cat_cols if f"cat_{c}" in batch}

    # ---------- API principal ----------
    @torch.inference_mode()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        wait_threshold: float = 0.55,
        return_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Recebe um batch no formato do seu DataLoader (collate) e retorna:
        - 'probs1': probabilidade da classe 1 por token (B,K)
        - 'pred_act': ação com wait {0,1,2} (B,K)
        - 'y': rótulos verdadeiros (se presente) (B,K)
        - 'mask': attn_mask (B,K)
        - opcionalmente 'logits'
        """
        batch = self._to_device(batch)
        cats = self._cats_from_batch(batch)

        s_emb = self.state_tok(batch["states"].float(), cats)
        a_emb = self.action_tok(batch["actions_in"])
        r_emb = self.rtg_tok(batch["rtg"].float())
        t_emb = self.time_tok(batch["delta_t"].float())

        logits = self.model(s_emb, a_emb, r_emb, t_emb, batch["attn_mask"])
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        probs1 = torch.softmax(logits, dim=-1)[..., 1]
        pred_act = decode_with_wait(logits, wait_threshold)

        out = {
            "probs1": probs1.detach(),
            "pred_act": pred_act.detach(),
            "mask": batch["attn_mask"].detach(),
        }
        if "actions_out" in batch:
            out["y"] = batch["actions_out"].detach()
        if return_logits:
            out["logits"] = logits.detach()
        return out
