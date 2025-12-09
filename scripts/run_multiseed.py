#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
import yaml


# =============================
#  Import robusto do baseline
# =============================
def _import_run_dt_baseline():
    """Importa scripts.run_dt_baseline de forma robusta.

    Funciona com:
      - python scripts/run_multiseed.py
      - python -m scripts.run_multiseed
    """
    try:
        import scripts.run_dt_baseline as rdb  # type: ignore

        return rdb
    except ModuleNotFoundError:
        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        try:
            import scripts.run_dt_baseline as rdb  # type: ignore

            return rdb
        except ModuleNotFoundError:
            rb_path = ROOT / "scripts" / "run_dt_baseline.py"
            if not rb_path.exists():
                raise
            spec = importlib.util.spec_from_file_location("run_dt_baseline", rb_path)
            if spec is None or spec.loader is None:
                raise
            rdb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rdb)  # type: ignore
            return rdb


# =============================
#  Utilidades
# =============================
def _parse_seeds(arg: str | None) -> List[int]:
    if arg is None or str(arg).strip() == "":
        return [42]
    if "," in arg:
        return [int(s.strip()) for s in arg.split(",") if s.strip()]
    # também permitir espaços
    return [int(s) for s in arg.split() if s.strip()]


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _list_artifacts(dirpath: Path) -> List[Path]:
    return sorted([p for p in dirpath.glob("*.pt") if p.is_file()], key=lambda p: p.stat().st_mtime)


def _latest_ckpt(dirpath: Path) -> Optional[Path]:
    arts = _list_artifacts(dirpath)
    return arts[-1] if arts else None


@contextlib.contextmanager
def _patch_trainer_yaml(max_epochs: Optional[int] = None, steps_per_epoch: Optional[int] = None):
    """Monkeypatch temporário de Path.read_text para sobrescrever configs/trainer.yaml
    com novos valores para training.max_epochs e training.steps_per_epoch.

    Seguro o suficiente para uso local e temporário durante a chamada de rdb.main().
    """
    if max_epochs is None and steps_per_epoch is None:
        # nada a fazer
        yield
        return

    orig_read_text = Path.read_text

    def patched_read_text(self: Path, *args, **kwargs):
        s = orig_read_text(self, *args, **kwargs)
        # normaliza path para comparar
        p_str = str(self).replace("\\", "/")
        if p_str.endswith("configs/trainer.yaml"):
            try:
                cfg = yaml.safe_load(s)
            except Exception:
                return s  # se der algo errado, deixa original
            # cria chaves se não existirem
            if "training" not in cfg:
                cfg["training"] = {}
            if max_epochs is not None:
                cfg["training"]["max_epochs"] = int(max_epochs)
            if steps_per_epoch is not None:
                cfg["training"]["steps_per_epoch"] = int(steps_per_epoch)
            # devolve yaml atualizado (preserva ordem básica)
            return yaml.safe_dump(cfg, sort_keys=False)
        return s

    try:
        Path.read_text = patched_read_text  # type: ignore[assignment]
        yield
    finally:
        Path.read_text = orig_read_text  # type: ignore[assignment]


def _rename_with_seed(ckpt_path: Path, seed: int) -> Path:
    """Renomeia o checkpoint para incluir a semente, preservando o diretório."""
    if not ckpt_path.exists():
        return ckpt_path
    parent = ckpt_path.parent
    stem = ckpt_path.stem  # ex: dt_baseline_20251110-221610
    suffix = ckpt_path.suffix
    new_name = f"{stem}_seed{seed}{suffix}"
    new_path = parent / new_name
    try:
        ckpt_path.replace(new_path)
        return new_path
    except Exception:
        return ckpt_path


# =============================
#  Programa principal
# =============================
def main():
    ap = argparse.ArgumentParser(description="Multi-seed runner for Decision Transformer baseline.")
    ap.add_argument("--seeds", type=str, default="42", help="Lista de sementes (ex: '42,43,44').")
    ap.add_argument("--epochs", type=int, default=None, help="Override de training.max_epochs (opcional).")
    ap.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Override de training.steps_per_epoch (opcional).",
    )
    ap.add_argument("--artifacts-dir", type=str, default="artifacts", help="Diretório de saída de checkpoints.")
    args = ap.parse_args()

    # Logs de device
    print(f"[DEVICE] torch={torch.__version__} | cuda={torch.cuda.is_available()} | build={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"[DEVICE] gpu={torch.cuda.get_device_name(0)} | cc={torch.cuda.get_device_capability(0)}")

    seeds = _parse_seeds(args.seeds)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Importa baseline
    rdb = _import_run_dt_baseline()

    # Loop de sementes
    for i, seed in enumerate(seeds, start=1):
        print(f"\n=== [Seed {seed}] {i}/{len(seeds)} =======================================")
        _set_global_seed(seed)

        before = set(_list_artifacts(artifacts_dir))

        # Para override de épocas/steps
        with _patch_trainer_yaml(max_epochs=args.epochs, steps_per_epoch=args.steps_per_epoch):
            # Chama o treinamento do baseline (o rdb.main() vai ler os YAMLs)
            rdb.main()

        # Descobre o novo checkpoint e renomeia para incluir seed
        after = set(_list_artifacts(artifacts_dir))
        new_files = sorted(list(after - before), key=lambda p: p.stat().st_mtime)
        if not new_files:
            # fallback: pega o mais recente
            latest = _latest_ckpt(artifacts_dir)
            if latest is not None:
                new_files = [latest]
        if new_files:
            final_ckpt = _rename_with_seed(new_files[-1], seed)
            print(f"[SEED {seed}] checkpoint: {final_ckpt}")
        else:
            print(f"[SEED {seed}] WARNING: no new checkpoint detected in {artifacts_dir}")

    print("\n[DONE] Multi-seed run finished.")


if __name__ == "__main__":
    main()
